import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Presentation, PresentationFile } from "file:///C:/Users/YOSC/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const __filename = fileURLToPath(import.meta.url);
const PROJECT_ROOT = path.dirname(__filename);

const SELF_CSV = path.join(PROJECT_ROOT, "outputs", "point_analysis_20260630_40ns_200M_global_noise", "point_snr_summary.csv");
const GG_CSV = path.join(PROJECT_ROOT, "outputs", "point_analysis_guanggu_40ns_global_noise", "point_snr_summary.csv");
const OUTPUT_DIR = path.join(PROJECT_ROOT, "outputs", "comparison_40ns_200M");
const OUTPUT_PPTX = path.join(OUTPUT_DIR, "40ns_200M点位SNR对比_新方案_vs_光谷互联_含SNR曲线.pptx");
const OUTPUT_CSV = path.join(OUTPUT_DIR, "comparison_summary_global_noise.csv");
const FINAL_OUTPUT_PPTX = path.join(OUTPUT_DIR, "40ns_200M点位SNR对比_新方案_vs_光谷互联_全局底噪SNR.pptx");
const QA_DIR = path.join(OUTPUT_DIR, "ppt_preview_global_noise");

const SLIDE_W = 1280;
const SLIDE_H = 720;
const COLORS = {
  canvas: "#FFFFFF",
  ink: "#101828",
  muted: "#667085",
  panel: "#F7F8FA",
  rule: "#D0D5DD",
  accent: "#C2410C",
  green: "#D9F7E8",
  red: "#FEE4E2",
  yellow: "#FEF3C7",
  blue: "#EAF2FF",
  darkBlue: "#1D4ED8",
};

function parseCsv(text) {
  const rows = [];
  let row = [];
  let cell = "";
  let inQuotes = false;
  const cleaned = text.replace(/^\uFEFF/, "");
  for (let i = 0; i < cleaned.length; i += 1) {
    const ch = cleaned[i];
    if (ch === "\"") {
      if (inQuotes && cleaned[i + 1] === "\"") {
        cell += "\"";
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (ch === "," && !inQuotes) {
      row.push(cell);
      cell = "";
    } else if ((ch === "\n" || ch === "\r") && !inQuotes) {
      if (ch === "\r" && cleaned[i + 1] === "\n") i += 1;
      row.push(cell);
      if (row.some((value) => value.length > 0)) rows.push(row);
      row = [];
      cell = "";
    } else {
      cell += ch;
    }
  }
  if (cell.length || row.length) {
    row.push(cell);
    if (row.some((value) => value.length > 0)) rows.push(row);
  }
  const [headers, ...data] = rows;
  return data.map((values) => Object.fromEntries(headers.map((key, index) => [key, values[index] ?? ""])));
}

function csvEscape(value) {
  const text = String(value ?? "");
  if (/[",\r\n]/.test(text)) return `"${text.replace(/"/g, "\"\"")}"`;
  return text;
}

function toCsv(rows, headers) {
  return [
    headers.join(","),
    ...rows.map((row) => headers.map((header) => csvEscape(row[header])).join(",")),
  ].join("\n");
}

function num(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function fmt(value, digits = 2, suffix = "") {
  const n = num(value);
  return n === null ? "" : `${n.toFixed(digits)}${suffix}`;
}

function keyFor(row) {
  return [
    Number(row.frequency_hz),
    Number(row.pzt_mvpp),
    Number(row.pulse_width_ns),
    Number(row.sampling_hz),
  ].join("|");
}

function sortByParams(a, b) {
  return (
    Number(a.sampling_hz) - Number(b.sampling_hz)
    || Number(a.frequency_hz) - Number(b.frequency_hz)
    || Number(a.pzt_mvpp) - Number(b.pzt_mvpp)
  );
}

function average(rows, key) {
  const values = rows.map((row) => num(row[key])).filter((value) => value !== null);
  return values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
}

function groupAverage(rows, groupKey, valueKey) {
  const groups = new Map();
  for (const row of rows) {
    const key = row[groupKey];
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(row);
  }
  return [...groups.entries()].map(([key, groupRows]) => ({
    key,
    avg: average(groupRows, valueKey),
    count: groupRows.length,
  }));
}

function addText(slide, text, position, style = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position,
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    fontFace: "Microsoft YaHei",
    fontSize: 18,
    color: COLORS.ink,
    ...style,
  };
  return shape;
}

function addRule(slide, top) {
  slide.shapes.add({
    geometry: "rect",
    position: { left: 42, top, width: 1196, height: 1 },
    fill: COLORS.rule,
    line: { style: "solid", fill: "none", width: 0 },
  });
}

function addSlideTitle(slide, title, subtitle = "") {
  addText(slide, title, { left: 42, top: 32, width: 920, height: 54 }, { fontSize: 35, bold: true });
  if (subtitle) {
    addText(slide, subtitle, { left: 900, top: 45, width: 338, height: 28 }, { fontSize: 17, color: COLORS.muted, alignment: "right" });
  }
  addRule(slide, 94);
}

function addMetric(slide, label, value, left, top, width = 220, fill = COLORS.panel) {
  slide.shapes.add({
    geometry: "rect",
    position: { left, top, width, height: 114 },
    fill,
    line: { style: "solid", fill: COLORS.rule, width: 1 },
  });
  addText(slide, label, { left: left + 18, top: top + 18, width: width - 36, height: 26 }, { fontSize: 16, bold: true, color: COLORS.muted });
  addText(slide, value, { left: left + 18, top: top + 56, width: width - 36, height: 42 }, { fontSize: 30, bold: true, color: COLORS.ink });
}

function addBulletList(slide, items, left, top, width, fontSize = 18, gap = 50) {
  items.forEach((item, index) => {
    addText(slide, `- ${item}`, { left, top: top + index * gap, width, height: gap - 4 }, { fontSize, color: COLORS.ink });
  });
}

function addDataTable(slide, values, options) {
  const {
    left = 42,
    top = 120,
    width = 1196,
    rowHeight = 30,
    colWidths,
    headerFill = "#EEF2F6",
    fontSize = 12,
    maxRows = values.length,
    deltaColumn = -1,
  } = options;
  const rows = values.slice(0, maxRows);
  const totalHeight = rowHeight * rows.length;
  const table = slide.tables.add({
    rows: rows.length,
    columns: rows[0].length,
    left,
    top,
    width,
    height: totalHeight,
    columnTracks: colWidths.map((value) => ({ mode: "fixed", value })),
    values: rows,
  });
  table.styleOptions = { headerRow: true, bandedRows: false };
  table.borders.assign({ style: "solid", fill: COLORS.rule, width: 1 });
  table.cells.block({ row: 0, column: 0, rowCount: 1, columnCount: rows[0].length }).assign({
    fill: headerFill,
    textStyle: { fontFace: "Microsoft YaHei", fontSize, bold: true, color: COLORS.ink },
  });
  if (rows.length > 1) {
    table.cells.block({ row: 1, column: 0, rowCount: rows.length - 1, columnCount: rows[0].length }).assign({
      textStyle: { fontFace: "Microsoft YaHei", fontSize: fontSize - 1, color: COLORS.ink },
    });
  }
  if (deltaColumn >= 0) {
    for (let rowIndex = 1; rowIndex < rows.length; rowIndex += 1) {
      const delta = num(String(rows[rowIndex][deltaColumn]).replace(" dB", ""));
      const fill = delta === null ? COLORS.panel : delta >= 5 ? COLORS.green : delta <= -5 ? COLORS.red : COLORS.yellow;
      table.cells.block({ row: rowIndex, column: deltaColumn, rowCount: 1, columnCount: 1 }).assign({
        fill,
        textStyle: { fontFace: "Microsoft YaHei", fontSize: fontSize - 1, bold: true, color: COLORS.ink },
      });
    }
  }
  return table;
}

function addDeltaMatrix(slide, values, left, top, cellWidths, rowHeight) {
  const totalColumns = values[0].length;
  for (let rowIndex = 0; rowIndex < values.length; rowIndex += 1) {
    let currentLeft = left;
    for (let colIndex = 0; colIndex < totalColumns; colIndex += 1) {
      const value = values[rowIndex][colIndex];
      const width = cellWidths[colIndex];
      let fill = rowIndex === 0 ? "#EEF2F6" : "#FFFFFF";
      if (rowIndex > 0 && colIndex > 0) {
        const delta = num(String(value).replace(" dB", ""));
        if (delta === null) fill = "#F2F4F7";
        else if (delta >= 5) fill = COLORS.green;
        else if (delta <= -5) fill = COLORS.red;
        else fill = COLORS.yellow;
      }
      slide.shapes.add({
        geometry: "rect",
        position: { left: currentLeft, top: top + rowIndex * rowHeight, width, height: rowHeight },
        fill,
        line: { style: "solid", fill: COLORS.rule, width: 1 },
      });
      addText(slide, value, {
        left: currentLeft + 8,
        top: top + rowIndex * rowHeight + 11,
        width: width - 16,
        height: rowHeight - 14,
      }, {
        fontSize: rowIndex === 0 ? 17 : 18,
        bold: rowIndex === 0 || colIndex === 0,
        alignment: "center",
        color: COLORS.ink,
      });
      currentLeft += width;
    }
  }
}

async function readImageBlob(imagePath) {
  const bytes = await fs.readFile(imagePath);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

function makeComparisonRows(selfRows, ggRows) {
  const selfExported = selfRows.filter((row) => row.status === "exported");
  const ggExported = ggRows.filter((row) => row.status === "exported");
  const ggByKey = new Map(ggExported.map((row) => [keyFor(row), row]));
  const matched = [];
  const matchedKeys = new Set();

  for (const self of selfExported.sort(sortByParams)) {
    const key = keyFor(self);
    const gg = ggByKey.get(key);
    if (!gg) continue;
    matchedKeys.add(key);
    const delta = Number(self.peak_line_snr_db) - Number(gg.peak_line_snr_db);
    const excelDelta = Number(self.excel_snr_db) - Number(gg.excel_snr_db);
    matched.push({
      frequency_hz: self.frequency_hz,
      pzt_mvpp: self.pzt_mvpp,
      pulse_width_ns: self.pulse_width_ns,
      aom_mhz: "200",
      sampling_hz: self.sampling_hz,
      self_point_snr_db: fmt(self.peak_line_snr_db),
      guanggu_point_snr_db: fmt(gg.peak_line_snr_db),
      delta_point_snr_db: fmt(delta),
      self_peak_freq_hz: fmt(self.peak_freq_hz, 3),
      guanggu_peak_freq_hz: fmt(gg.peak_freq_hz, 3),
      self_global_noise_level_db: fmt(self.global_noise_level_db || self.noise_level_db),
      guanggu_global_noise_level_db: fmt(gg.global_noise_level_db || gg.noise_level_db),
      noise_mode: self.noise_mode || gg.noise_mode || "global",
      exclude_band_hz: self.exclude_band_hz || gg.exclude_band_hz || "5.000",
      self_excel_snr_db: fmt(self.excel_snr_db),
      guanggu_excel_snr_db: fmt(gg.excel_snr_db),
      delta_excel_snr_db: fmt(excelDelta),
      self_image: self.image,
      guanggu_image: gg.image,
      self_time: self.time_range || "",
      guanggu_time: gg.selected_file_time || "",
      guanggu_status_note: gg.status_note || "",
    });
  }

  const excluded = [];
  for (const row of selfExported) {
    if (!matchedKeys.has(keyFor(row))) {
      excluded.push({
        source: "新方案40ns/200M",
        frequency_hz: row.frequency_hz,
        pzt_mvpp: row.pzt_mvpp,
        sampling_hz: row.sampling_hz,
        reason: "光谷互联侧无相同实际采样率的有效记录",
      });
    }
  }
  for (const row of ggRows) {
    if (row.status !== "exported") {
      excluded.push({
        source: "光谷互联40ns/200M",
        frequency_hz: row.frequency_hz,
        pzt_mvpp: row.pzt_mvpp,
        sampling_hz: row.sampling_hz || row.sampling_hz_record,
        reason: row.status_note || "记录被跳过",
      });
    } else if (!matchedKeys.has(keyFor(row))) {
      const expected = num(row.frequency_hz);
      const peak = num(row.peak_freq_hz);
      const tolerance = expected === null ? 0 : Math.max(1, expected * 0.1);
      const mismatch = expected !== null && peak !== null && Math.abs(expected - peak) > tolerance;
      excluded.push({
        source: "光谷互联40ns/200M",
        frequency_hz: row.frequency_hz,
        pzt_mvpp: row.pzt_mvpp,
        sampling_hz: row.sampling_hz,
        reason: mismatch ? `自动峰值 ${fmt(row.peak_freq_hz, 3, " Hz")} 偏离目标频率` : "新方案侧无相同实际采样率的有效记录",
      });
    }
  }
  return { matched, excluded };
}

function tableRowsForSummary(rows) {
  return [
    ["频率", "PZT", "采样率", "新方案SNR", "光谷SNR", "差值", "新方案峰值", "光谷峰值"],
    ...rows.map((row) => [
      `${row.frequency_hz} Hz`,
      `${row.pzt_mvpp} mVpp`,
      `${row.sampling_hz} Hz`,
      `${row.self_point_snr_db} dB`,
      `${row.guanggu_point_snr_db} dB`,
      `${row.delta_point_snr_db} dB`,
      `${row.self_peak_freq_hz} Hz`,
      `${row.guanggu_peak_freq_hz} Hz`,
    ]),
  ];
}

function tableRowsForExcel(rows) {
  return [
    ["频率", "PZT", "采样率", "新方案原记录SNR", "光谷原记录SNR", "差值"],
    ...rows.map((row) => [
      `${row.frequency_hz} Hz`,
      `${row.pzt_mvpp} mVpp`,
      `${row.sampling_hz} Hz`,
      `${row.self_excel_snr_db} dB`,
      `${row.guanggu_excel_snr_db} dB`,
      `${row.delta_excel_snr_db} dB`,
    ]),
  ];
}

function tableRowsForExcluded(rows) {
  return [
    ["来源", "频率", "PZT", "实际采样率", "未纳入原因"],
    ...rows.map((row) => [
      row.source,
      `${row.frequency_hz} Hz`,
      `${row.pzt_mvpp} mVpp`,
      row.sampling_hz ? `${row.sampling_hz} Hz` : "",
      row.reason,
    ]),
  ];
}

function addSnrTrendSlide(presentation, freq, rowsForFreq) {
  const rows = [...rowsForFreq].sort((a, b) => Number(a.pzt_mvpp) - Number(b.pzt_mvpp));
  const categories = rows.map((row) => `${row.pzt_mvpp}`);
  const selfValues = rows.map((row) => Number(row.self_point_snr_db));
  const ggValues = rows.map((row) => Number(row.guanggu_point_snr_db));
  const allValues = [...selfValues, ...ggValues].filter((value) => Number.isFinite(value));
  const minY = Math.max(0, Math.floor((Math.min(...allValues) - 5) / 5) * 5);
  const maxY = Math.ceil((Math.max(...allValues) + 5) / 5) * 5;
  const avgDelta = average(rows, "delta_point_snr_db");

  const slide = presentation.slides.add();
  slide.background.fill = COLORS.canvas;
  addSlideTitle(slide, `${freq} Hz 频率组 SNR-电压曲线`, "横坐标：PZT幅值 mVpp；纵坐标：峰值谱线SNR dB");

  slide.charts.add("line", {
    position: { left: 72, top: 130, width: 820, height: 420 },
    title: `${freq} Hz SNR对比`,
    titleTextStyle: { fontSize: 18, fill: COLORS.ink, bold: true },
    categories,
    series: [
      {
        name: "新方案40ns/200M",
        values: selfValues,
        line: { style: "solid", fill: COLORS.darkBlue, width: 3 },
        marker: { symbol: "circle", size: 7 },
      },
      {
        name: "光谷互联40ns/200M",
        values: ggValues,
        line: { style: "solid", fill: COLORS.accent, width: 3 },
        marker: { symbol: "circle", size: 7 },
      },
    ],
    lineOptions: { grouping: "standard", smooth: false },
    legend: { position: "bottom", overlay: false, textStyle: { fontSize: 13, fill: COLORS.ink } },
    xAxis: {
      title: { text: "PZT幅值 (mVpp)", textStyle: { fontSize: 13, fill: COLORS.ink, bold: true } },
      textStyle: { fontSize: 12, fill: COLORS.ink },
      line: { style: "solid", fill: COLORS.rule, width: 1 },
    },
    yAxis: {
      title: { text: "SNR (dB)", textStyle: { fontSize: 13, fill: COLORS.ink, bold: true } },
      min: minY,
      max: maxY,
      majorUnit: 5,
      numberFormatCode: "0.0",
      textStyle: { fontSize: 12, fill: COLORS.ink },
      majorGridlines: { style: "solid", fill: "#E4E7EC", width: 1 },
      line: { style: "solid", fill: COLORS.rule, width: 1 },
    },
    chartFill: "#FFFFFF",
    plotAreaFill: "#FFFFFF",
    dataLabels: { showValue: true, position: "outEnd", textStyle: { fontSize: 10, fill: COLORS.ink } },
  });

  addMetric(slide, "本组平均差值", `${avgDelta.toFixed(2)} dB`, 940, 150, 240, avgDelta >= 0 ? COLORS.green : COLORS.red);
  const winner = avgDelta >= 0 ? "新方案整体更高" : "光谷互联整体更高";
  addText(slide, winner, { left: 940, top: 286, width: 240, height: 34 }, { fontSize: 22, bold: true, color: avgDelta >= 0 ? COLORS.darkBlue : COLORS.accent });
  const pointsText = rows
    .map((row) => `${row.pzt_mvpp}mVpp：${row.delta_point_snr_db}dB`)
    .join("\n");
  addText(slide, "差值明细（新方案 - 光谷互联）", { left: 940, top: 354, width: 260, height: 28 }, { fontSize: 17, bold: true });
  addText(slide, pointsText, { left: 940, top: 392, width: 260, height: 180 }, { fontSize: 16, color: COLORS.ink });
}

async function addComparisonImageSlide(presentation, row, index, total) {
  const slide = presentation.slides.add();
  slide.background.fill = COLORS.canvas;
  addSlideTitle(slide, `${row.frequency_hz} Hz / PZT ${row.pzt_mvpp} mVpp 对照`, `${index + 1}/${total}  |  ${row.sampling_hz} Hz采样`);

  const leftFrame = { left: 42, top: 132, width: 568, height: 390 };
  const rightFrame = { left: 670, top: 132, width: 568, height: 390 };
  slide.images.add({
    blob: await readImageBlob(row.self_image),
    contentType: "image/png",
    alt: `新方案 ${row.frequency_hz}Hz ${row.pzt_mvpp}mVpp`,
    fit: "contain",
    position: leftFrame,
  });
  slide.images.add({
    blob: await readImageBlob(row.guanggu_image),
    contentType: "image/png",
    alt: `光谷互联 ${row.frequency_hz}Hz ${row.pzt_mvpp}mVpp`,
    fit: "contain",
    position: rightFrame,
  });

  addText(slide, `新方案40ns/200M  SNR ${row.self_point_snr_db} dB`, { left: leftFrame.left, top: 548, width: leftFrame.width, height: 32 }, { fontSize: 20, bold: true, color: COLORS.darkBlue });
  addText(slide, `光谷互联40ns/200M  SNR ${row.guanggu_point_snr_db} dB`, { left: rightFrame.left, top: 548, width: rightFrame.width, height: 32 }, { fontSize: 20, bold: true, color: COLORS.accent });
  addText(slide, `差值：${row.delta_point_snr_db} dB（新方案 - 光谷互联）`, { left: 42, top: 600, width: 560, height: 28 }, { fontSize: 18, bold: true });
  addText(slide, `峰值频率：新方案 ${row.self_peak_freq_hz} Hz；光谷互联 ${row.guanggu_peak_freq_hz} Hz`, { left: 670, top: 600, width: 568, height: 28 }, { fontSize: 18, color: COLORS.muted });
}

async function writePreviews(presentation) {
  await fs.rm(QA_DIR, { recursive: true, force: true });
  await fs.mkdir(QA_DIR, { recursive: true });
  for (const [index, slide] of presentation.slides.items.entries()) {
    const png = await presentation.export({ slide, format: "png", scale: 1 });
    await fs.writeFile(path.join(QA_DIR, `slide-${String(index + 1).padStart(2, "0")}.png`), new Uint8Array(await png.arrayBuffer()));
  }
  const montage = await presentation.export({ format: "webp", montage: true, scale: 1 });
  await fs.writeFile(path.join(QA_DIR, "montage.webp"), new Uint8Array(await montage.arrayBuffer()));
}

async function main() {
  await fs.mkdir(OUTPUT_DIR, { recursive: true });
  await fs.rm(QA_DIR, { recursive: true, force: true });
  await fs.mkdir(QA_DIR, { recursive: true });

  const selfRows = parseCsv(await fs.readFile(SELF_CSV, "utf8"));
  const ggRows = parseCsv(await fs.readFile(GG_CSV, "utf8"));
  const { matched, excluded } = makeComparisonRows(selfRows, ggRows);

  await fs.writeFile(OUTPUT_CSV, toCsv(matched, [
    "frequency_hz",
    "pzt_mvpp",
    "pulse_width_ns",
    "aom_mhz",
    "sampling_hz",
    "self_point_snr_db",
    "guanggu_point_snr_db",
    "delta_point_snr_db",
    "self_peak_freq_hz",
    "guanggu_peak_freq_hz",
    "self_global_noise_level_db",
    "guanggu_global_noise_level_db",
    "noise_mode",
    "exclude_band_hz",
    "self_excel_snr_db",
    "guanggu_excel_snr_db",
    "delta_excel_snr_db",
    "self_time",
    "guanggu_time",
    "self_image",
    "guanggu_image",
  ]), "utf8");

  const selfAvg = average(matched, "self_point_snr_db");
  const ggAvg = average(matched, "guanggu_point_snr_db");
  const deltaAvg = average(matched, "delta_point_snr_db");
  const selfWins = matched.filter((row) => Number(row.delta_point_snr_db) > 0).length;
  const ggWins = matched.filter((row) => Number(row.delta_point_snr_db) < 0).length;
  const byFreq = groupAverage(matched, "frequency_hz", "delta_point_snr_db").sort((a, b) => Number(a.key) - Number(b.key));

  const presentation = Presentation.create({ slideSize: { width: SLIDE_W, height: SLIDE_H } });

  const titleSlide = presentation.slides.add();
  titleSlide.background.fill = COLORS.canvas;
  addText(titleSlide, "40ns / 200MHz 点位 SNR 对比", { left: 42, top: 80, width: 1100, height: 76 }, { fontSize: 50, bold: true });
  addText(titleSlide, "新方案 DAS 与光谷互联设备的相同参数试验对照", { left: 42, top: 170, width: 1100, height: 38 }, { fontSize: 24, color: COLORS.muted });
  addText(titleSlide, "全局底噪口径：FFT后排除目标频率总宽5Hz范围，剩余所有频点dB均值作为底噪", { left: 42, top: 214, width: 1100, height: 28 }, { fontSize: 18, color: COLORS.accent, bold: true });
  addRule(titleSlide, 248);
  addMetric(titleSlide, "严格匹配组数", `${matched.length} 组`, 42, 300, 220, COLORS.blue);
  addMetric(titleSlide, "新方案平均SNR", `${selfAvg.toFixed(2)} dB`, 302, 300, 220);
  addMetric(titleSlide, "光谷平均SNR", `${ggAvg.toFixed(2)} dB`, 562, 300, 220);
  addMetric(titleSlide, "平均差值", `${deltaAvg.toFixed(2)} dB`, 822, 300, 220, deltaAvg >= 0 ? COLORS.green : COLORS.red);
  addText(titleSlide, "对照口径：频率、PZT幅值、脉宽、AOM频移、实际采样频率均一致；FFT点位分析时间按已有结果口径为10s。", { left: 42, top: 470, width: 1120, height: 38 }, { fontSize: 19, color: COLORS.ink });
  addText(titleSlide, "主指标采用图片分析导出的峰值谱线SNR，差值均为“新方案 - 光谷互联”。", { left: 42, top: 516, width: 1120, height: 34 }, { fontSize: 19, color: COLORS.ink });

  const summarySlide = presentation.slides.add();
  summarySlide.background.fill = COLORS.canvas;
  addSlideTitle(summarySlide, "严格匹配后共有16组可直接对照", "主指标：峰值谱线SNR");
  addDataTable(summarySlide, tableRowsForSummary(matched), {
    left: 32,
    top: 116,
    width: 1216,
    rowHeight: 29,
    colWidths: [78, 100, 98, 140, 140, 106, 148, 148],
    fontSize: 12,
    deltaColumn: 5,
  });

  const deltaSlide = presentation.slides.add();
  deltaSlide.background.fill = COLORS.canvas;
  addSlideTitle(deltaSlide, "差值显示新方案在低频和中频多数占优", "绿色为新方案高于光谷互联，红色为低于光谷互联");
  const deltaRows = [
    ["频率", "PZT 50", "PZT 100", "PZT 200", "PZT 400"],
  ];
  const freqOrder = ["1", "5", "10", "30", "500"];
  for (const freq of freqOrder) {
    const row = [`${freq} Hz`];
    for (const pzt of ["50", "100", "200", "400"]) {
      const found = matched.find((item) => item.frequency_hz === freq && item.pzt_mvpp === pzt);
      row.push(found ? `${found.delta_point_snr_db} dB` : "");
    }
    deltaRows.push(row);
  }
  addDeltaMatrix(deltaSlide, deltaRows, 130, 136, [150, 210, 210, 210, 210], 58);
  addBulletList(deltaSlide, [
    `新方案在 ${selfWins} / ${matched.length} 组中点位SNR更高。`,
    "500Hz高采样率分组中光谷互联优势明显，提示高频工况还需要单独优化。",
    "30Hz / 200mVpp 已采用光谷互联 11:02:24 修正版，避免旧文件自动峰值偏到 5Hz 附近。",
  ], 130, 500, 1020, 20, 42);

  for (const freq of freqOrder) {
    const rowsForFreq = matched.filter((row) => row.frequency_hz === freq);
    if (rowsForFreq.length) addSnrTrendSlide(presentation, freq, rowsForFreq);
  }

  const conclusionSlide = presentation.slides.add();
  conclusionSlide.background.fill = COLORS.canvas;
  addSlideTitle(conclusionSlide, "主结论集中在低频稳定性和高频差距", "基于16组严格匹配试验");
  addMetric(conclusionSlide, "新方案平均SNR", `${selfAvg.toFixed(2)} dB`, 62, 132, 250, COLORS.blue);
  addMetric(conclusionSlide, "光谷互联平均SNR", `${ggAvg.toFixed(2)} dB`, 354, 132, 250);
  addMetric(conclusionSlide, "平均差值", `${deltaAvg.toFixed(2)} dB`, 646, 132, 250, deltaAvg >= 0 ? COLORS.green : COLORS.red);
  addMetric(conclusionSlide, "新方案占优", `${selfWins} 组`, 938, 132, 250, COLORS.green);
  const freqText = byFreq.map((item) => `${item.key}Hz：${item.avg.toFixed(2)}dB`).join("；");
  addText(conclusionSlide, "分频段平均差值", { left: 62, top: 306, width: 280, height: 30 }, { fontSize: 22, bold: true });
  addText(conclusionSlide, freqText, { left: 62, top: 350, width: 1120, height: 52 }, { fontSize: 21, color: COLORS.ink });
  addText(conclusionSlide, "一句话结论", { left: 62, top: 444, width: 280, height: 32 }, { fontSize: 24, bold: true, color: COLORS.accent });
  addText(conclusionSlide, `在相同参数的16组对照中，新方案40ns/200M平均峰值谱线SNR为${selfAvg.toFixed(2)} dB，光谷互联40ns/200M为${ggAvg.toFixed(2)} dB，整体新方案略高${deltaAvg.toFixed(2)} dB；但500Hz高频分组光谷互联更强，后续应重点补测和优化高频响应。`, { left: 62, top: 490, width: 1120, height: 92 }, { fontSize: 22, color: COLORS.ink });

  for (let index = 0; index < matched.length; index += 1) {
    await addComparisonImageSlide(presentation, matched[index], index, matched.length);
  }

  const excelSlide = presentation.slides.add();
  excelSlide.background.fill = COLORS.canvas;
  addSlideTitle(excelSlide, "原记录SNR作为辅助对照", "Excel记录：特征频率峰值 - 底噪水平");
  addDataTable(excelSlide, tableRowsForExcel(matched), {
    left: 106,
    top: 118,
    width: 1068,
    rowHeight: 30,
    colWidths: [100, 120, 120, 210, 210, 150],
    fontSize: 13,
    deltaColumn: 5,
  });

  const excludedSlide = presentation.slides.add();
  excludedSlide.background.fill = COLORS.canvas;
  addSlideTitle(excludedSlide, "未纳入主对照的记录单独保留", "原因包括采样率不一致、缺少文件或自动峰值偏离");
  const excludedRows = tableRowsForExcluded(excluded);
  addDataTable(excludedSlide, excludedRows.slice(0, 16), {
    left: 38,
    top: 116,
    width: 1204,
    rowHeight: 32,
    colWidths: [190, 86, 100, 112, 716],
    fontSize: 12,
  });
  if (excludedRows.length > 16) {
    addText(excludedSlide, `另有 ${excludedRows.length - 16} 条未显示，完整清单保留在 comparison_summary.csv 旁的源CSV中。`, { left: 38, top: 650, width: 900, height: 26 }, { fontSize: 16, color: COLORS.muted });
  }

  await writePreviews(presentation);
  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(FINAL_OUTPUT_PPTX);

  console.log(`matched=${matched.length}`);
  console.log(`self_avg=${selfAvg.toFixed(2)}`);
  console.log(`guanggu_avg=${ggAvg.toFixed(2)}`);
  console.log(`delta_avg=${deltaAvg.toFixed(2)}`);
  console.log(FINAL_OUTPUT_PPTX);
  console.log(OUTPUT_CSV);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
