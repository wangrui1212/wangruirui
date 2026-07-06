import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Presentation, PresentationFile } from "file:///C:/Users/YOSC/AppData/Local/Temp/codex-presentations/das-point-analysis-ppt/tmp/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const __filename = fileURLToPath(import.meta.url);
const PROJECT_ROOT = path.dirname(__filename);
const SUMMARY_CSV = path.join(PROJECT_ROOT, "outputs", "point_analysis_batch", "snr_summary.csv");
const FINAL_PPTX = path.join(PROJECT_ROOT, "outputs", "DAS点位分析图片与峰值SNR汇总_1000Hz对比版.pptx");
const QA_DIR = path.join(process.env.TEMP || PROJECT_ROOT, "codex-presentations", "das-point-analysis-ppt", "tmp", "qa");

const SLIDE_W = 1280;
const SLIDE_H = 720;
const COLORS = {
  canvas: "#FFFFFF",
  ink: "#000000",
  muted: "#555555",
  panel: "#EDEDED",
  rule: "#B8BCC4",
  accent: "#FF6B35",
};

function parseCsv(text) {
  const rows = [];
  let row = [];
  let cell = "";
  let inQuotes = false;
  const cleaned = text.replace(/^\uFEFF/, "");
  for (let i = 0; i < cleaned.length; i += 1) {
    const ch = cleaned[i];
    if (ch === '"') {
      if (inQuotes && cleaned[i + 1] === '"') {
        cell += '"';
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

function numeric(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function sortRows(rows) {
  return [...rows].sort((a, b) => {
    const sheetCompare = groupSortOrder(a) - groupSortOrder(b);
    if (sheetCompare !== 0) return sheetCompare;
    const samplingCompare = numeric(a.sampling_hz) - numeric(b.sampling_hz);
    if (samplingCompare !== 0) return samplingCompare;
    const freqCompare = numeric(a.frequency_hz) - numeric(b.frequency_hz);
    if (freqCompare !== 0) return freqCompare;
    return numeric(a.pzt_mvpp) - numeric(b.pzt_mvpp);
  });
}

function groupKey(row) {
  if (row.sheet.includes("光谷")) return "光谷互联40ns";
  if (row.sheet.includes("100")) return "100ns";
  return "40ns";
}

function groupSortOrder(row) {
  const key = groupKey(row);
  if (key === "100ns") return 1;
  if (key === "40ns") return 2;
  return 3;
}

function groupBySheet(rows) {
  const groups = new Map();
  for (const row of rows) {
    const key = groupKey(row);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(row);
  }
  return groups;
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
    fontSize: 20,
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
  addText(slide, title, { left: 42, top: 34, width: 930, height: 52 }, { fontSize: 36, bold: true });
  if (subtitle) {
    addText(slide, subtitle, { left: 972, top: 46, width: 266, height: 34 }, { fontSize: 18, color: COLORS.muted, alignment: "right" });
  }
  addRule(slide, 96);
}

async function imageBytes(imagePath) {
  const bytes = await fs.readFile(imagePath);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

function imageCaption(row) {
  return `${row.frequency_hz} Hz · PZT ${row.pzt_mvpp} mVpp · 点位 ${row.point} · SNR ${row.snr_db} dB`;
}

function chunk(items, size) {
  const chunks = [];
  for (let i = 0; i < items.length; i += size) chunks.push(items.slice(i, i + size));
  return chunks;
}

function freqLabel(value) {
  return `${Number(value).toString()} Hz`;
}

function pztLabel(value) {
  return `${Number(value).toString()} mVpp`;
}

function buildSnrMatrix(rows) {
  const pzts = [50, 100, 200, 400];
  const sheetOrder = ["100ns", "40ns", "光谷互联40ns"];
  const matrix = [["组别", "振动频率", ...pzts.map(pztLabel)]];
  for (const sheet of sheetOrder) {
    const sheetRows = rows.filter((row) => groupKey(row) === sheet);
    const freqs = [...new Set(sheetRows.map((row) => numeric(row.frequency_hz)))].sort((a, b) => a - b);
    for (const freq of freqs) {
      const values = pzts.map((pzt) => {
        const found = sheetRows.find(
          (row) =>
            numeric(row.frequency_hz) === freq &&
            numeric(row.pzt_mvpp) === pzt,
        );
        return found ? `${Number(found.snr_db).toFixed(2)}` : "";
      });
      matrix.push([sheet, freqLabel(freq), ...values]);
    }
  }
  return matrix;
}

function buildDeltaMatrix(rows) {
  const pzts = [50, 100, 200, 400];
  const baselineRows = rows.filter((row) => groupKey(row) === "光谷互联40ns");
  const compareGroups = ["100ns", "40ns"];
  const matrix = [["对比项", "振动频率", ...pzts.map(pztLabel)]];

  const baselineKeys = new Set(
    baselineRows.map((row) => `${numeric(row.frequency_hz)}|${numeric(row.pzt_mvpp)}`),
  );
  const sharedFreqs = [...new Set(baselineRows.map((row) => numeric(row.frequency_hz)))]
    .filter((freq) =>
      compareGroups.every((group) =>
        rows.some((row) => groupKey(row) === group && numeric(row.frequency_hz) === freq),
      ),
    )
    .sort((a, b) => a - b);

  for (const group of compareGroups) {
    for (const freq of sharedFreqs) {
      const values = pzts.map((pzt) => {
        if (!baselineKeys.has(`${freq}|${pzt}`)) return "";
        const base = baselineRows.find((row) => numeric(row.frequency_hz) === freq && numeric(row.pzt_mvpp) === pzt);
        const current = rows.find(
          (row) => groupKey(row) === group && numeric(row.frequency_hz) === freq && numeric(row.pzt_mvpp) === pzt,
        );
        if (!base || !current) return "";
        const delta = numeric(current.snr_db) - numeric(base.snr_db);
        return `${delta >= 0 ? "+" : ""}${delta.toFixed(2)}`;
      });
      if (values.some((value) => value !== "")) {
        matrix.push([`${group} - 光谷互联40ns`, freqLabel(freq), ...values]);
      }
    }
  }
  return matrix;
}

function average(rows, predicate) {
  const values = rows.filter(predicate).map((row) => numeric(row.snr_db));
  return values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
}

async function addImageSlide(presentation, groupName, rows, pageIndex, pageCount) {
  const slide = presentation.slides.add();
  slide.background.fill = COLORS.canvas;
  addSlideTitle(slide, `${groupName} 点位分析图片`, `${pageIndex + 1}/${pageCount}`);

  const frames = [
    { left: 42, top: 132, width: 578, height: 404 },
    { left: 660, top: 132, width: 578, height: 404 },
  ];
  for (let i = 0; i < rows.length; i += 1) {
    const row = rows[i];
    const frame = frames[i];
    slide.images.add({
      blob: await imageBytes(row.image),
      contentType: "image/png",
      alt: imageCaption(row),
      fit: "contain",
      position: frame,
    });
    addText(slide, imageCaption(row), { left: frame.left, top: frame.top + frame.height + 16, width: frame.width, height: 38 }, { fontSize: 18, bold: true });
    addText(slide, path.basename(row.image), { left: frame.left, top: frame.top + frame.height + 52, width: frame.width, height: 28 }, { fontSize: 13, color: COLORS.muted });
  }
}

async function main() {
  await fs.mkdir(path.dirname(FINAL_PPTX), { recursive: true });
  await fs.mkdir(QA_DIR, { recursive: true });

  const allRows = sortRows(parseCsv(await fs.readFile(SUMMARY_CSV, "utf8")).filter((row) => row.status.startsWith("ok")));
  const rows = allRows.filter((row) => numeric(row.sampling_hz) === 1000);
  const imageRows = rows.filter((row) => row.image && row.point);
  const presentation = Presentation.create({ slideSize: { width: SLIDE_W, height: SLIDE_H } });

  const titleSlide = presentation.slides.add();
  titleSlide.background.fill = COLORS.canvas;
  addText(titleSlide, "DAS 点位分析图片与峰值 SNR 汇总", { left: 42, top: 86, width: 1030, height: 150 }, { fontSize: 58, bold: true });
  addText(titleSlide, "基于 outputs/point_analysis_batch 中的 100ns 与 40ns 分组图片", { left: 42, top: 260, width: 980, height: 46 }, { fontSize: 24, color: COLORS.muted });
  addText(titleSlide, "说明：每张点位分析图片均取对应测试时间段中间 10 秒数据进行 FFT 分析；为保证对比一致，本PPT仅保留采样频率 1000 Hz 的记录。", { left: 42, top: 306, width: 1120, height: 34 }, { fontSize: 20, color: COLORS.ink });
  addRule(titleSlide, 344);
  const avg100 = average(rows, (row) => groupKey(row) === "100ns");
  const avg40 = average(rows, (row) => groupKey(row) === "40ns");
  const avgGg = average(rows, (row) => groupKey(row) === "光谷互联40ns");
  const metrics = [
    ["100ns 图片", `${imageRows.filter((row) => groupKey(row) === "100ns").length} 张`],
    ["40ns 图片", `${imageRows.filter((row) => groupKey(row) === "40ns").length} 张`],
    ["光谷记录", `${rows.filter((row) => groupKey(row) === "光谷互联40ns").length} 条`],
    ["100ns 平均 SNR", `${avg100.toFixed(2)} dB`],
    ["40ns 平均 SNR", `${avg40.toFixed(2)} dB`],
    ["光谷平均 SNR", `${avgGg.toFixed(2)} dB`],
  ];
  metrics.forEach(([label, value], index) => {
    const left = 42 + index * 196;
    slidePanel(titleSlide, left, 392, 178, 158);
    addText(titleSlide, label, { left: left + 14, top: 414, width: 150, height: 32 }, { fontSize: 15, color: COLORS.muted, bold: true });
    addText(titleSlide, value, { left: left + 14, top: 466, width: 150, height: 58 }, { fontSize: 27, bold: true });
  });

  const tableSlide = presentation.slides.add();
  tableSlide.background.fill = COLORS.canvas;
  addSlideTitle(tableSlide, "峰值谱线 SNR 汇总表", "单位: dB");
  const values = buildSnrMatrix(rows);
  const table = tableSlide.tables.add({
    rows: values.length,
    columns: values[0].length,
    left: 42,
    top: 124,
    width: 1196,
    height: 510,
    columnTracks: [
      { mode: "fixed", value: 130 },
      { mode: "fixed", value: 100 },
      { mode: "fr", value: 1 },
      { mode: "fr", value: 1 },
      { mode: "fr", value: 1 },
      { mode: "fr", value: 1 },
    ],
    values,
  });
  table.styleOptions = { headerRow: true, bandedRows: true };
  table.borders.assign({ style: "solid", fill: COLORS.rule, width: 1 });
  table.cells.block({ row: 0, column: 0, rowCount: 1, columnCount: values[0].length }).assign({
    fill: COLORS.panel,
    textStyle: { fontSize: 13, bold: true, color: COLORS.ink },
  });
  table.cells.block({ row: 1, column: 0, rowCount: values.length - 1, columnCount: values[0].length }).assign({
    textStyle: { fontSize: 11, color: COLORS.ink },
  });
  addText(tableSlide, "仅保留采样频率 1000 Hz 的可比记录；空白表示该组没有对应 PZT 幅值记录；光谷互联40ns按“特征频率峰值-底噪水平”计算。", { left: 42, top: 650, width: 1160, height: 28 }, { fontSize: 15, color: COLORS.muted });

  const deltaSlide = presentation.slides.add();
  deltaSlide.background.fill = COLORS.canvas;
  addSlideTitle(deltaSlide, "相对光谷互联40ns的峰值 SNR 差值表", "单位: dB");
  const deltaValues = buildDeltaMatrix(rows);
  const deltaTable = deltaSlide.tables.add({
    rows: deltaValues.length,
    columns: deltaValues[0].length,
    left: 42,
    top: 124,
    width: 1196,
    height: 390,
    columnTracks: [
      { mode: "fixed", value: 220 },
      { mode: "fixed", value: 110 },
      { mode: "fr", value: 1 },
      { mode: "fr", value: 1 },
      { mode: "fr", value: 1 },
      { mode: "fr", value: 1 },
    ],
    values: deltaValues,
  });
  deltaTable.styleOptions = { headerRow: true, bandedRows: true };
  deltaTable.borders.assign({ style: "solid", fill: COLORS.rule, width: 1 });
  deltaTable.cells.block({ row: 0, column: 0, rowCount: 1, columnCount: deltaValues[0].length }).assign({
    fill: COLORS.panel,
    textStyle: { fontSize: 15, bold: true, color: COLORS.ink },
  });
  deltaTable.cells.block({ row: 1, column: 0, rowCount: deltaValues.length - 1, columnCount: deltaValues[0].length }).assign({
    textStyle: { fontSize: 15, color: COLORS.ink },
  });
  addText(deltaSlide, "计算方式：新方案对应 SNR - 光谷互联40ns SNR；正值表示新方案高于光谷互联，负值表示低于光谷互联。仅比较三组共有的 1000 Hz 采样条件。", { left: 42, top: 560, width: 1120, height: 52 }, { fontSize: 19, color: COLORS.muted });

  const groups = groupBySheet(imageRows);
  for (const [groupName, groupRows] of groups.entries()) {
    addDividerSlide(presentation, groupName, groupRows);
    const pages = chunk(groupRows, 2);
    for (let index = 0; index < pages.length; index += 1) {
      await addImageSlide(presentation, groupName, pages[index], index, pages.length);
    }
  }

  for (const [index, slide] of presentation.slides.items.entries()) {
    const stem = `slide-${String(index + 1).padStart(2, "0")}`;
    const png = await presentation.export({ slide, format: "png", scale: 1 });
    await fs.writeFile(path.join(QA_DIR, `${stem}.png`), new Uint8Array(await png.arrayBuffer()));
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(path.join(QA_DIR, `${stem}.layout.json`), await layout.text(), "utf8");
  }
  const montage = await presentation.export({ format: "webp", montage: true, scale: 1 });
  await fs.writeFile(path.join(QA_DIR, "deck-montage.webp"), new Uint8Array(await montage.arrayBuffer()));

  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(FINAL_PPTX);
  console.log(JSON.stringify({ finalPptx: FINAL_PPTX, qaDir: QA_DIR, slides: presentation.slides.items.length }, null, 2));
}

function slidePanel(slide, left, top, width, height) {
  slide.shapes.add({
    geometry: "rect",
    position: { left, top, width, height },
    fill: COLORS.panel,
    line: { style: "solid", fill: "none", width: 0 },
  });
}

function addDividerSlide(presentation, groupName, rows) {
  const slide = presentation.slides.add();
  slide.background.fill = COLORS.canvas;
  addText(slide, `${groupName} 点位分析图片`, { left: 42, top: 104, width: 900, height: 92 }, { fontSize: 58, bold: true });
  addText(slide, `共 ${rows.length} 张图片 · FFT 分析时间 10 s · 点位 ${rows[0]?.point ?? ""}`, { left: 42, top: 228, width: 780, height: 42 }, { fontSize: 24, color: COLORS.muted });
  addRule(slide, 336);
  const freqs = [...new Set(rows.map((row) => Number(row.frequency_hz)))].sort((a, b) => a - b);
  addText(slide, `覆盖频率：${freqs.map((freq) => `${freq} Hz`).join(" / ")}`, { left: 42, top: 390, width: 1120, height: 40 }, { fontSize: 24, color: COLORS.ink });
  addText(slide, "后续页面按振动频率和 PZT 幅值顺序排列，每页最多放置两张分析图。", { left: 42, top: 460, width: 940, height: 40 }, { fontSize: 22, color: COLORS.muted });
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
