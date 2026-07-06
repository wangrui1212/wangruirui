import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Presentation, PresentationFile } from "file:///C:/Users/YOSC/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const __filename = fileURLToPath(import.meta.url);
const PROJECT_ROOT = path.dirname(__filename);
const SUMMARY_CSV = path.join(PROJECT_ROOT, "outputs", "point_analysis_20260630_40ns_200M", "point_snr_summary.csv");
const OUTPUT_PPTX = path.join(PROJECT_ROOT, "outputs", "point_analysis_20260630_40ns_200M", "DAS点位分析汇报_20260630_40ns_200M频移.pptx");
const QA_DIR = path.join(PROJECT_ROOT, "outputs", "point_analysis_20260630_40ns_200M", "ppt_preview");

const SLIDE_W = 1280;
const SLIDE_H = 720;
const COLORS = {
  canvas: "#FFFFFF",
  ink: "#111111",
  muted: "#555555",
  panel: "#F2F4F7",
  rule: "#B8BCC4",
  accent: "#D94F2B",
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

function numeric(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function fmt(value, digits = 2, suffix = "") {
  const n = numeric(value);
  return n === null ? "" : `${n.toFixed(digits)}${suffix}`;
}

function sortRows(rows) {
  return [...rows].sort((a, b) => Number(a.serial) - Number(b.serial));
}

function average(rows, key = "peak_line_snr_db") {
  const values = rows.map((row) => numeric(row[key])).filter((value) => value !== null);
  return values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
}

function maxRow(rows, key = "peak_line_snr_db") {
  return rows.reduce((best, row) => (numeric(row[key]) > numeric(best[key]) ? row : best), rows[0]);
}

function selectedWindow(row) {
  const files = String(row.selected_files || "").split(";").filter(Boolean);
  if (!files.length) return "";
  const times = files.map((file) => {
    const match = path.basename(file).match(/20260630(\d{2})(\d{2})(\d{2})/);
    return match ? `${match[1]}:${match[2]}:${match[3]}` : "";
  }).filter(Boolean);
  return times.length ? `${times[0]}-${times[times.length - 1]}` : "";
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
  addText(slide, title, { left: 42, top: 34, width: 870, height: 54 }, { fontSize: 36, bold: true });
  if (subtitle) {
    addText(slide, subtitle, { left: 900, top: 46, width: 338, height: 30 }, { fontSize: 18, color: COLORS.muted, alignment: "right" });
  }
  addRule(slide, 96);
}

function addPanel(slide, left, top, width, height) {
  slide.shapes.add({
    geometry: "rect",
    position: { left, top, width, height },
    fill: COLORS.panel,
    line: { style: "solid", fill: "none", width: 0 },
  });
}

function addBullets(slide, title, bullets, left, top, width) {
  addText(slide, title, { left, top, width, height: 34 }, { fontSize: 24, bold: true });
  bullets.forEach((bullet, index) => {
    addText(slide, `- ${bullet}`, { left, top: top + 52 + index * 58, width, height: 48 }, { fontSize: 20, color: COLORS.ink });
  });
}

function tableValues(rows) {
  return [
    ["序号", "频率", "PZT", "原记录SNR", "点位SNR", "特征峰", "采样率", "窗口"],
    ...rows.map((row) => [
      row.serial,
      `${row.frequency_hz} Hz`,
      `${row.pzt_mvpp} mVpp`,
      fmt(row.excel_snr_db, 2, " dB"),
      fmt(row.peak_line_snr_db, 2, " dB"),
      fmt(row.peak_freq_hz, 3, " Hz"),
      `${row.sampling_hz} Hz`,
      selectedWindow(row),
    ]),
  ];
}

function addSummaryTable(slide, rows) {
  const values = tableValues(rows);
  const table = slide.tables.add({
    rows: values.length,
    columns: values[0].length,
    left: 42,
    top: 118,
    width: 1196,
    height: 510,
    columnTracks: [
      { mode: "fixed", value: 52 },
      { mode: "fixed", value: 78 },
      { mode: "fixed", value: 96 },
      { mode: "fixed", value: 112 },
      { mode: "fixed", value: 106 },
      { mode: "fixed", value: 118 },
      { mode: "fixed", value: 92 },
      { mode: "fr", value: 1 },
    ],
    values,
  });
  table.styleOptions = { headerRow: true, bandedRows: true };
  table.borders.assign({ style: "solid", fill: COLORS.rule, width: 1 });
  table.cells.block({ row: 0, column: 0, rowCount: 1, columnCount: values[0].length }).assign({
    fill: COLORS.panel,
    textStyle: { fontFace: "Microsoft YaHei", fontSize: 12, bold: true, color: COLORS.ink },
  });
  table.cells.block({ row: 1, column: 0, rowCount: values.length - 1, columnCount: values[0].length }).assign({
    textStyle: { fontFace: "Microsoft YaHei", fontSize: 10.5, color: COLORS.ink },
  });
}

async function readImageBlob(imagePath) {
  const bytes = await fs.readFile(imagePath);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

function rowLabel(row) {
  return `${row.frequency_hz} Hz / PZT ${row.pzt_mvpp} mVpp / 点位 ${row.point_index} / SNR ${fmt(row.peak_line_snr_db, 2, " dB")}`;
}

function chunk(items, size) {
  const result = [];
  for (let i = 0; i < items.length; i += size) result.push(items.slice(i, i + size));
  return result;
}

async function addImageSlide(presentation, rows, pageIndex, pageCount) {
  const slide = presentation.slides.add();
  slide.background.fill = COLORS.canvas;
  addSlideTitle(slide, "点位分析图片", `${pageIndex + 1}/${pageCount}`);
  const frames = [
    { left: 42, top: 132, width: 578, height: 404 },
    { left: 660, top: 132, width: 578, height: 404 },
  ];
  for (let index = 0; index < rows.length; index += 1) {
    const row = rows[index];
    const frame = frames[index];
    slide.images.add({
      blob: await readImageBlob(row.image),
      contentType: "image/png",
      alt: rowLabel(row),
      fit: "contain",
      position: frame,
    });
    addText(slide, rowLabel(row), { left: frame.left, top: frame.top + frame.height + 14, width: frame.width, height: 32 }, { fontSize: 18, bold: true });
    addText(slide, `${selectedWindow(row)}  ${row.status_note || ""}`, { left: frame.left, top: frame.top + frame.height + 48, width: frame.width, height: 56 }, { fontSize: 16, color: COLORS.muted });
  }
}

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

async function main() {
  await fs.mkdir(path.dirname(OUTPUT_PPTX), { recursive: true });
  await fs.mkdir(QA_DIR, { recursive: true });
  const rows = sortRows(parseCsv(await fs.readFile(SUMMARY_CSV, "utf8")).filter((row) => row.status === "exported"));
  const rows1000 = rows.filter((row) => Number(row.sampling_hz) === 1000);
  const rows4000 = rows.filter((row) => Number(row.sampling_hz) === 4000);
  const best = maxRow(rows);
  const pzt100 = rows.filter((row) => Number(row.pzt_mvpp) === 100);
  const pzt200 = rows.filter((row) => Number(row.pzt_mvpp) === 200);

  const presentation = Presentation.create({ slideSize: { width: SLIDE_W, height: SLIDE_H } });

  const titleSlide = presentation.slides.add();
  titleSlide.background.fill = COLORS.canvas;
  addText(titleSlide, "DAS点位分析汇报", { left: 42, top: 82, width: 980, height: 92 }, { fontSize: 58, bold: true });
  addText(titleSlide, "2026.06.30  |  40ns脉宽、200M频移  |  FFT分析窗口 10 s", { left: 42, top: 210, width: 1120, height: 38 }, { fontSize: 24, color: COLORS.muted });
  addText(titleSlide, `目标位置 8141.32 m；实际最近点位 ${rows[0].point_index} / ${fmt(rows[0].distance_m, 3, " m")}；误差 ${fmt(rows[0].distance_error_m, 3, " m")}`, { left: 42, top: 258, width: 1120, height: 34 }, { fontSize: 22, color: COLORS.muted });
  addRule(titleSlide, 334);
  const metrics = [
    ["记录数", `${rows.length} 组`],
    ["1000Hz采样", `${rows1000.length} 组`],
    ["4000Hz采样", `${rows4000.length} 组`],
    ["平均点位SNR", `${average(rows).toFixed(2)} dB`],
    ["最高SNR", `${fmt(best.peak_line_snr_db, 2, " dB")}`],
  ];
  metrics.forEach(([label, value], index) => {
    const left = 42 + index * 238;
    addPanel(titleSlide, left, 392, 208, 140);
    addText(titleSlide, label, { left: left + 16, top: 412, width: 176, height: 28 }, { fontSize: 17, color: COLORS.muted, bold: true });
    addText(titleSlide, value, { left: left + 16, top: 462, width: 176, height: 48 }, { fontSize: 30, bold: true });
  });

  const tableSlide = presentation.slides.add();
  tableSlide.background.fill = COLORS.canvas;
  addSlideTitle(tableSlide, "测试记录与峰值谱线 SNR 汇总", "单位: dB");
  addSummaryTable(tableSlide, rows);
  addText(tableSlide, "原记录SNR=表格特征频率峰值-底噪水平；点位SNR由 8141.32 m 附近点位的中间 10 秒窗口重新计算。500Hz记录采用4000Hz采样，其余记录为1000Hz采样。", { left: 42, top: 644, width: 1150, height: 38 }, { fontSize: 16, color: COLORS.muted });

  const conclusionSlide = presentation.slides.add();
  conclusionSlide.background.fill = COLORS.canvas;
  addSlideTitle(conclusionSlide, "分析结论");
  addBullets(
    conclusionSlide,
    "主要观察",
    [
      `200M频移、40ns配置下，全部22组均完成10秒点位FFT分析，平均点位SNR为 ${average(rows).toFixed(2)} dB。`,
      `1Hz至100Hz的自动识别峰值均贴近目标频率；500Hz三组在4000Hz采样下也稳定识别到约500Hz。`,
      `最高点位SNR出现在 ${best.frequency_hz}Hz / PZT ${best.pzt_mvpp}mVpp，SNR为 ${fmt(best.peak_line_snr_db, 2, " dB")}。`,
      `PZT 100mVpp与200mVpp整体信噪表现较好，平均SNR分别为 ${average(pzt100).toFixed(2)} dB 和 ${average(pzt200).toFixed(2)} dB。`,
    ],
    42,
    132,
    1130,
  );

  const suggestionSlide = presentation.slides.add();
  suggestionSlide.background.fill = COLORS.canvas;
  addSlideTitle(suggestionSlide, "后续试验建议");
  addBullets(
    suggestionSlide,
    "建议动作",
    [
      "保留200M频移配置继续与此前40ns/80M频移数据横向对比，重点看低频段和小PZT幅值下的稳定性差异。",
      "对已知激励频率建议同时保留自动识别和目标频率带宽复核，便于区分基频、谐波和偶发噪声峰。",
      "500Hz测试已经切换到4000Hz采样，后续对比表中应与1000Hz采样记录分开说明。",
      "建议补充空白组或无激励段，作为同点位底噪漂移和环境干扰参考。",
      "后续PZT幅值可重点围绕100mVpp和200mVpp加密测试，同时保留50mVpp作为低幅值边界。"
    ],
    42,
    132,
    1130,
  );

  const pages = chunk(rows, 2);
  for (let index = 0; index < pages.length; index += 1) {
    await addImageSlide(presentation, pages[index], index, pages.length);
  }

  for (const [index, slide] of presentation.slides.items.entries()) {
    const stem = `slide-${String(index + 1).padStart(2, "0")}`;
    await writeBlob(path.join(QA_DIR, `${stem}.png`), await presentation.export({ slide, format: "png", scale: 1 }));
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(path.join(QA_DIR, `${stem}.layout.json`), await layout.text(), "utf8");
  }
  await writeBlob(path.join(QA_DIR, "deck-montage.webp"), await presentation.export({ format: "webp", montage: true, scale: 1 }));

  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(OUTPUT_PPTX);
  console.log(JSON.stringify({ output: OUTPUT_PPTX, qaDir: QA_DIR, slides: presentation.slides.items.length }, null, 2));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
