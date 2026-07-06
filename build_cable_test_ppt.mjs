import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Presentation, PresentationFile } from "file:///C:/Users/YOSC/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const __filename = fileURLToPath(import.meta.url);
const PROJECT_ROOT = path.dirname(__filename);
const SUMMARY_CSV = path.join(PROJECT_ROOT, "outputs", "cable_test", "point_analysis", "cable_point_snr_summary.csv");
const OUTPUT_PPTX = path.join(PROJECT_ROOT, "outputs", "cable_test", "缆线测试点位分析汇报.pptx");
const QA_DIR = path.join(PROJECT_ROOT, "outputs", "cable_test", "ppt_preview");

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

function safeNumber(value, digits = 2, suffix = "") {
  const parsed = numeric(value);
  return parsed === null ? "" : `${parsed.toFixed(digits)}${suffix}`;
}

function sortRows(rows) {
  return [...rows].sort((a, b) => Number(a.serial) - Number(b.serial));
}

function average(rows) {
  const values = rows.map((row) => numeric(row.peak_line_snr_db)).filter((value) => value !== null);
  return values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
}

function selectedWindow(row) {
  const files = String(row.selected_files || "").split(";").filter(Boolean);
  if (!files.length) return "";
  const times = files.map((file) => {
    const match = path.basename(file).match(/20260624(\d{2})(\d{2})(\d{2})/);
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
  addText(slide, title, { left: 42, top: 34, width: 860, height: 54 }, { fontSize: 36, bold: true });
  if (subtitle) {
    addText(slide, subtitle, { left: 890, top: 46, width: 348, height: 30 }, { fontSize: 18, color: COLORS.muted, alignment: "right" });
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
    addText(slide, `- ${bullet}`, { left, top: top + 52 + index * 60, width, height: 48 }, { fontSize: 20, color: COLORS.ink });
  });
}

function tableValues(rows) {
  return [
    ["序号", "频率", "音量", "原记录SNR", "点位SNR", "特征峰", "分析窗口"],
    ...rows.map((row) => [
      row.serial,
      `${row.frequency_hz} Hz`,
      row.volume || "",
      row.excel_snr_db ? safeNumber(row.excel_snr_db, 2, " dB") : "未识别",
      safeNumber(row.peak_line_snr_db, 2, " dB"),
      safeNumber(row.peak_freq_hz, 3, " Hz"),
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
    top: 126,
    width: 1196,
    height: 468,
    columnTracks: [
      { mode: "fixed", value: 62 },
      { mode: "fixed", value: 92 },
      { mode: "fixed", value: 78 },
      { mode: "fixed", value: 130 },
      { mode: "fixed", value: 120 },
      { mode: "fixed", value: 138 },
      { mode: "fr", value: 1 },
    ],
    values,
  });
  table.styleOptions = { headerRow: true, bandedRows: true };
  table.borders.assign({ style: "solid", fill: COLORS.rule, width: 1 });
  table.cells.block({ row: 0, column: 0, rowCount: 1, columnCount: values[0].length }).assign({
    fill: COLORS.panel,
    textStyle: { fontFace: "Microsoft YaHei", fontSize: 16, bold: true, color: COLORS.ink },
  });
  table.cells.block({ row: 1, column: 0, rowCount: values.length - 1, columnCount: values[0].length }).assign({
    textStyle: { fontFace: "Microsoft YaHei", fontSize: 16, color: COLORS.ink },
  });
}

async function readImageBlob(imagePath) {
  const bytes = await fs.readFile(imagePath);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

function rowLabel(row) {
  return `${row.frequency_hz} Hz / 音量 ${row.volume || "-"} / 点位 ${row.point_index} / SNR ${safeNumber(row.peak_line_snr_db, 2, " dB")}`;
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
  const bySerial = Object.fromEntries(rows.map((row) => [row.serial, row]));
  const maxSnr = Math.max(...rows.map((row) => Number(row.peak_line_snr_db)).filter(Number.isFinite));

  const presentation = Presentation.create({ slideSize: { width: SLIDE_W, height: SLIDE_H } });

  const titleSlide = presentation.slides.add();
  titleSlide.background.fill = COLORS.canvas;
  addText(titleSlide, "缆线测试点位分析汇报", { left: 42, top: 82, width: 980, height: 92 }, { fontSize: 58, bold: true });
  addText(titleSlide, "点位位置 1848.43 m；实际最近点位 181 / 1845.661 m；FFT 分析窗口 10 s", { left: 42, top: 220, width: 1120, height: 38 }, { fontSize: 24, color: COLORS.muted });
  addRule(titleSlide, 328);
  const metrics = [
    ["记录数", `${rows.length} 组`],
    ["采样率", "4000 Hz"],
    ["平均点位SNR", `${average(rows).toFixed(2)} dB`],
    ["最高SNR", `${maxSnr.toFixed(2)} dB`],
  ];
  metrics.forEach(([label, value], index) => {
    const left = 42 + index * 296;
    addPanel(titleSlide, left, 382, 260, 150);
    addText(titleSlide, label, { left: left + 18, top: 404, width: 220, height: 28 }, { fontSize: 18, color: COLORS.muted, bold: true });
    addText(titleSlide, value, { left: left + 18, top: 456, width: 220, height: 52 }, { fontSize: 36, bold: true });
  });

  const tableSlide = presentation.slides.add();
  tableSlide.background.fill = COLORS.canvas;
  addSlideTitle(tableSlide, "测试记录与峰值谱线 SNR 汇总", "单位: dB");
  addSummaryTable(tableSlide, rows);
  addText(tableSlide, "原记录SNR来自表格“峰-底噪”；点位SNR由 1848.43 m 附近点位的 10 秒 FFT 窗口重新计算。200Hz 组按目标频率 200+/-30Hz 复核。", { left: 42, top: 626, width: 1120, height: 40 }, { fontSize: 16, color: COLORS.muted });

  const conclusionSlide = presentation.slides.add();
  conclusionSlide.background.fill = COLORS.canvas;
  addSlideTitle(conclusionSlide, "分析结论");
  addBullets(
    conclusionSlide,
    "主要观察",
    [
      `10Hz、30Hz、50Hz均能稳定识别；50Hz 点位SNR为 ${safeNumber(bySerial["4"].peak_line_snr_db, 2, " dB")}，响应最清晰。`,
      `100Hz 重新在测试时间段内自动选取 10 秒窗口，特征峰 ${safeNumber(bySerial["5"].peak_freq_hz, 3, " Hz")}，SNR ${safeNumber(bySerial["5"].peak_line_snr_db, 2, " dB")}。`,
      `200Hz 按目标频率 200+/-30Hz 复核后，特征峰 ${safeNumber(bySerial["6"].peak_freq_hz, 3, " Hz")}，SNR ${safeNumber(bySerial["6"].peak_line_snr_db, 2, " dB")}。`,
      `500Hz 100% 音量采用 15:14:44 往后 10 秒窗口，特征峰 ${safeNumber(bySerial["7"].peak_freq_hz, 3, " Hz")}，SNR ${safeNumber(bySerial["7"].peak_line_snr_db, 2, " dB")}；50% 音量下SNR为 ${safeNumber(bySerial["10"].peak_line_snr_db, 2, " dB")}。`,
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
      "对已知激励频率建议同时保留自动识别和目标频率带宽复核，便于区分基频、谐波和偶发噪声峰。",
      "500Hz 建议继续做音量梯度测试，例如 30%、50%、70%、100%，确认饱和阈值和最佳激励幅度。",
      "低频段建议适当延长分析窗口或提高激励强度，1Hz/5Hz尤其需要更长时间窗才能稳定评价。",
      "每轮测试前后保留空白组，作为环境底噪漂移和系统状态变化的参考。",
      "固定点位、采样率、脉宽、音量和文件窗口选择规则，保证后续数据可以横向对比。",
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
