import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Presentation, PresentationFile } from "file:///C:/Users/YOSC/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const __filename = fileURLToPath(import.meta.url);
const PROJECT_ROOT = path.dirname(__filename);
const SUMMARY_CSV = path.join(PROJECT_ROOT, "outputs", "point_analysis_guanggu_40ns", "point_snr_summary.csv");
const OUTPUT_PPTX = path.join(PROJECT_ROOT, "outputs", "point_analysis_guanggu_40ns", "光谷互联40ns点位分析汇报_110224修正版.pptx");
const QA_DIR = path.join(PROJECT_ROOT, "outputs", "point_analysis_guanggu_40ns", "ppt_preview");

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

function frequencyMismatchRows(rows) {
  return rows.filter((row) => {
    const expected = numeric(row.frequency_hz);
    const peak = numeric(row.peak_freq_hz);
    if (expected === null || peak === null) return false;
    const tolerance = Math.max(1, expected * 0.1);
    return Math.abs(expected - peak) > tolerance;
  });
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
    ["序号", "频率", "PZT", "原记录SNR", "点位SNR", "自动峰", "采样率", "选中文件时间", "状态"],
    ...rows.map((row) => [
      row.serial,
      `${row.frequency_hz} Hz`,
      `${row.pzt_mvpp} mVpp`,
      fmt(row.excel_snr_db, 2, " dB"),
      row.peak_line_snr_db ? fmt(row.peak_line_snr_db, 2, " dB") : "",
      row.peak_freq_hz ? fmt(row.peak_freq_hz, 3, " Hz") : "",
      row.sampling_hz || row.sampling_hz_record ? `${row.sampling_hz || row.sampling_hz_record} Hz` : "",
      row.selected_file_time ? row.selected_file_time.slice(11) : "",
      row.status === "exported" ? "完成" : "跳过",
    ]),
  ];
}

function skipReason(row) {
  const delta = row.time_delta_s || "";
  return `记录时间附近缺少可用文件，最近文件相差 ${delta}s，超过60s阈值`;
}

function addSummaryTable(slide, rows) {
  const values = tableValues(rows);
  const table = slide.tables.add({
    rows: values.length,
    columns: values[0].length,
    left: 34,
    top: 116,
    width: 1212,
    height: 520,
    columnTracks: [
      { mode: "fixed", value: 44 },
      { mode: "fixed", value: 70 },
      { mode: "fixed", value: 84 },
      { mode: "fixed", value: 104 },
      { mode: "fixed", value: 96 },
      { mode: "fixed", value: 112 },
      { mode: "fixed", value: 80 },
      { mode: "fixed", value: 142 },
      { mode: "fr", value: 1 },
    ],
    values,
  });
  table.styleOptions = { headerRow: true, bandedRows: true };
  table.borders.assign({ style: "solid", fill: COLORS.rule, width: 1 });
  table.cells.block({ row: 0, column: 0, rowCount: 1, columnCount: values[0].length }).assign({
    fill: COLORS.panel,
    textStyle: { fontFace: "Microsoft YaHei", fontSize: 10.5, bold: true, color: COLORS.ink },
  });
  table.cells.block({ row: 1, column: 0, rowCount: values.length - 1, columnCount: values[0].length }).assign({
    textStyle: { fontFace: "Microsoft YaHei", fontSize: 9.2, color: COLORS.ink },
  });
}

async function readImageBlob(imagePath) {
  const bytes = await fs.readFile(imagePath);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

function rowLabel(row) {
  return `${row.frequency_hz} Hz / PZT ${row.pzt_mvpp} mVpp / 显示点位 ${row.requested_display_point} / SNR ${fmt(row.peak_line_snr_db, 2, " dB")}`;
}

function statusNote(row) {
  if (String(row.status_note || "").startsWith("time_inferred")) {
    return "时间已自动推断并匹配同名时间文件";
  }
  if (String(row.status_note || "").startsWith("use next")) {
    return "按要求改用下一时刻的单个10s光谷文件";
  }
  if (String(row.status_note || "").startsWith("use specified")) {
    return "按要求改用11:02:24的单个10s光谷文件";
  }
  return "取记录时间最近的单个10s光谷文件";
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
    addText(slide, `${row.selected_file_time || ""}  ${statusNote(row)}`, { left: frame.left, top: frame.top + frame.height + 48, width: frame.width, height: 56 }, { fontSize: 16, color: COLORS.muted });
  }
}

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

async function main() {
  await fs.mkdir(path.dirname(OUTPUT_PPTX), { recursive: true });
  await fs.mkdir(QA_DIR, { recursive: true });
  const rows = sortRows(parseCsv(await fs.readFile(SUMMARY_CSV, "utf8")));
  const exported = rows.filter((row) => row.status === "exported");
  const skipped = rows.filter((row) => row.status !== "exported");
  const rows1000 = exported.filter((row) => Number(row.sampling_hz) === 1000);
  const rows4000 = exported.filter((row) => Number(row.sampling_hz) === 4000);
  const mismatch = frequencyMismatchRows(exported);
  const best = maxRow(exported);
  const first = exported[0];

  const presentation = Presentation.create({ slideSize: { width: SLIDE_W, height: SLIDE_H } });

  const titleSlide = presentation.slides.add();
  titleSlide.background.fill = COLORS.canvas;
  addText(titleSlide, "光谷互联40ns点位分析汇报", { left: 42, top: 82, width: 1060, height: 92 }, { fontSize: 54, bold: true });
  addText(titleSlide, "单文件约10s，不做拼接；每条记录取时间最接近的单个.dat文件", { left: 42, top: 204, width: 1120, height: 38 }, { fontSize: 24, color: COLORS.muted });
  addText(titleSlide, `位置 8202 m；显示点位 ${first.requested_display_point}；内部索引 ${first.point_index}；实际距离 ${fmt(first.distance_m, 3, " m")}`, { left: 42, top: 252, width: 1120, height: 34 }, { fontSize: 22, color: COLORS.muted });
  addRule(titleSlide, 334);
  const metrics = [
    ["记录数", `${rows.length} 条`],
    ["完成图片", `${exported.length} 张`],
    ["跳过", `${skipped.length} 条`],
    ["平均点位SNR", `${average(exported).toFixed(2)} dB`],
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
  addText(tableSlide, "原记录SNR=表格特征频率峰值-底噪水平；点位SNR为单个10s文件自动峰值SNR。跳过项表示记录时间附近缺少可用文件。", { left: 42, top: 648, width: 1150, height: 36 }, { fontSize: 15, color: COLORS.muted });

  const conclusionSlide = presentation.slides.add();
  conclusionSlide.background.fill = COLORS.canvas;
  addSlideTitle(conclusionSlide, "分析结论");
  addBullets(
    conclusionSlide,
    "主要观察",
    [
      `光谷互联数据判定为10s单文件格式，本次25条记录中完成 ${exported.length} 条，跳过 ${skipped.length} 条。`,
      `完成记录平均点位SNR为 ${average(exported).toFixed(2)} dB；最高SNR出现在 ${best.frequency_hz}Hz / PZT ${best.pzt_mvpp}mVpp，为 ${fmt(best.peak_line_snr_db, 2, " dB")}。`,
      `1000Hz采样完成 ${rows1000.length} 条，4000Hz采样完成 ${rows4000.length} 条；两类采样率已在表格中分开标注。`,
      `自动识别有 ${mismatch.length} 条峰值未贴近表格频率，建议这些记录后续用目标频率模式复核。`,
    ],
    42,
    132,
    1130,
  );

  const noteSlide = presentation.slides.add();
  noteSlide.background.fill = COLORS.canvas;
  addSlideTitle(noteSlide, "时间匹配与复核建议");
  const skippedText = skipped.length
    ? skipped.map((row) => `${row.frequency_hz}Hz / PZT ${row.pzt_mvpp}mVpp：${row.time_text}，${skipReason(row)}`).join("\n")
    : "无跳过记录。";
  const mismatchText = mismatch.slice(0, 6).map((row) => `${row.frequency_hz}Hz / PZT ${row.pzt_mvpp}mVpp：自动峰 ${fmt(row.peak_freq_hz, 3, " Hz")}`).join("\n");
  addText(noteSlide, "跳过记录", { left: 42, top: 132, width: 260, height: 34 }, { fontSize: 24, bold: true });
  addText(noteSlide, skippedText, { left: 42, top: 184, width: 1140, height: 120 }, { fontSize: 18, color: COLORS.ink });
  addText(noteSlide, "建议目标频率复核记录", { left: 42, top: 342, width: 420, height: 34 }, { fontSize: 24, bold: true });
  addText(noteSlide, mismatchText || "自动峰均贴近记录频率。", { left: 42, top: 394, width: 1140, height: 180 }, { fontSize: 18, color: COLORS.ink });
  addText(noteSlide, "说明：记录 `2026-06-15-08-41` 自动推断为 `2026-06-15 13:08:41`，并匹配到同名时间文件；该推断已写入summary。", { left: 42, top: 608, width: 1140, height: 34 }, { fontSize: 16, color: COLORS.muted });

  const pages = chunk(exported, 2);
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
