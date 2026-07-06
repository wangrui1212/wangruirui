import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "file:///C:/Users/YOSC/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const __filename = fileURLToPath(import.meta.url);
const PROJECT_ROOT = path.dirname(__filename);
const OUTPUT_DIR = path.join(PROJECT_ROOT, "outputs", "image_to_excel");
const OUTPUT_XLSX = path.join(OUTPUT_DIR, "DAS测试记录表格_20260630_40ns_200M频移.xlsx");
const PREVIEW_PNG = path.join(OUTPUT_DIR, "DAS测试记录表格_20260630_40ns_200M频移_preview.png");

const rows = [
  [1, 50, -32, -55, 1000, 40, 8, "AM 9:16:47—9:17:37 位置 8141.32"],
  [1, 100, -24, -50, 1000, 40, 8, "AM 9:18:58—9:19:30"],
  [1, 200, -24, -57, 1000, 40, 8, "AM 9:21:05—9:21:40"],
  [1, 400, -18, -55, 1000, 40, 8, "AM 9:22:09—9:22:40"],
  [5, 50, -22, -50, 1000, 40, 8, "AM 9:23:45—9:24:20"],
  [5, 100, -15.5, -56, 1000, 40, 8, "AM 9:25:40—9:26:20"],
  [5, 200, -15.5, -56, 1000, 40, 8, "AM 9:28:32—9:29:15"],
  [10, 50, -25, -50, 1000, 40, 8, "AM 9:30:21—9:31:00"],
  [10, 100, -19, -50, 1000, 40, 8, "AM 9:32:01—9:33:10"],
  [10, 200, -16, -53, 1000, 40, 8, "AM 9:36:03—9:36:40"],
  [30, 50, -24, -55, 1000, 40, 8, "AM 9:38:08—9:38:50"],
  [30, 100, -14, -55, 1000, 40, 8, "AM 9:39:38—9:40:20"],
  [30, 200, -14, -58, 1000, 40, 8, "AM 9:42:23—9:43:00"],
  [50, 50, -26, -55, 1000, 40, 8, "AM 9:43:52—9:44:30"],
  [50, 100, -16, -55, 1000, 40, 8, "AM 9:45:41—9:46:20"],
  [50, 200, -18, -52, 1000, 40, 8, "AM 9:47:28—9:48:05"],
  [100, 50, -26, -57, 1000, 40, 8, "AM 9:51:38—9:52:36"],
  [100, 100, -16, -55, 1000, 40, 8, "AM 9:54:00—9:55:00"],
  [100, 200, -16, -55, 1000, 40, 8, "AM 9:56:30—9:57:10"],
  [500, 50, -26, -57, 4000, 40, 8, "AM 10:03:52—10:05:30"],
  [500, 100, -14, -52, 4000, 40, 8, "AM 10:06:17—10:07:00"],
  [500, 200, -14, -50, 4000, 40, 8, "AM 10:07:47—10:08:40"],
];

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

async function main() {
  await fs.mkdir(OUTPUT_DIR, { recursive: true });
  const workbook = Workbook.create();
  const sheet = workbook.worksheets.add("40ns_200M频移");
  sheet.showGridLines = false;

  sheet.getRange("A1:H1").merge();
  sheet.getRange("A1:H1").values = [["DAS测试测试记录表格    日期：2026.06.30        40ns脉宽、200M频移"]];
  sheet.getRange("A2:H2").merge();
  sheet.getRange("A2:H2").values = [["实验三  新方案DAS测试，采用PZT作为振动源，信号发生器控制振动频率和幅度，测试DAS响应情况（特征频率、信噪比）"]];
  sheet.getRange("A3:H3").merge();
  sheet.getRange("A3:H3").values = [["测试位置8147.32Km、EDFA 电流200mA"]];
  sheet.getRange("A4:H4").merge();
  sheet.getRange("A4:H4").values = [["备注：每次更改条件，需要记录原始数据，在下方备注栏记录原始文件名称"]];

  const header = [
    "振动频率\n(Hz)",
    "PZT幅值\n(mVpp)",
    "特征频率峰值\n(dB)",
    "底噪水平\n(dB)",
    "采样频率\n(Hz)",
    "脉宽\n(ns)",
    "光纤长度\n(km)",
    "备注（1s测试，数据采集≥30s）",
  ];
  sheet.getRange("A5:H5").values = [header];
  sheet.getRange(`A6:H${5 + rows.length}`).values = rows;

  sheet.freezePanes.freezeRows(5);
  sheet.getRange("A1:H1").format = {
    font: { bold: true, size: 18, color: "#000000" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
  };
  sheet.getRange("A1:H1").format.font.color = "#000000";
  sheet.getRange("A2:H4").format = {
    font: { bold: true, size: 12, color: "#000000" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
  };
  sheet.getRange("A4:H4").format.horizontalAlignment = "left";
  sheet.getRange("A5:H5").format = {
    fill: "#F2F2F2",
    font: { bold: true, color: "#000000" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "all", style: "thin", color: "#000000" },
  };
  sheet.getRange(`A6:H${5 + rows.length}`).format = {
    horizontalAlignment: "center",
    verticalAlignment: "center",
    borders: { preset: "all", style: "thin", color: "#000000" },
  };
  sheet.getRange(`H6:H${5 + rows.length}`).format.horizontalAlignment = "left";
  sheet.getRange(`F6:F${5 + rows.length}`).format.font.color = "#FF0000";
  sheet.getRange(`A6:B${5 + rows.length}`).format.numberFormat = "0";
  sheet.getRange(`C6:D${5 + rows.length}`).format.numberFormat = "General";
  sheet.getRange(`E6:G${5 + rows.length}`).format.numberFormat = "0";
  sheet.getRange(`H6:H${5 + rows.length}`).format.numberFormat = "@";
  sheet.getRange("A1:H5").format.borders = { preset: "all", style: "thin", color: "#000000" };

  sheet.getRange("A:A").format.columnWidthPx = 88;
  sheet.getRange("B:B").format.columnWidthPx = 130;
  sheet.getRange("C:C").format.columnWidthPx = 150;
  sheet.getRange("D:D").format.columnWidthPx = 126;
  sheet.getRange("E:E").format.columnWidthPx = 112;
  sheet.getRange("F:F").format.columnWidthPx = 84;
  sheet.getRange("G:G").format.columnWidthPx = 120;
  sheet.getRange("H:H").format.columnWidthPx = 360;
  sheet.getRange("A1:H1").format.rowHeightPx = 52;
  sheet.getRange("A2:H2").format.rowHeightPx = 62;
  sheet.getRange("A3:H4").format.rowHeightPx = 48;
  sheet.getRange("A5:H5").format.rowHeightPx = 62;
  sheet.getRange(`A6:H${5 + rows.length}`).format.rowHeightPx = 25;

  const inspect = await workbook.inspect({
    kind: "table",
    sheetId: "40ns_200M频移",
    range: `A1:H${5 + rows.length}`,
    include: "values",
    tableMaxRows: 32,
    tableMaxCols: 8,
    maxChars: 8000,
  });
  console.log(inspect.ndjson);

  const errors = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
    options: { useRegex: true, maxResults: 50 },
    summary: "formula error scan",
  });
  console.log(errors.ndjson);

  const preview = await workbook.render({
    sheetName: "40ns_200M频移",
    range: `A1:H${5 + rows.length}`,
    scale: 1,
    format: "png",
  });
  await writeBlob(PREVIEW_PNG, preview);

  const xlsx = await SpreadsheetFile.exportXlsx(workbook);
  await xlsx.save(OUTPUT_XLSX);
  console.log(JSON.stringify({ output: OUTPUT_XLSX, preview: PREVIEW_PNG, rows: rows.length }, null, 2));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
