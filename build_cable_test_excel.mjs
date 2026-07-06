import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "file:///C:/Users/YOSC/AppData/Local/Temp/codex-spreadsheets/cable-test-excel/tmp/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const __filename = fileURLToPath(import.meta.url);
const PROJECT_ROOT = path.dirname(__filename);
const OUTPUT_DIR = path.join(PROJECT_ROOT, "outputs", "cable_test");
const OUTPUT_XLSX = path.join(OUTPUT_DIR, "缆线测试记录整理.xlsx");
const PREVIEW_DIR = path.join(OUTPUT_DIR, "preview");

const records = [
  [1, "5", "×", "×", "", "15:20:40 - 15:21:10", "100%", "", "低于50Hz音量小无特征频率，统一按100%测试"],
  [2, "10", -35.7, -65, "", "15:07:11 - 15:07:41", "100%", "", ""],
  [3, "30", -32, -65, "", "15:09:08 - 15:09:38", "100%", "", ""],
  [4, "50", -19, -61, "", "15:10:51 - 15:11:31", "100%", "", ""],
  [5, "100", -23, -68, "", "15:12:27 - 15:13:00", "100%", "", ""],
  [6, "200", -26, -67, "", "15:13:33 - 15:14:03", "100%", "", ""],
  [7, "500", -2, -66, "", "15:14:30 - 15:15:38", "100%", "", "500Hz频率高、音量大，出现饱和，底噪抬升"],
  [8, "空白", "×", "×", "", "15:22:10 - 15:23:09", "100%", "", "空白测试"],
  [9, "1", "×", "×", "", "15:23:41 - 15:24:20", "100%", "", "低频无有效特征峰"],
  [10, "500", -8, -68, "", "15:16:48 - 15:17:18", "50%", "", "降低音量后测试"],
];

const notes = [
  ["项目", "内容"],
  ["整体说明", "测试时由手机控制音量，基本都是100%。"],
  ["低频说明", "低频、低于50Hz时音量小了没有特征频率，所以统一到100%测试。"],
  ["500Hz说明", "500Hz时频率高，音量也大，出现饱和，底噪直接抬升。"],
  ["处理方式", "降低音量后补测500Hz。"],
  ["测试条件", "4K采样；测试位置1848.43 m；100ns。"],
  ["来源图片1", "D:/SummaryofWork/200DataTreating/203TDAS/TDAS_python/缆线测试/企业微信截图_17822872131942.png"],
  ["来源图片2", "D:/SummaryofWork/200DataTreating/203TDAS/TDAS_python/缆线测试/企业微信截图_17826591607635.png"],
];

async function imageDataUrl(filePath) {
  const bytes = await fs.readFile(filePath);
  return `data:image/png;base64,${bytes.toString("base64")}`;
}

function styleTitle(range) {
  range.format = {
    fill: "#1F4E78",
    font: { bold: true, color: "#FFFFFF", size: 16 },
    horizontalAlignment: "center",
    verticalAlignment: "center",
  };
}

function styleHeader(range) {
  range.format = {
    fill: "#D9EAF7",
    font: { bold: true, color: "#000000" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    borders: { preset: "all", style: "thin", color: "#B7B7B7" },
  };
}

async function main() {
  await fs.mkdir(OUTPUT_DIR, { recursive: true });
  await fs.mkdir(PREVIEW_DIR, { recursive: true });

  const workbook = Workbook.create();
  const sheet = workbook.worksheets.add("测试记录");
  sheet.showGridLines = false;

  sheet.getRange("A1:I1").merge();
  sheet.getRange("A1").values = [["缆线测试记录整理"]];
  styleTitle(sheet.getRange("A1:I1"));

  sheet.getRange("A2:I2").merge();
  sheet.getRange("A2").values = [["测试条件：4K采样；测试位置1848.43 m；100ns；音量主要为100%，500Hz补充50%音量测试"]];
  sheet.getRange("A2:I2").format = {
    fill: "#F2F2F2",
    font: { color: "#333333" },
    horizontalAlignment: "left",
    wrapText: true,
  };

  const headers = [["序号", "频率(Hz)", "峰(dB)", "底噪(dB)", "信噪差(dB)", "测试时间段", "音量", "确认", "备注"]];
  sheet.getRange("A3:I3").values = headers;
  styleHeader(sheet.getRange("A3:I3"));
  sheet.getRange("A4:I13").values = records;
  sheet.getRange("E4:E13").formulas = records.map((_, index) => {
    const rowNumber = index + 4;
    return [`=IF(OR(C${rowNumber}="×",D${rowNumber}="×"),"",C${rowNumber}-D${rowNumber})`];
  });
  sheet.getRange("A3:I13").format.borders = { preset: "all", style: "thin", color: "#D9D9D9" };
  sheet.getRange("A4:A13").format.horizontalAlignment = "center";
  sheet.getRange("B4:B13").format.horizontalAlignment = "center";
  sheet.getRange("C4:E13").format.numberFormat = "0.00";
  sheet.getRange("G4:G13").format.horizontalAlignment = "center";
  sheet.getRange("I4:I13").format.wrapText = true;
  sheet.freezePanes.freezeRows(3);

  const widths = [9, 12, 12, 12, 14, 26, 10, 10, 36];
  widths.forEach((width, index) => {
    sheet.getRangeByIndexes(0, index, 1, 1).format.columnWidth = width;
  });
  sheet.getRange("A1:I13").format.rowHeight = 24;
  sheet.getRange("A1:I1").format.rowHeight = 34;
  sheet.getRange("A2:I2").format.rowHeight = 38;
  sheet.getRange("I4:I13").format.rowHeight = 42;

  const noteSheet = workbook.worksheets.add("测试说明");
  noteSheet.showGridLines = false;
  noteSheet.getRange("A1:B1").merge();
  noteSheet.getRange("A1").values = [["截图文字说明整理"]];
  styleTitle(noteSheet.getRange("A1:B1"));
  noteSheet.getRange("A2:B9").values = notes;
  styleHeader(noteSheet.getRange("A2:B2"));
  noteSheet.getRange("A2:B9").format.borders = { preset: "all", style: "thin", color: "#D9D9D9" };
  noteSheet.getRange("A3:B9").format.wrapText = true;
  noteSheet.getRange("A:A").format.columnWidth = 18;
  noteSheet.getRange("B:B").format.columnWidth = 92;
  noteSheet.getRange("A1:B1").format.rowHeight = 34;
  noteSheet.getRange("A2:B9").format.rowHeight = 42;

  const imageSheet = workbook.worksheets.add("原始截图");
  imageSheet.showGridLines = false;
  imageSheet.getRange("A1:H1").merge();
  imageSheet.getRange("A1").values = [["原始截图"]];
  styleTitle(imageSheet.getRange("A1:H1"));
  imageSheet.images.add({
    dataUrl: await imageDataUrl(path.join(PROJECT_ROOT, "缆线测试", "企业微信截图_17822872131942.png")),
    anchor: { from: { row: 2, col: 0 }, extent: { widthPx: 760, heightPx: 530 } },
  });
  imageSheet.images.add({
    dataUrl: await imageDataUrl(path.join(PROJECT_ROOT, "缆线测试", "企业微信截图_17826591607635.png")),
    anchor: { from: { row: 2, col: 9 }, extent: { widthPx: 520, heightPx: 180 } },
  });
  for (let c = 0; c < 16; c += 1) {
    imageSheet.getRangeByIndexes(0, c, 1, 1).format.columnWidth = 12;
  }

  for (const sheetName of ["测试记录", "测试说明"]) {
    const preview = await workbook.render({ sheetName, autoCrop: "all", scale: 1, format: "png" });
    await fs.writeFile(path.join(PREVIEW_DIR, `${sheetName}.png`), new Uint8Array(await preview.arrayBuffer()));
  }

  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(OUTPUT_XLSX);
  console.log(JSON.stringify({ output: OUTPUT_XLSX, previewDir: PREVIEW_DIR }, null, 2));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
