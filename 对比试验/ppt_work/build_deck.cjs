const fs = require("node:fs/promises");
const path = require("node:path");
const { Presentation, PresentationFile } = require("@oai/artifact-tool");

const root = "D:/SummaryofWork/200DataTreating/203TDAS/TDAS_python/对比试验";
const work = path.join(root, "ppt_work");
const previewDir = path.join(work, "preview");
const layoutDir = path.join(work, "layout");
const qaDir = path.join(work, "qa");
const pptxPath = path.join(root, "对比试验_设备差距说明.pptx");

const W = 1280;
const H = 720;
const C = {
  bg: "#f6f8fb",
  ink: "#14213d",
  muted: "#5b667a",
  header: "#102a43",
  blue: "#2563eb",
  teal: "#0f766e",
  orange: "#f97316",
  red: "#dc2626",
  line: "#d6dde8",
  white: "#ffffff",
  paleBlue: "#eaf2ff",
  paleTeal: "#e9f8f5",
  paleOrange: "#fff3e6",
  paleRed: "#feecec",
};
const font = "Microsoft YaHei";

async function readAsset(manifest, i) {
  const m = manifest[i - 1];
  const bytes = await fs.readFile(m.file);
  return {
    blob: bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
    contentType: m.contentType,
    alt: `对比试验原始图片 ${i}: ${m.name}`,
  };
}

function addShape(slide, geometry, position, fill = "none", lineFill = "none", radius) {
  const cfg = {
    geometry,
    position,
    fill,
    line: { style: "solid", fill: lineFill, width: lineFill === "none" ? 0 : 1 },
  };
  if (radius) cfg.borderRadius = radius;
  return slide.shapes.add(cfg);
}

function addText(slide, text, position, style = {}) {
  const box = addShape(slide, "textbox", position, "none", "none");
  box.text = text;
  box.text.style = {
    typeface: font,
    fontSize: style.fontSize ?? 24,
    color: style.color ?? C.ink,
    bold: style.bold ?? false,
    alignment: style.alignment ?? "left",
    verticalAlignment: style.verticalAlignment ?? "top",
    lineSpacing: style.lineSpacing ?? 1.12,
    autoFit: style.autoFit ?? "shrinkText",
    insets: style.insets ?? { top: 4, right: 6, bottom: 4, left: 6 },
  };
  return box;
}

function addTitle(slide, title, kicker = "TDAS 对比试验") {
  addShape(slide, "rect", { left: 0, top: 0, width: W, height: 74 }, C.header, "none");
  addText(slide, kicker, { left: 54, top: 16, width: 220, height: 18 }, {
    fontSize: 13,
    color: "#b7c6d9",
    bold: true,
  });
  addText(slide, title, { left: 52, top: 34, width: 900, height: 34 }, {
    fontSize: 30,
    color: C.white,
    bold: true,
  });
  addText(slide, "来源：对比试验.docx", { left: 1000, top: 30, width: 220, height: 24 }, {
    fontSize: 12,
    color: "#d7e2ef",
    alignment: "right",
  });
}

function addFooter(slide, page) {
  addShape(slide, "line", { left: 54, top: 676, width: 1172, height: 0 }, "none", C.line);
  addText(slide, "100 Hz 低音量音乐响应分析", { left: 54, top: 685, width: 360, height: 18 }, {
    fontSize: 11,
    color: C.muted,
  });
  addText(slide, String(page).padStart(2, "0"), { left: 1180, top: 682, width: 46, height: 22 }, {
    fontSize: 13,
    color: C.muted,
    bold: true,
    alignment: "right",
  });
}

function bulletText(slide, items, position, fontSize = 19) {
  const box = addShape(slide, "textbox", position, "none", "none");
  box.text.set(items.map(([lead, rest]) => ({
    bulletCharacter: "•",
    marginLeft: 22,
    indent: -12,
    spaceAfter: 8,
    runs: [
      { run: lead, textStyle: { bold: true, color: C.ink } },
      { run: rest, textStyle: { color: C.muted } },
    ],
  })));
  box.text.style = {
    typeface: font,
    fontSize,
    lineSpacing: 1.18,
    autoFit: "shrinkText",
    insets: { top: 4, right: 8, bottom: 4, left: 8 },
  };
  return box;
}

function card(slide, title, body, position, color = C.blue, fill = C.white) {
  addShape(slide, "roundRect", position, fill, C.line, 8);
  addShape(slide, "rect", { left: position.left, top: position.top, width: 6, height: position.height }, color, "none");
  addText(slide, title, { left: position.left + 20, top: position.top + 16, width: position.width - 40, height: 28 }, {
    fontSize: 21,
    color,
    bold: true,
  });
  addText(slide, body, { left: position.left + 20, top: position.top + 54, width: position.width - 38, height: position.height - 68 }, {
    fontSize: position.height <= 120 ? 16 : 17,
    color: C.muted,
    lineSpacing: 1.18,
  });
}

async function addImage(slide, manifest, idx, position, fit = "contain", label) {
  const img = await readAsset(manifest, idx);
  slide.images.add({
    ...img,
    position,
    fit,
    geometry: "roundRect",
    borderRadius: 6,
  });
  addShape(slide, "roundRect", position, "none", C.line, 6);
  if (label) {
    addShape(slide, "roundRect", { left: position.left + 10, top: position.top + 10, width: 205, height: 26 }, "#111827CC", "none", 6);
    addText(slide, label, { left: position.left + 18, top: position.top + 14, width: 190, height: 17 }, {
      fontSize: 12,
      color: C.white,
      bold: true,
    });
  }
}

function addTag(slide, text, position, color = C.ink) {
  addShape(slide, "roundRect", position, "#ffffffE6", color, 6);
  addText(slide, text, { left: position.left + 8, top: position.top + 5, width: position.width - 16, height: position.height - 8 }, {
    fontSize: 12,
    color,
    bold: true,
    alignment: "center",
  });
}

function addMarker(slide, x, y, label, color = C.red, labelDx = 12, labelDy = -32) {
  addShape(slide, "ellipse", { left: x - 5, top: y - 5, width: 10, height: 10 }, color, color);
  addShape(slide, "line", { left: x, top: y, width: labelDx, height: labelDy }, "none", color);
  addTag(slide, label, { left: x + labelDx, top: y + labelDy - 8, width: 144, height: 30 }, color);
}

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

async function main() {
  await fs.mkdir(previewDir, { recursive: true });
  await fs.mkdir(layoutDir, { recursive: true });
  await fs.mkdir(qaDir, { recursive: true });

  const manifest = JSON.parse(await fs.readFile(path.join(work, "image-manifest.json"), "utf8"));
  const presentation = Presentation.create({ slideSize: { width: W, height: H } });

  {
    const s = presentation.slides.add();
    s.background.fill = C.bg;
    addShape(s, "rect", { left: 0, top: 0, width: 440, height: H }, C.header, "none");
    addText(s, "对比试验\n设备差距说明", { left: 58, top: 116, width: 340, height: 158 }, {
      fontSize: 47,
      color: C.white,
      bold: true,
      lineSpacing: 1.06,
    });
    addText(s, "100 Hz 低音量音乐响应分析", { left: 62, top: 292, width: 330, height: 34 }, {
      fontSize: 22,
      color: "#cbd8e6",
      bold: true,
    });
    addText(s, "技术评审版", { left: 64, top: 608, width: 160, height: 26 }, {
      fontSize: 18,
      color: "#e2e8f0",
      bold: true,
    });
    await addImage(s, manifest, 1, { left: 500, top: 94, width: 320, height: 440 }, "cover", "现场布置");
    await addImage(s, manifest, 3, { left: 842, top: 140, width: 360, height: 260 }, "contain", "相位谱热力图");
    addShape(s, "roundRect", { left: 842, top: 424, width: 360, height: 110 }, C.white, C.line, 8);
    addText(s, "核心结论", { left: 866, top: 445, width: 120, height: 26 }, { fontSize: 20, color: C.blue, bold: true });
    addText(s, "两设备 100 Hz 响应一致；主要差距体现在显示方式、滤波策略、高频噪声、参数解耦与工程适应性。", { left: 866, top: 478, width: 300, height: 44 }, {
      fontSize: 17,
      color: C.muted,
    });
    addFooter(s, 1);
  }

  {
    const s = presentation.slides.add();
    s.background.fill = C.bg;
    addTitle(s, "关键前提：两设备 AOM 载波频率不同，导致试验脉宽设置不同");
    addShape(s, "roundRect", { left: 86, top: 132, width: 500, height: 252 }, C.white, C.line, 8);
    addText(s, "自研设备", { left: 126, top: 164, width: 180, height: 36 }, { fontSize: 30, color: C.blue, bold: true });
    addText(s, "80 MHz", { left: 126, top: 222, width: 240, height: 62 }, { fontSize: 54, color: C.ink, bold: true });
    addText(s, "AOM 载波频率", { left: 375, top: 246, width: 150, height: 24 }, { fontSize: 19, color: C.muted });
    addText(s, "脉宽：100 ns\n标距：10 m", { left: 126, top: 306, width: 360, height: 52 }, { fontSize: 24, color: C.ink, bold: true, lineSpacing: 1.22 });

    addShape(s, "roundRect", { left: 694, top: 132, width: 500, height: 252 }, C.white, C.line, 8);
    addText(s, "光谷互联设备", { left: 734, top: 164, width: 240, height: 36 }, { fontSize: 30, color: C.orange, bold: true });
    addText(s, "200 MHz", { left: 734, top: 222, width: 260, height: 62 }, { fontSize: 54, color: C.ink, bold: true });
    addText(s, "AOM 载波频率", { left: 1012, top: 246, width: 150, height: 24 }, { fontSize: 19, color: C.muted });
    addText(s, "设置脉宽：10 ns（实际 25 ns）\n标距：0.5 m", { left: 734, top: 306, width: 390, height: 52 }, { fontSize: 24, color: C.ink, bold: true, lineSpacing: 1.22 });

    addShape(s, "line", { left: 590, top: 252, width: 100, height: 0 }, "none", C.red);
    addText(s, "载波不同", { left: 590, top: 214, width: 100, height: 28 }, { fontSize: 18, color: C.red, bold: true, alignment: "center" });

    addShape(s, "roundRect", { left: 110, top: 440, width: 1060, height: 136 }, C.paleOrange, C.orange, 8);
    addText(s, "试验条件解读", { left: 148, top: 468, width: 180, height: 30 }, { fontSize: 24, color: C.orange, bold: true });
    addText(s, "本次对比中，两设备 AOM 载波频率不一致，因此脉宽条件不完全相同：自研设备为 100 ns，光谷互联设置 10 ns、实际约 25 ns；同时标距分别为 10 m 与 0.5 m。后续信噪比、分辨率和频谱底噪差异需结合该参数前提理解。", { left: 332, top: 456, width: 790, height: 72 }, {
      fontSize: 20,
      color: C.ink,
      lineSpacing: 1.18,
    });
    addFooter(s, 2);
  }

  {
    const s = presentation.slides.add();
    s.background.fill = C.bg;
    addTitle(s, "量化响应：100 Hz 响应均在 20-30 dB，两设备无明显差距");
    await addImage(s, manifest, 1, { left: 72, top: 124, width: 350, height: 470 }, "cover", "测试现场");
    await addImage(s, manifest, 2, { left: 454, top: 124, width: 250, height: 470 }, "contain", "100 Hz 截图");
    addShape(s, "roundRect", { left: 742, top: 124, width: 456, height: 470 }, C.white, C.line, 8);
    addText(s, "量化指标", { left: 778, top: 154, width: 180, height: 34 }, { fontSize: 26, color: C.ink, bold: true });
    addShape(s, "roundRect", { left: 778, top: 204, width: 170, height: 88 }, C.paleBlue, C.blue, 8);
    addText(s, "自研设备", { left: 798, top: 222, width: 120, height: 24 }, { fontSize: 18, color: C.blue, bold: true });
    addText(s, "20-30 dB", { left: 798, top: 252, width: 130, height: 28 }, { fontSize: 25, color: C.ink, bold: true });
    addShape(s, "roundRect", { left: 972, top: 204, width: 170, height: 88 }, C.paleOrange, C.orange, 8);
    addText(s, "光谷互联", { left: 992, top: 222, width: 120, height: 24 }, { fontSize: 18, color: C.orange, bold: true });
    addText(s, "20-30 dB", { left: 992, top: 252, width: 130, height: 28 }, { fontSize: 25, color: C.ink, bold: true });
    addShape(s, "roundRect", { left: 778, top: 314, width: 364, height: 54 }, C.paleTeal, C.teal, 8);
    addText(s, "结论：该 100 Hz 激励下，两设备响应幅度无明显差距。", { left: 798, top: 330, width: 324, height: 24 }, { fontSize: 19, color: C.teal, bold: true });
    bulletText(s, [
      ["激励方式：", "将音响置于光纤之上，播放 100 Hz、低音量音乐。"],
      ["分析对象：", "对两套设备的响应、热力图、频谱表现进行对比。"],
      ["证据来源：", "本 PPT 所有图片均来自对比试验.docx 原始插图。"],
    ], { left: 778, top: 400, width: 370, height: 120 }, 17);
    addShape(s, "roundRect", { left: 778, top: 526, width: 370, height: 82 }, C.paleOrange, C.orange, 8);
    addText(s, "读取口径", { left: 802, top: 546, width: 110, height: 24 }, { fontSize: 20, color: C.orange, bold: true });
    addText(s, "响应指标用于说明 100 Hz 响应幅度层面的对比结论。", { left: 802, top: 574, width: 312, height: 22 }, { fontSize: 14, color: C.muted });
    addFooter(s, 3);
  }

  {
    const s = presentation.slides.add();
    s.background.fill = C.bg;
    addTitle(s, "响应一致，但显示策略不同：全点位 vs 抽样");
    await addImage(s, manifest, 3, { left: 64, top: 126, width: 565, height: 292 }, "contain", "自研设备：相位谱热力图");
    await addImage(s, manifest, 7, { left: 651, top: 126, width: 565, height: 292 }, "contain", "光谷互联：热力图");
    card(s, "共同点", "对 100 Hz 信号响应两设备一致，说明在该激励条件下两套设备均能捕捉到目标响应。", { left: 96, top: 462, width: 330, height: 128 }, C.teal, C.paleTeal);
    card(s, "自研设备", "采用全部点位显示，空间点位信息保留更完整；噪声和非目标点位也会直接呈现", { left: 475, top: 462, width: 330, height: 128 }, C.blue, C.paleBlue);
    card(s, "光谷互联设备", "采用抽样显示，界面呈现更轻量，适合快速查看趋势，但完整点位细节不如全量显示直观。", { left: 854, top: 462, width: 330, height: 128 }, C.orange, C.paleOrange);
    addFooter(s, 4);
  }

  {
    const s = presentation.slides.add();
    s.background.fill = C.bg;
    addTitle(s, "低频处理差异：光谷互联加入低频滤波以抑制相位噪声");
    await addImage(s, manifest, 4, { left: 72, top: 132, width: 540, height: 406 }, "contain", "自研设备频谱/响应图");
    await addImage(s, manifest, 8, { left: 668, top: 132, width: 540, height: 406 }, "contain", "光谷互联频谱/响应图");
    addMarker(s, 139, 410, "100 Hz\n(0.1 kHz, 约 -42 dB)", C.red, 18, -44);
    addMarker(s, 354, 458, "1 kHz 底噪\n约 -70 dB", C.blue, 18, -42);
    addMarker(s, 735, 405, "100 Hz\n(0.1 kHz, 约 -50 dB)", C.red, 18, -44);
    addMarker(s, 950, 432, "1 kHz 底噪\n约 -65 dB", C.blue, -158, -38);
    addShape(s, "roundRect", { left: 156, top: 558, width: 968, height: 68 }, C.white, C.line, 8);
    addText(s, "说明：100 Hz 信号峰值可在 FFT 频谱中定位；1 kHz 处底噪显示，自研设备底噪近水平线，光谷互联底噪呈抛物线趋势。", { left: 185, top: 574, width: 910, height: 34 }, {
      fontSize: 19,
      color: C.ink,
      bold: true,
    });
    addFooter(s, 5);
  }

  {
    const s = presentation.slides.add();
    s.background.fill = C.bg;
    addTitle(s, "高频噪声差异：光谷互联没有高频特征噪声");
    await addImage(s, manifest, 5, { left: 72, top: 128, width: 520, height: 310 }, "contain", "自研设备频谱图");
    await addImage(s, manifest, 9, { left: 688, top: 128, width: 520, height: 310 }, "contain", "光谷互联频谱图");
    addTag(s, "颜色条区间：-46 ~ -69", { left: 520, top: 410, width: 170, height: 30 }, C.red);
    addTag(s, "颜色条区间：-46 ~ -69", { left: 1095, top: 410, width: 170, height: 30 }, C.red);
    addShape(s, "roundRect", { left: 86, top: 482, width: 1108, height: 110 }, C.white, C.line, 8);
    addText(s, "观察结论", { left: 118, top: 508, width: 150, height: 30 }, { fontSize: 24, color: C.red, bold: true });
    addText(s, "颜色条位于 -46 ~ -69 区间。文档结论明确指出：光谷互联设备没有高频特征噪声；自研设备需进一步定位高频特征噪声来源，并区分硬件、解调算法、滤波链路与显示阈值带来的影响。", { left: 270, top: 503, width: 850, height: 58 }, {
      fontSize: 20,
      color: C.ink,
      lineSpacing: 1.14,
    });
    addFooter(s, 6);
  }

  {
    const s = presentation.slides.add();
    s.background.fill = C.bg;
    addTitle(s, "参数配置差异：两设备脉宽与标距设置明显不同");
    addShape(s, "roundRect", { left: 88, top: 142, width: 500, height: 315 }, C.white, C.line, 8);
    addText(s, "自研设备", { left: 124, top: 176, width: 220, height: 40 }, { fontSize: 32, color: C.blue, bold: true });
    addText(s, "100 ns", { left: 124, top: 238, width: 190, height: 58 }, { fontSize: 52, color: C.ink, bold: true });
    addText(s, "脉宽", { left: 326, top: 262, width: 90, height: 24 }, { fontSize: 22, color: C.muted });
    addText(s, "10 m", { left: 124, top: 328, width: 190, height: 58 }, { fontSize: 52, color: C.ink, bold: true });
    addText(s, "标距", { left: 326, top: 352, width: 90, height: 24 }, { fontSize: 22, color: C.muted });
    addShape(s, "roundRect", { left: 692, top: 142, width: 500, height: 315 }, C.white, C.line, 8);
    addText(s, "光谷互联设备", { left: 728, top: 176, width: 260, height: 40 }, { fontSize: 32, color: C.orange, bold: true });
    addText(s, "10 ns", { left: 728, top: 232, width: 190, height: 58 }, { fontSize: 52, color: C.ink, bold: true });
    addText(s, "实际 25 ns", { left: 728, top: 286, width: 160, height: 24 }, { fontSize: 19, color: C.orange, bold: true });
    addText(s, "脉宽", { left: 930, top: 262, width: 90, height: 24 }, { fontSize: 22, color: C.muted });
    addText(s, "0.5 m", { left: 728, top: 328, width: 190, height: 58 }, { fontSize: 52, color: C.ink, bold: true });
    addText(s, "标距", { left: 930, top: 352, width: 90, height: 24 }, { fontSize: 22, color: C.muted });
    card(s, "影响解释", "光谷互联将脉宽和标距设置得更小，目标是增加空间分辨率；代价是会降低信噪比。参数差异是后续信噪比与工程适配差异的重要原因。", { left: 180, top: 498, width: 920, height: 96 }, C.teal, C.paleTeal);
    addFooter(s, 7);
  }

  {
    const s = presentation.slides.add();
    s.background.fill = C.bg;
    addTitle(s, "权衡关系：空间分辨率提升会牺牲高频信噪比");
    await addImage(s, manifest, 6, { left: 70, top: 124, width: 545, height: 335 }, "contain", "自研设备：抬高阈值后");
    await addImage(s, manifest, 10, { left: 665, top: 124, width: 545, height: 335 }, "contain", "光谷互联：抬高阈值后");
    addTag(s, "颜色条区间：-42 ~ 57", { left: 525, top: 430, width: 160, height: 30 }, C.red);
    addTag(s, "颜色条区间：-42 ~ 57", { left: 1095, top: 430, width: 160, height: 30 }, C.red);
    addShape(s, "roundRect", { left: 118, top: 502, width: 1044, height: 94 }, C.white, C.line, 8);
    addText(s, "颜色条位于 -42 ~ 57 区间。文档判断：自研设备在 100 Hz 以上的高频信噪比优于光谷互联设备；光谷互联设置用于增加空间分辨率，因此降低了信噪比。", { left: 154, top: 520, width: 960, height: 44 }, {
      fontSize: 21,
      color: C.ink,
      bold: true,
      lineSpacing: 1.12,
    });
    addFooter(s, 8);
  }

  {
    const s = presentation.slides.add();
    s.background.fill = C.bg;
    addTitle(s, "工程适应性差异：光谷互联三参数解耦，自研设备需改进");
    card(s, "标距", "光谷互联可独立调节；自研设备与其他参数自动绑定。", { left: 108, top: 154, width: 300, height: 178 }, C.blue, C.paleBlue);
    card(s, "脉宽", "光谷互联可按分辨率与信噪比目标单独配置。", { left: 490, top: 154, width: 300, height: 178 }, C.orange, C.paleOrange);
    card(s, "点位间隔", "光谷互联与标距、脉宽解绑，工程适配性更强。", { left: 872, top: 154, width: 300, height: 178 }, C.teal, C.paleTeal);
    addShape(s, "roundRect", { left: 150, top: 430, width: 980, height: 118 }, C.white, C.line, 8);
    addText(s, "关键差距", { left: 188, top: 458, width: 150, height: 32 }, { fontSize: 26, color: C.ink, bold: true });
    addText(s, "光谷互联设备将标距、脉宽、点位间隔设计为解绑的三个参数，使工程上可按场景调节；自研设备目前三个参数自动为一个值，需要改进。", { left: 340, top: 456, width: 740, height: 52 }, {
      fontSize: 21,
      color: C.muted,
      lineSpacing: 1.15,
    });
    addFooter(s, 9);
  }

  {
    const s = presentation.slides.add();
    s.background.fill = C.bg;
    addTitle(s, "改进建议：优先补齐参数解耦、滤波策略和噪声定位能力");
    const recs = [
      ["参数解耦配置", "支持标距、脉宽、点位间隔分别设置，并提供场景化默认模板", C.blue, C.paleBlue],
      ["低频滤波策略", "增加低频滤波开关、滤波参数和相位噪声抑制策略配置。", C.orange, C.paleOrange],
      ["高频噪声定位", "专项分析高频特征噪声来源，拆分硬件、算法、滤波与显示链路影响。", C.red, C.paleRed],
      ["阈值显示标准化", "形成能量显示阈值调节规则，避免不同设备展示口径影响判断", C.teal, C.paleTeal],
    ];
    for (let i = 0; i < recs.length; i++) {
      const x = i % 2 === 0 ? 92 : 668;
      const y = i < 2 ? 138 : 386;
      card(s, recs[i][0], recs[i][1], { left: x, top: y, width: 520, height: 170 }, recs[i][2], recs[i][3]);
    }
    addShape(s, "roundRect", { left: 410, top: 305, width: 460, height: 58 }, C.header, "none", 8);
    addText(s, "建议按“工程配置能力”优先推进", { left: 438, top: 323, width: 405, height: 24 }, {
      fontSize: 22,
      color: C.white,
      bold: true,
      alignment: "center",
    });
    addFooter(s, 10);
  }

  for (const [index, slide] of presentation.slides.items.entries()) {
    const stem = `slide-${String(index + 1).padStart(2, "0")}`;
    await writeBlob(path.join(previewDir, `${stem}.png`), await presentation.export({ slide, format: "png", scale: 1 }));
    await fs.writeFile(path.join(layoutDir, `${stem}.layout.json`), await (await slide.export({ format: "layout" })).text(), "utf8");
  }
  await writeBlob(path.join(previewDir, "deck-montage.webp"), await presentation.export({ format: "webp", montage: true, scale: 1 }));

  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(pptxPath);

  await fs.writeFile(path.join(work, "slide-plan.txt"), [
    "对比试验设备差距说明 PPT",
    "页数：10",
    "风格：中文技术评审，深色标题栏，浅色内容区。",
    `输出：${pptxPath}`,
    "",
  ].join("\n"), "utf8");

  await fs.writeFile(path.join(qaDir, "visual-qa.txt"), [
    "Mechanical",
    "- PPTX exists and is non-empty: pending verification",
    "- Expected slide count: 10",
    "- Every final slide rendered: yes",
    "- Contact sheet or montage reviewed: pending visual inspection",
    "- Source notes reviewed: yes",
    "",
    "Issue Ledger",
    "| Issue | Slide(s) | Severity | Fix path | Status |",
    "|---|---:|---|---|---|",
    "| Initial render created | all | review | inspect montage and key slides | open |",
    "",
  ].join("\n"), "utf8");

  console.log(JSON.stringify({ pptxPath, slides: presentation.slides.items.length, previewDir, layoutDir }, null, 2));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
