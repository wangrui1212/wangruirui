#!/usr/bin/env python
"""Spatial segment PSD comparisons for the confirmed 2026-07-10 DAS tests."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import types
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def install_headless_qt_stubs() -> None:
    """Let tdas_dat_viewer.py import in batch environments without PyQt6."""

    class DummySignal:
        def connect(self, *_args: object, **_kwargs: object) -> None:
            return None

        def emit(self, *_args: object, **_kwargs: object) -> None:
            return None

    class DummyBase:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def __call__(self, *_args: object, **_kwargs: object) -> "DummyBase":
            return self

        def __getattr__(self, _name: str) -> "DummyBase":
            return self

    class DummyModule(types.ModuleType):
        def __getattr__(self, name: str) -> object:
            if name == "pyqtSignal":
                return lambda *_args, **_kwargs: DummySignal()
            if name == "pyqtSlot":
                return lambda *_args, **_kwargs: (lambda func: func)
            if name in {"QObject", "QRunnable", "QMainWindow", "QWidget", "QDialog"}:
                return DummyBase
            return DummyBase

    if "PyQt6" not in sys.modules:
        pyqt6 = types.ModuleType("PyQt6")
        qtcore = DummyModule("PyQt6.QtCore")
        qtgui = DummyModule("PyQt6.QtGui")
        qtwidgets = DummyModule("PyQt6.QtWidgets")
        pyqt6.QtCore = qtcore  # type: ignore[attr-defined]
        pyqt6.QtGui = qtgui  # type: ignore[attr-defined]
        pyqt6.QtWidgets = qtwidgets  # type: ignore[attr-defined]
        sys.modules["PyQt6"] = pyqt6
        sys.modules["PyQt6.QtCore"] = qtcore
        sys.modules["PyQt6.QtGui"] = qtgui
        sys.modules["PyQt6.QtWidgets"] = qtwidgets

    if "pyqtgraph" not in sys.modules:
        pg = DummyModule("pyqtgraph")
        pg.opengl = DummyModule("pyqtgraph.opengl")  # type: ignore[attr-defined]
        sys.modules["pyqtgraph"] = pg
        sys.modules["pyqtgraph.opengl"] = pg.opengl  # type: ignore[attr-defined]


install_headless_qt_stubs()

from batch_psd_20260710_analysis import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DEFAULT_WORKBOOK,
    DEVICE_CONFIGS,
    FileSegment,
    GroupRecord,
    build_segments,
    index_data_files,
    read_records,
    safe_filename,
)
from tdas_dat_viewer import (  # noqa: E402
    compute_spatial_segment_psd,
    normalize_silixa_matrix,
    psd_to_db,
    read_signal_matrix,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "井中流动分析" / "PSD分析_20260710" / "空间段PSD比对_稳定中间1分钟"

SPATIAL_SEGMENTS = {
    "自研_2U": (
        ("筒内光纤", 293.0, 302.0),
        ("筒外缆线", 348.0, 358.0),
    ),
    "OV": (
        ("筒内光纤", 320.0, 330.0),
        ("筒外缆线", 376.0, 392.0),
    ),
    "silixia": (
        ("筒内光纤", 1048.0, 1060.0),
        ("筒外缆线", 1104.0, 1116.0),
    ),
}

EXPERIMENT_TYPES = ("大流量测试", "气水两相实验")
FREQ_MAX_HZ = 500.0
WINDOW_SECONDS = 2.0
OVERLAP_PERCENT = 50.0
WINDOW_NAME = "Hann"


@dataclass
class CurveRow:
    experiment_type: str
    experiment_id: str
    source_phase: str
    device: str
    segment_name: str
    requested_range_m: str
    actual_range_m: str
    stable_start: str
    stable_end: str
    analysis_start: str
    analysis_end: str
    sample_rate_hz: int | str
    selected_channel_count: int
    valid_channel_count: int
    window_count: int
    status: str
    message: str = ""
    curve_csv: str = ""


@dataclass
class CurveData:
    row: CurveRow
    freqs_hz: np.ndarray
    psd_db: np.ndarray


def find_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
        Path(r"C:\Windows\Fonts\arial.ttf"),
    ):
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def phase_interval(record: GroupRecord, label: str) -> tuple[datetime, datetime] | None:
    for phase in record.phases:
        if phase.label == label:
            return phase.start, phase.end
    return None


def stable_phase(record: GroupRecord) -> tuple[datetime, datetime] | None:
    return phase_interval(record, "稳定")


def analysis_window(record: GroupRecord, phase_label: str = "稳定") -> tuple[datetime, datetime]:
    phase = phase_interval(record, phase_label)
    if phase is None:
        raise ValueError(f"实验组没有{phase_label}阶段")
    start, end = phase
    duration = (end - start).total_seconds()
    if duration < 60.0:
        raise ValueError(f"{phase_label}阶段不足60秒: {duration:.1f}s")
    center = start + timedelta(seconds=duration / 2.0)
    return center - timedelta(seconds=30), center + timedelta(seconds=30)


def load_stitched_matrix(segments: list[FileSegment]) -> np.ndarray:
    matrices: list[np.ndarray] = []
    for segment in segments:
        matrix = read_signal_matrix(segment.item.path, segment.header).astype(np.float32, copy=False)
        matrices.append(matrix[:, segment.start_index : segment.stop_index])
    if not matrices:
        raise ValueError("没有可拼接的数据段")
    return matrices[0] if len(matrices) == 1 else np.concatenate(matrices, axis=1)


def compute_curve(
    record: GroupRecord,
    device_title: str,
    segment_name: str,
    start_m: float,
    end_m: float,
    files: list[Any],
    phase_label: str = "稳定",
) -> CurveData:
    win_start, win_end = analysis_window(record, phase_label)
    window_record = replace(record, start=win_start, end=win_end)
    segments = build_segments(files, window_record)
    if not segments:
        raise ValueError(f"{phase_label}阶段中间1分钟没有可用数据")
    matrix = load_stitched_matrix(segments)
    header = replace(
        segments[0].header,
        sample_count=matrix.shape[1],
        acquisition_time_s=matrix.shape[1] / max(1, int(segments[0].header.frequency_hz)),
        timestamp_text=win_start.isoformat(sep=" "),
    )
    if header.protocol == "Silixa TDMS":
        matrix, _scale = normalize_silixa_matrix(matrix)
    result = compute_spatial_segment_psd(
        matrix,
        header,
        start_m,
        end_m,
        WINDOW_SECONDS,
        OVERLAP_PERCENT,
        WINDOW_NAME,
    )
    mask = result.freqs_hz <= FREQ_MAX_HZ
    freqs = result.freqs_hz[mask]
    psd_db = psd_to_db(result.mean_psd)[mask]
    selected_phase = phase_interval(record, phase_label)
    row = CurveRow(
        experiment_type=record.group,
        experiment_id=record.seq,
        source_phase=phase_label,
        device=device_title,
        segment_name=segment_name,
        requested_range_m=f"{start_m:g}-{end_m:g}",
        actual_range_m=f"{result.actual_start_m:.3f}-{result.actual_end_m:.3f}",
        stable_start=selected_phase[0].isoformat(sep=" ") if selected_phase else "",
        stable_end=selected_phase[1].isoformat(sep=" ") if selected_phase else "",
        analysis_start=win_start.isoformat(sep=" "),
        analysis_end=win_end.isoformat(sep=" "),
        sample_rate_hz=result.sample_rate_hz,
        selected_channel_count=result.selected_channel_count,
        valid_channel_count=result.valid_channel_count,
        window_count=result.window_count,
        status="ok",
    )
    return CurveData(row, freqs, psd_db)


def write_curve_csv(curve: CurveData, output_dir: Path) -> str:
    curve_dir = output_dir / "curves" / safe_filename(curve.row.experiment_type) / safe_filename(curve.row.device)
    curve_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"{safe_filename(curve.row.experiment_type)}_"
        f"{safe_filename(curve.row.experiment_id)}_"
        f"{safe_filename(curve.row.source_phase)}_"
        f"{safe_filename(curve.row.device)}_"
        f"{safe_filename(curve.row.segment_name)}_"
        f"{safe_filename(curve.row.requested_range_m)}m_PSD.csv"
    )
    path = curve_dir / filename
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frequency_hz", "mean_psd_db"])
        for freq, value in zip(curve.freqs_hz, curve.psd_db):
            writer.writerow([f"{float(freq):.6f}", f"{float(value):.6f}"])
    curve.row.curve_csv = str(path)
    return str(path)


def draw_centered_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont, fill: str) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.text((xy[0] - (bbox[2] - bbox[0]) // 2, xy[1]), text, font=font, fill=fill)


def draw_rotated_label(
    image: Image.Image,
    center: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
) -> None:
    bbox = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0] + 18
    height = bbox[3] - bbox[1] + 18
    layer = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    layer_draw = ImageDraw.Draw(layer)
    layer_draw.text((9 - bbox[0], 9 - bbox[1]), text, font=font, fill=fill)
    rotated = layer.rotate(90, expand=True)
    image.paste(rotated, (center[0] - rotated.width // 2, center[1] - rotated.height // 2), rotated)


def nice_axis_limits(values: np.ndarray, target_ticks: int = 7) -> tuple[float, float, np.ndarray]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return -160.0, 0.0, np.linspace(-160.0, 0.0, target_ticks)
    raw_min = float(np.min(finite))
    raw_max = float(np.max(finite))
    span = max(raw_max - raw_min, 1.0)
    padded_min = raw_min - max(2.0, span * 0.08)
    padded_max = raw_max + max(2.0, span * 0.08)
    raw_step = max((padded_max - padded_min) / max(target_ticks - 1, 1), 1e-9)
    magnitude = 10.0 ** math.floor(math.log10(raw_step))
    step = next(candidate * magnitude for candidate in (1.0, 2.0, 5.0, 10.0) if raw_step <= candidate * magnitude)
    y_min = math.floor(padded_min / step) * step
    y_max = math.ceil(padded_max / step) * step
    ticks = np.arange(y_min, y_max + step * 0.5, step)
    if ticks.size > 9:
        step *= 2.0
        y_min = math.floor(padded_min / step) * step
        y_max = math.ceil(padded_max / step) * step
        ticks = np.arange(y_min, y_max + step * 0.5, step)
    return float(y_min), float(y_max), ticks


def plot_group(curves: list[CurveData], output_path: Path, title: str, subtitle: str) -> None:
    canvas_w, canvas_h = 1900, 1120
    left, top, plot_w, plot_h = 220, 150, 1180, 790
    right, bottom = left + plot_w, top + plot_h
    image = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(image)
    title_font = find_font(42)
    label_font = find_font(24)
    tick_font = find_font(18)
    legend_font = find_font(18)
    small_font = find_font(17)

    draw.text((70, 42), title, font=title_font, fill="#111111")
    draw.text((72, 100), subtitle, font=label_font, fill="#444444")

    finite_chunks = [curve.psd_db[np.isfinite(curve.psd_db)] for curve in curves if curve.psd_db.size]
    finite_values = np.concatenate(finite_chunks) if finite_chunks else np.array([], dtype=float)
    y_min, y_max, y_ticks = nice_axis_limits(finite_values)
    x_min, x_max = 0.0, FREQ_MAX_HZ

    draw.rectangle((left, top, right, bottom), outline="#111111", width=2)
    for tick in np.linspace(x_min, x_max, 6):
        x = left + int(round((tick - x_min) / (x_max - x_min) * plot_w))
        draw.line((x, top, x, bottom), fill="#e5e7eb", width=1)
        draw.line((x, bottom, x, bottom + 8), fill="#111111", width=2)
        draw_centered_text(draw, (x, bottom + 14), f"{tick:.0f}", tick_font, "#111111")
    for tick in y_ticks:
        y = bottom - int(round((tick - y_min) / max(y_max - y_min, 1e-9) * plot_h))
        draw.line((left, y, right, y), fill="#e5e7eb", width=1)
        draw.line((left - 8, y, left, y), fill="#111111", width=2)
        label = f"{tick:.0f}"
        bbox = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((left - 16 - (bbox[2] - bbox[0]), y - 10), label, font=tick_font, fill="#111111")

    colors = [
        "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
        "#17becf", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
    ]
    for index, curve in enumerate(curves):
        color = "#111111" if curve.row.source_phase == "底噪" else colors[index % len(colors)]
        width = 4 if curve.row.source_phase == "底噪" else 3
        x = left + ((curve.freqs_hz - x_min) / (x_max - x_min) * plot_w)
        y = bottom - ((curve.psd_db - y_min) / max(y_max - y_min, 1e-9) * plot_h)
        points = [
            (int(round(px)), int(round(py)))
            for px, py, ok in zip(x, y, np.isfinite(curve.psd_db))
            if bool(ok) and left <= px <= right and top <= py <= bottom
        ]
        if len(points) >= 2:
            draw.line(points, fill=color, width=width)

    draw_centered_text(draw, ((left + right) // 2, bottom + 58), "Frequency (Hz)", label_font, "#111111")
    draw_rotated_label(image, (54, (top + bottom) // 2), "PSD (dB re 1 rad²/Hz)", label_font, "#111111")

    legend_x, legend_y = right + 42, top + 8
    draw.text((legend_x, legend_y), "实验组", font=label_font, fill="#111111")
    legend_y += 44
    for index, curve in enumerate(curves):
        color = "#111111" if curve.row.source_phase == "底噪" else colors[index % len(colors)]
        y = legend_y + index * 34
        draw.line((legend_x, y + 12, legend_x + 32, y + 12), fill=color, width=4)
        legend_text = f"底噪 {curve.row.experiment_id}" if curve.row.source_phase == "底噪" else curve.row.experiment_id
        draw.text((legend_x + 42, y), legend_text, font=legend_font, fill="#111111")

    draw.text((70, canvas_h - 78), "PSD算法: tdas_dat_viewer.compute_spatial_segment_psd；空间平均 mean_psd；Hann窗2s，50%重叠；单位 dB re 1 rad²/Hz。", font=small_font, fill="#555555")
    draw.text((70, canvas_h - 46), "取样: 稳定曲线取各组稳定阶段中心1分钟；底噪曲线取该大类第一组底噪阶段中心1分钟；频率显示0-500Hz。", font=small_font, fill="#555555")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def write_summary_csv(path: Path, rows: list[CurveRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(CurveRow.__annotations__.keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="空间段PSD稳定阶段中间1分钟对比")
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = [record for record in read_records(args.workbook) if record.group in EXPERIMENT_TYPES]
    files_by_device = {
        device.title: index_data_files(args.data_root / device.data_subdir)
        for device in DEVICE_CONFIGS
    }

    curves: list[CurveData] = []
    rows: list[CurveRow] = []
    for record in records:
        for device in DEVICE_CONFIGS:
            for segment_name, start_m, end_m in SPATIAL_SEGMENTS[device.title]:
                if device.title == "silixia" and record.seq == "G5_单相气3m3h":
                    rows.append(
                        CurveRow(
                            experiment_type=record.group,
                            experiment_id=record.seq,
                            source_phase="",
                            device=device.title,
                            segment_name=segment_name,
                            requested_range_m=f"{start_m:g}-{end_m:g}",
                            actual_range_m="",
                            stable_start="",
                            stable_end="",
                            analysis_start="",
                            analysis_end="",
                            sample_rate_hz="",
                            selected_channel_count=0,
                            valid_channel_count=0,
                            window_count=0,
                            status="skipped",
                            message="跳过：用户确认最后一个实验数据丢失",
                        )
                    )
                    continue
                try:
                    curve = compute_curve(record, device.title, segment_name, start_m, end_m, files_by_device[device.title])
                    write_curve_csv(curve, args.output_dir)
                    curves.append(curve)
                    rows.append(curve.row)
                    print(f"ok {record.group} {record.seq} {device.title} {segment_name}", flush=True)
                except Exception as exc:
                    phase = stable_phase(record)
                    rows.append(
                        CurveRow(
                            experiment_type=record.group,
                            experiment_id=record.seq,
                            source_phase="稳定",
                            device=device.title,
                            segment_name=segment_name,
                            requested_range_m=f"{start_m:g}-{end_m:g}",
                            actual_range_m="",
                            stable_start=phase[0].isoformat(sep=" ") if phase else "",
                            stable_end=phase[1].isoformat(sep=" ") if phase else "",
                            analysis_start="",
                            analysis_end="",
                            sample_rate_hz="",
                            selected_channel_count=0,
                            valid_channel_count=0,
                            window_count=0,
                            status="failed",
                            message=str(exc),
                        )
                    )
                    print(f"failed {record.group} {record.seq} {device.title} {segment_name}: {exc}", flush=True)

    for experiment_type in EXPERIMENT_TYPES:
        noise_record = next(
            (record for record in records if record.group == experiment_type and phase_interval(record, "底噪") is not None),
            None,
        )
        if noise_record is None:
            continue
        for device in DEVICE_CONFIGS:
            for segment_name, start_m, end_m in SPATIAL_SEGMENTS[device.title]:
                try:
                    curve = compute_curve(
                        noise_record,
                        device.title,
                        segment_name,
                        start_m,
                        end_m,
                        files_by_device[device.title],
                        phase_label="底噪",
                    )
                    write_curve_csv(curve, args.output_dir)
                    curves.append(curve)
                    rows.append(curve.row)
                    print(f"ok {experiment_type} {noise_record.seq} {device.title} {segment_name} 底噪", flush=True)
                except Exception as exc:
                    phase = phase_interval(noise_record, "底噪")
                    rows.append(
                        CurveRow(
                            experiment_type=experiment_type,
                            experiment_id=noise_record.seq,
                            source_phase="底噪",
                            device=device.title,
                            segment_name=segment_name,
                            requested_range_m=f"{start_m:g}-{end_m:g}",
                            actual_range_m="",
                            stable_start=phase[0].isoformat(sep=" ") if phase else "",
                            stable_end=phase[1].isoformat(sep=" ") if phase else "",
                            analysis_start="",
                            analysis_end="",
                            sample_rate_hz="",
                            selected_channel_count=0,
                            valid_channel_count=0,
                            window_count=0,
                            status="failed",
                            message=str(exc),
                        )
                    )
                    print(f"failed {experiment_type} {noise_record.seq} {device.title} {segment_name} 底噪: {exc}", flush=True)

    figures: list[dict[str, str]] = []
    for experiment_type in EXPERIMENT_TYPES:
        for device in DEVICE_CONFIGS:
            for segment_name, start_m, end_m in SPATIAL_SEGMENTS[device.title]:
                group_curves = [
                    curve for curve in curves
                    if curve.row.experiment_type == experiment_type
                    and curve.row.device == device.title
                    and curve.row.segment_name == segment_name
                ]
                group_curves.sort(key=lambda curve: (0 if curve.row.source_phase == "底噪" else 1, curve.row.experiment_id))
                if not group_curves:
                    continue
                output_path = (
                    args.output_dir
                    / "figures"
                    / safe_filename(experiment_type)
                    / f"{safe_filename(experiment_type)}_{safe_filename(device.title)}_{safe_filename(segment_name)}_{start_m:g}-{end_m:g}m_稳定中间1分钟_含底噪_PSD对比.png"
                )
                title = f"{experiment_type} | {device.title} | {segment_name} {start_m:g}-{end_m:g} m"
                subtitle = f"稳定阶段中心1分钟 + 底噪中心1分钟 | 空间平均PSD | 0-500 Hz | {len(group_curves)} 条曲线"
                plot_group(group_curves, output_path, title, subtitle)
                figures.append(
                    {
                        "experiment_type": experiment_type,
                        "device": device.title,
                        "segment_name": segment_name,
                        "range_m": f"{start_m:g}-{end_m:g}",
                        "image": str(output_path),
                        "curve_count": str(len(group_curves)),
                    }
                )

    write_summary_csv(args.output_dir / "空间段PSD比对_曲线汇总.csv", rows)
    (args.output_dir / "空间段PSD比对_图像清单.json").write_text(
        json.dumps(figures, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"RESULT curves={len(curves)} figures={len(figures)} output={args.output_dir}", flush=True)
    return 0 if any(row.status == "ok" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
