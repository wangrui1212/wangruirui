#!/usr/bin/env python
"""Spatial segment PSD comparisons for the 2026-07-09 well-flow tests."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from batch_well_flow_three_device_analysis import (
    DEFAULT_DATA_ROOT,
    DEFAULT_WORKBOOK,
    DEVICE_CONFIGS,
    DatFile,
    DatHeader,
    ExperimentRecord,
    index_data_files,
    parse_time_value,
    read_header,
    read_records,
    safe_filename,
)
from tdas_dat_viewer import (
    compute_spatial_segment_psd,
    normalize_silixa_matrix,
    psd_to_db,
    read_signal_matrix,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "井中流动分析" / "空间段PSD谱线_20260709稳定中间1分钟_含底噪"

SPATIAL_SEGMENTS: dict[str, tuple[tuple[str, float, float], ...]] = {
    "silixia": (
        ("筒内光纤", 1048.0, 1060.0),
        ("筒外缆线", 1104.0, 1116.0),
    ),
    "OV": (
        ("筒内光纤", 320.0, 330.0),
        ("筒外缆线", 376.0, 392.0),
    ),
    "自研_2U": (
        ("筒外光缆", 348.0, 358.0),
    ),
}

FREQ_MAX_HZ = 500.0
WINDOW_SECONDS = 2.0
OVERLAP_PERCENT = 50.0
WINDOW_NAME = "Hann"


@dataclass(frozen=True)
class WindowRecord:
    start: datetime
    end: datetime


@dataclass(frozen=True)
class FileSegment:
    item: DatFile
    header: DatHeader
    start_index: int
    stop_index: int
    global_start_index: int

    @property
    def sample_count(self) -> int:
        return self.stop_index - self.start_index


@dataclass
class CurveRow:
    experiment_group: str
    experiment_id: str
    curve_type: str
    device: str
    segment_name: str
    requested_range_m: str
    actual_range_m: str
    source_start: str
    source_end: str
    analysis_start: str
    analysis_end: str
    flow_lh: float | str
    sample_rate_hz: int | str
    selected_channel_count: int
    valid_channel_count: int
    window_count: int
    file_count: int
    first_file: str
    last_file: str
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


def middle_datetime(record: ExperimentRecord) -> datetime:
    return datetime.combine(record.start.date(), parse_time_value(record.middle_text))


def stable_analysis_window(record: ExperimentRecord) -> tuple[datetime, datetime, datetime, datetime]:
    source_start = middle_datetime(record)
    source_end = record.end
    duration_s = (source_end - source_start).total_seconds()
    if duration_s < 60.0:
        raise ValueError(f"稳定/观察段不足60秒: {duration_s:.1f}s")
    center = source_start + timedelta(seconds=duration_s / 2.0)
    return source_start, source_end, center - timedelta(seconds=30), center + timedelta(seconds=30)


def noise_analysis_window(record: ExperimentRecord) -> tuple[datetime, datetime, datetime, datetime]:
    source_start = record.start
    source_end = record.start + timedelta(minutes=5)
    return source_start, source_end, record.start + timedelta(minutes=2), record.start + timedelta(minutes=3)


def validate_compatible(headers: Sequence[DatHeader]) -> DatHeader:
    if not headers:
        raise ValueError("没有文件头可校验")
    base = headers[0]
    for header in headers[1:]:
        checks = (
            (header.protocol == base.protocol, "协议"),
            (header.point_count == base.point_count, "点位数"),
            (header.frequency_hz == base.frequency_hz, "采样率"),
            (abs(header.position_start_m - base.position_start_m) <= 1e-6, "起始位置"),
            (abs(header.position_end_m - base.position_end_m) <= 1e-6, "结束位置"),
            (abs(header.point_interval_m - base.point_interval_m) <= 1e-6, "点位间距"),
        )
        for ok, label in checks:
            if not ok:
                raise ValueError(f"同一PSD窗口内文件{label}不一致")
    return base


def build_segments(files: list[DatFile], window: WindowRecord) -> list[FileSegment]:
    timestamps = [item.timestamp for item in files]
    first = max(0, bisect.bisect_right(timestamps, window.start) - 1)
    segments: list[FileSegment] = []
    stitched_samples = 0
    target_samples: int | None = None
    for item in files[first:]:
        header = read_header(item.path)
        rate = max(1, int(header.frequency_hz))
        if target_samples is None:
            target_samples = int(round((window.end - window.start).total_seconds() * rate))
        file_end = item.timestamp + timedelta(seconds=header.sample_count / rate)
        if file_end <= window.start:
            continue
        remaining = max(0, target_samples - stitched_samples)
        if remaining <= 0:
            break
        local_start = max(0, int(math.ceil((window.start - item.timestamp).total_seconds() * rate - 1e-9))) if not segments else 0
        local_stop = min(header.sample_count, local_start + remaining)
        if local_stop <= local_start:
            continue
        segments.append(FileSegment(item, header, local_start, local_stop, stitched_samples))
        stitched_samples += local_stop - local_start
        if target_samples is not None and stitched_samples >= target_samples:
            break
    if not segments:
        return []
    validate_compatible([segment.header for segment in segments])
    return segments


def load_stitched_matrix(segments: list[FileSegment]) -> np.ndarray:
    matrices: list[np.ndarray] = []
    for segment in segments:
        matrix = read_signal_matrix(segment.item.path, segment.header).astype(np.float32, copy=False)
        matrices.append(matrix[:, segment.start_index : segment.stop_index])
    if not matrices:
        raise ValueError("没有可拼接的数据段")
    return matrices[0] if len(matrices) == 1 else np.concatenate(matrices, axis=1)


def ensure_20260709(segments: list[FileSegment]) -> None:
    bad = [segment.item.path.name for segment in segments if segment.item.timestamp.strftime("%Y%m%d") != "20260709"]
    if bad:
        raise ValueError(f"PSD窗口包含非20260709文件: {'; '.join(bad[:3])}")


def compute_curve(
    record: ExperimentRecord,
    device_title: str,
    segment_name: str,
    start_m: float,
    end_m: float,
    files: list[DatFile],
    curve_type: str,
) -> CurveData:
    if curve_type == "底噪":
        source_start, source_end, win_start, win_end = noise_analysis_window(record)
        experiment_id = "底噪"
        flow_lh: float | str = ""
    else:
        source_start, source_end, win_start, win_end = stable_analysis_window(record)
        experiment_id = record.seq
        flow_lh = float(record.flow_lh)

    segments = build_segments(files, WindowRecord(win_start, win_end))
    if not segments:
        raise ValueError(f"{curve_type}窗口没有可用数据")
    ensure_20260709(segments)
    matrix = load_stitched_matrix(segments)
    header = segments[0].header
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
    psd_db = psd_to_db(result.mean_psd)[mask]
    row = CurveRow(
        experiment_group=record.group,
        experiment_id=experiment_id,
        curve_type=curve_type,
        device=device_title,
        segment_name=segment_name,
        requested_range_m=f"{start_m:g}-{end_m:g}",
        actual_range_m=f"{result.actual_start_m:.3f}-{result.actual_end_m:.3f}",
        source_start=source_start.isoformat(sep=" "),
        source_end=source_end.isoformat(sep=" "),
        analysis_start=win_start.isoformat(sep=" "),
        analysis_end=win_end.isoformat(sep=" "),
        flow_lh=flow_lh,
        sample_rate_hz=result.sample_rate_hz,
        selected_channel_count=result.selected_channel_count,
        valid_channel_count=result.valid_channel_count,
        window_count=result.window_count,
        file_count=len(segments),
        first_file=segments[0].item.path.name,
        last_file=segments[-1].item.path.name,
        status="ok",
    )
    return CurveData(row, result.freqs_hz[mask], psd_db)


def write_curve_csv(curve: CurveData, output_dir: Path) -> None:
    curve_dir = output_dir / "curves" / safe_filename(curve.row.experiment_group) / safe_filename(curve.row.device)
    curve_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"{safe_filename(curve.row.experiment_group)}_"
        f"{safe_filename(curve.row.experiment_id)}_"
        f"{safe_filename(curve.row.curve_type)}_"
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


def draw_centered_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont, fill: str) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.text((xy[0] - (bbox[2] - bbox[0]) // 2, xy[1]), text, font=font, fill=fill)


def draw_rotated_label(image: Image.Image, center: tuple[int, int], text: str, font: ImageFont.ImageFont, fill: str) -> None:
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

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
    regular_index = 0
    legend_items: list[tuple[str, str, int]] = []
    for curve in curves:
        if curve.row.curve_type == "底噪":
            color = "#111111"
            width = 4
            legend_label = "底噪"
        else:
            color = colors[regular_index % len(colors)]
            width = 3
            regular_index += 1
            legend_label = f"{curve.row.experiment_id} | {curve.row.flow_lh:g} L/h"
        x = left + ((curve.freqs_hz - x_min) / (x_max - x_min) * plot_w)
        y = bottom - ((curve.psd_db - y_min) / max(y_max - y_min, 1e-9) * plot_h)
        points = [
            (int(round(px)), int(round(py)))
            for px, py, ok in zip(x, y, np.isfinite(curve.psd_db))
            if bool(ok) and left <= px <= right and top <= py <= bottom
        ]
        if len(points) >= 2:
            draw.line(points, fill=color, width=width)
        legend_items.append((legend_label, color, width))

    draw_centered_text(draw, ((left + right) // 2, bottom + 58), "Frequency (Hz)", label_font, "#111111")
    draw_rotated_label(image, (54, (top + bottom) // 2), "PSD (dB re 1 rad²/Hz)", label_font, "#111111")

    legend_x, legend_y = right + 42, top + 8
    draw.text((legend_x, legend_y), "曲线", font=label_font, fill="#111111")
    legend_y += 44
    for index, (legend_text, color, width) in enumerate(legend_items):
        y = legend_y + index * 34
        draw.line((legend_x, y + 12, legend_x + 32, y + 12), fill=color, width=width)
        draw.text((legend_x + 42, y), legend_text, font=legend_font, fill="#111111")

    draw.text((70, canvas_h - 78), "PSD算法: tdas_dat_viewer.compute_spatial_segment_psd；空间平均 mean_psd；Hann窗2s，50%重叠；单位 dB re 1 rad²/Hz。", font=small_font, fill="#555555")
    draw.text((70, canvas_h - 46), "取样: 稳定曲线取中间时间至结束/观察时间的中心1分钟；底噪取本组头5分钟的中心1分钟；频率显示0-500Hz。", font=small_font, fill="#555555")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def write_summary_csv(path: Path, rows: list[CurveRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CurveRow.__annotations__.keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def failure_row(
    record: ExperimentRecord,
    device: str,
    segment_name: str,
    start_m: float,
    end_m: float,
    curve_type: str,
    message: str,
) -> CurveRow:
    if curve_type == "底噪":
        source_start, source_end, win_start, win_end = noise_analysis_window(record)
        experiment_id = "底噪"
        flow_lh: float | str = ""
    else:
        try:
            source_start, source_end, win_start, win_end = stable_analysis_window(record)
        except Exception:
            source_start, source_end, win_start, win_end = record.start, record.end, record.start, record.end
        experiment_id = record.seq
        flow_lh = float(record.flow_lh)
    return CurveRow(
        experiment_group=record.group,
        experiment_id=experiment_id,
        curve_type=curve_type,
        device=device,
        segment_name=segment_name,
        requested_range_m=f"{start_m:g}-{end_m:g}",
        actual_range_m="",
        source_start=source_start.isoformat(sep=" "),
        source_end=source_end.isoformat(sep=" "),
        analysis_start=win_start.isoformat(sep=" "),
        analysis_end=win_end.isoformat(sep=" "),
        flow_lh=flow_lh,
        sample_rate_hz="",
        selected_channel_count=0,
        valid_channel_count=0,
        window_count=0,
        file_count=0,
        first_file="",
        last_file="",
        status="failed",
        message=message,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="2026-07-09井中流动空间段PSD谱线对比")
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = read_records(args.workbook)
    files_by_device = {
        device.title: index_data_files(args.data_root / device.data_subdir)
        for device in DEVICE_CONFIGS
    }

    curves: list[CurveData] = []
    rows: list[CurveRow] = []
    for record in records:
        for device in DEVICE_CONFIGS:
            for segment_name, start_m, end_m in SPATIAL_SEGMENTS[device.title]:
                try:
                    curve = compute_curve(record, device.title, segment_name, start_m, end_m, files_by_device[device.title], "稳定")
                    write_curve_csv(curve, args.output_dir)
                    curves.append(curve)
                    rows.append(curve.row)
                    print(f"ok {record.group} {record.seq} {device.title} {segment_name}", flush=True)
                except Exception as exc:
                    rows.append(failure_row(record, device.title, segment_name, start_m, end_m, "稳定", str(exc)))
                    print(f"failed {record.group} {record.seq} {device.title} {segment_name}: {exc}", flush=True)

    first_records = {record.group: record for record in records if record.seq in {"1", "A"}}
    for group_name, record in first_records.items():
        for device in DEVICE_CONFIGS:
            for segment_name, start_m, end_m in SPATIAL_SEGMENTS[device.title]:
                try:
                    curve = compute_curve(record, device.title, segment_name, start_m, end_m, files_by_device[device.title], "底噪")
                    write_curve_csv(curve, args.output_dir)
                    curves.append(curve)
                    rows.append(curve.row)
                    print(f"ok {group_name} 底噪 {device.title} {segment_name}", flush=True)
                except Exception as exc:
                    rows.append(failure_row(record, device.title, segment_name, start_m, end_m, "底噪", str(exc)))
                    print(f"failed {group_name} 底噪 {device.title} {segment_name}: {exc}", flush=True)

    figures: list[dict[str, str]] = []
    for group_name in ("第一组实验", "第二组实验"):
        for device in DEVICE_CONFIGS:
            for segment_name, start_m, end_m in SPATIAL_SEGMENTS[device.title]:
                group_curves = [
                    curve for curve in curves
                    if curve.row.experiment_group == group_name
                    and curve.row.device == device.title
                    and curve.row.segment_name == segment_name
                ]
                group_curves.sort(key=lambda curve: (0 if curve.row.curve_type == "底噪" else 1, curve.row.experiment_id))
                if not group_curves:
                    continue
                actual_range = next((curve.row.actual_range_m for curve in group_curves if curve.row.actual_range_m), "")
                output_path = (
                    args.output_dir
                    / "figures"
                    / safe_filename(group_name)
                    / f"{safe_filename(group_name)}_{safe_filename(device.title)}_{safe_filename(segment_name)}_{start_m:g}-{end_m:g}m_稳定中间1分钟_含底噪_PSD对比.png"
                )
                title = f"{group_name} | {device.title} | {segment_name} {start_m:g}-{end_m:g} m"
                subtitle = f"实际点位 {actual_range} m | 稳定中心1分钟 + 底噪中心1分钟 | 空间平均PSD | 0-500 Hz"
                plot_group(group_curves, output_path, title, subtitle)
                figures.append(
                    {
                        "experiment_group": group_name,
                        "device": device.title,
                        "segment_name": segment_name,
                        "requested_range_m": f"{start_m:g}-{end_m:g}",
                        "actual_range_m": actual_range,
                        "image": str(output_path),
                        "curve_count": str(len(group_curves)),
                    }
                )

    summary_path = args.output_dir / "空间段PSD比对_曲线汇总.csv"
    manifest_path = args.output_dir / "空间段PSD比对_图像清单.json"
    write_summary_csv(summary_path, rows)
    manifest_path.write_text(json.dumps(figures, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"SUMMARY_CSV {summary_path}", flush=True)
    print(f"FIGURE_MANIFEST {manifest_path}", flush=True)
    print(f"RESULT curves={len(curves)} figures={len(figures)} output={args.output_dir}", flush=True)
    return 0 if figures else 1


if __name__ == "__main__":
    raise SystemExit(main())
