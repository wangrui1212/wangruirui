#!/usr/bin/env python
"""Batch point-analysis export for the 2026-06-30 40ns/200M DAS workbook."""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np

from batch_point_analysis_from_excel import is_number_like, read_sheet_rows
from point_analysis_export import load_concatenated_matrix
from tdas_dat_viewer import calculate_peak_line_snr, read_header


DEFAULT_EXCEL = Path("outputs/image_to_excel/DAS测试记录表格_20260630_40ns_200M频移.xlsx")
DEFAULT_DATA_ROOT = Path(r"E:\新方案das数据分析\20260630")
DEFAULT_OUTPUT_DIR = Path("outputs/point_analysis_20260630_40ns_200M")
DEFAULT_DATE = date(2026, 6, 30)
DEFAULT_POSITION_M = 8141.32
DEFAULT_SHEET = "40ns_200M频移"
TIME_RE = re.compile(r"(\d{1,2})\s*[:：]\s*(\d{2})\s*[:：]\s*(\d{2})")
DAT_RE = re.compile(r"(?P<stamp>\d{14})\.\d+\.dat$", re.IGNORECASE)


@dataclass(frozen=True)
class TestRecord:
    serial: int
    row_number: int
    frequency_hz: str
    pzt_mvpp: str
    excel_peak_db: str
    excel_noise_db: str
    sampling_hz_record: str
    pulse_width_ns: str
    fiber_length_km: str
    remark: str


@dataclass(frozen=True)
class DatFile:
    path: Path
    timestamp: datetime


def text(value: object) -> str:
    return "" if value is None else str(value).strip()


def fmt_number(value: object) -> str:
    raw = text(value)
    if not raw:
        return ""
    try:
        number = float(raw)
    except ValueError:
        return raw
    return str(int(number)) if number.is_integer() else str(number)


def iter_records(excel_path: Path, sheet_name: str) -> list[TestRecord]:
    records: list[TestRecord] = []
    for row in read_sheet_rows(excel_path, sheet_name):
        row_number = int(row.get(0, 0) or 0)
        frequency = row.get(1)
        pzt = row.get(2)
        remark = text(row.get(8))
        if row_number < 6 or not is_number_like(frequency) or not is_number_like(pzt) or not remark:
            continue
        records.append(
            TestRecord(
                serial=len(records) + 1,
                row_number=row_number,
                frequency_hz=fmt_number(frequency),
                pzt_mvpp=fmt_number(pzt),
                excel_peak_db=fmt_number(row.get(3)),
                excel_noise_db=fmt_number(row.get(4)),
                sampling_hz_record=fmt_number(row.get(5)),
                pulse_width_ns=fmt_number(row.get(6)),
                fiber_length_km=fmt_number(row.get(7)),
                remark=remark,
            )
        )
    return records


def parse_time_range(value: str, test_date: date) -> tuple[datetime, datetime] | None:
    matches = list(TIME_RE.finditer(value))
    if len(matches) < 2:
        return None

    def to_time(match: re.Match[str]) -> time:
        return time(int(match.group(1)), int(match.group(2)), int(match.group(3)))

    start = datetime.combine(test_date, to_time(matches[0]))
    end = datetime.combine(test_date, to_time(matches[1]))
    if end < start:
        start, end = end, start
    return (start, end) if end > start else None


def parse_dat_timestamp(path: Path) -> datetime | None:
    match = DAT_RE.search(path.name)
    if not match:
        return None
    return datetime.strptime(match.group("stamp"), "%Y%m%d%H%M%S")


def minute_dir_candidates(data_root: Path, minute: datetime) -> list[Path]:
    return [
        data_root / f"{minute:%H}" / f"{minute:%M}",
        data_root / f"{minute:%M}",
    ]


def find_candidate_files(data_root: Path, start: datetime, end: datetime) -> list[DatFile]:
    candidates: list[DatFile] = []
    seen_dirs: set[Path] = set()
    current = start.replace(second=0, microsecond=0)
    end_minute = end.replace(second=0, microsecond=0)
    while current <= end_minute:
        for minute_dir in minute_dir_candidates(data_root, current):
            if minute_dir in seen_dirs or not minute_dir.exists():
                continue
            seen_dirs.add(minute_dir)
            for path in minute_dir.glob("*.dat"):
                timestamp = parse_dat_timestamp(path)
                if timestamp is not None and start <= timestamp <= end:
                    candidates.append(DatFile(path, timestamp))
        current += timedelta(minutes=1)
    return sorted(candidates, key=lambda item: item.timestamp)


def choose_middle_files(files: list[DatFile], start: datetime, end: datetime, count: int) -> list[DatFile]:
    if len(files) <= count:
        return files
    midpoint = start + (end - start) / 2
    nearest_index = min(range(len(files)), key=lambda index: abs(files[index].timestamp - midpoint))
    left = nearest_index - count // 2
    right = left + count
    if left < 0:
        left = 0
        right = count
    if right > len(files):
        right = len(files)
        left = right - count
    return files[left:right]


def nearest_point(path: Path, target_position_m: float) -> tuple[int, float, float, int]:
    header = read_header(path)
    raw_index = round((target_position_m - header.position_start_m) / header.point_interval_m)
    point_index = int(np.clip(raw_index, 0, header.point_count - 1))
    distance_m = header.position_start_m + point_index * header.point_interval_m
    return point_index, distance_m, abs(distance_m - target_position_m), header.frequency_hz


def excel_snr(record: TestRecord) -> str:
    try:
        return f"{float(record.excel_peak_db) - float(record.excel_noise_db):.2f}"
    except ValueError:
        return ""


def sanitize(value: str) -> str:
    return re.sub(r"[^\w.-]+", "_", value, flags=re.UNICODE).strip("_") or "blank"


def image_name(record: TestRecord, point_index: int, selected: list[DatFile]) -> str:
    return (
        f"freq_{sanitize(record.frequency_hz)}Hz_PZT_{sanitize(record.pzt_mvpp)}mVpp_"
        f"point_{point_index}_{selected[0].timestamp:%H%M%S}_{selected[-1].timestamp:%H%M%S}.png"
    )


def default_export_python(project_root: Path) -> Path:
    candidate = project_root / ".venv-build" / "Scripts" / "python.exe"
    return candidate if candidate.exists() else Path(sys.executable)


def export_image(
    export_python: Path,
    export_script: Path,
    files: list[DatFile],
    point_index: int,
    output: Path,
    target_frequency_hz: float,
    snr_noise_mode: str,
    exclude_band_hz: float,
) -> subprocess.CompletedProcess[str]:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(export_python),
        str(export_script),
        *[str(item.path) for item in files],
        "--point",
        str(point_index),
        "--output",
        str(output),
        "--width",
        "1200",
        "--height",
        "850",
        "--target-frequency",
        str(target_frequency_hz),
        "--search-half-width",
        "5",
        "--snr-noise-mode",
        snr_noise_mode,
        "--exclude-band-hz",
        str(exclude_band_hz),
    ]
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    return subprocess.run(command, text=True, capture_output=True, env=env)


def compute_snr(
    files: list[DatFile],
    point_index: int,
    target_frequency_hz: float,
    snr_noise_mode: str,
    exclude_band_hz: float,
) -> tuple[dict[str, float] | None, int]:
    matrix, header = load_concatenated_matrix([item.path for item in files])
    trace = matrix[point_index, :].copy()
    n = 1 << (trace.size.bit_length() - 1)
    samples = trace[:n] - np.mean(trace[:n])
    window_arr = np.hanning(n).astype(np.float32)
    spectrum = np.fft.rfft(samples * window_arr)
    freqs = np.fft.rfftfreq(n, d=1.0 / max(1, header.frequency_hz))
    magnitude = np.abs(spectrum) * 2.0 / n
    return calculate_peak_line_snr(
        freqs,
        magnitude,
        mode="target",
        target_freq_hz=target_frequency_hz,
        search_half_width_hz=5.0,
        noise_mode=snr_noise_mode,
        exclude_band_hz=exclude_band_hz,
    ), header.frequency_hz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export 20260630 DAS point-analysis images and SNR summary.")
    parser.add_argument("--excel", type=Path, default=DEFAULT_EXCEL)
    parser.add_argument("--sheet", default=DEFAULT_SHEET)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--date", default=DEFAULT_DATE.isoformat())
    parser.add_argument("--position-m", type=float, default=DEFAULT_POSITION_M)
    parser.add_argument("--window-seconds", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--snr-noise-mode", choices=["local", "global"], default="local")
    parser.add_argument("--exclude-band-hz", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    excel_path = args.excel if args.excel.is_absolute() else project_root / args.excel
    output_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    image_dir = output_dir / "images"
    summary_path = output_dir / "point_snr_summary.csv"
    export_python = default_export_python(project_root)
    export_script = project_root / "point_analysis_export.py"
    test_date = date.fromisoformat(args.date)

    records = iter_records(excel_path, args.sheet)
    if args.limit is not None:
        records = records[: args.limit]
    output_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "serial",
        "row_number",
        "frequency_hz",
        "pzt_mvpp",
        "excel_peak_db",
        "excel_noise_db",
        "excel_snr_db",
        "sampling_hz_record",
        "pulse_width_ns",
        "fiber_length_km",
        "time_range",
        "remark",
        "target_position_m",
        "point_index",
        "distance_m",
        "distance_error_m",
        "selected_files",
        "sampling_hz",
        "peak_line_snr_db",
        "peak_freq_hz",
        "peak_value_db",
        "noise_level_db",
        "global_noise_level_db",
        "global_peak_snr_db",
        "noise_mode",
        "exclude_band_hz",
        "image",
        "status",
        "status_note",
    ]

    exported = 0
    skipped = 0
    with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            parsed = parse_time_range(record.remark, test_date)
            time_range = ""
            if parsed is not None:
                time_range = f"{parsed[0]:%H:%M:%S}-{parsed[1]:%H:%M:%S}"
            base = {
                "serial": record.serial,
                "row_number": record.row_number,
                "frequency_hz": record.frequency_hz,
                "pzt_mvpp": record.pzt_mvpp,
                "excel_peak_db": record.excel_peak_db,
                "excel_noise_db": record.excel_noise_db,
                "excel_snr_db": excel_snr(record),
                "sampling_hz_record": record.sampling_hz_record,
                "pulse_width_ns": record.pulse_width_ns,
                "fiber_length_km": record.fiber_length_km,
                "time_range": time_range,
                "remark": record.remark,
                "target_position_m": f"{args.position_m:.3f}",
                "point_index": "",
                "distance_m": "",
                "distance_error_m": "",
                "selected_files": "",
                "sampling_hz": "",
                "peak_line_snr_db": "",
                "peak_freq_hz": "",
                "peak_value_db": "",
                "noise_level_db": "",
                "global_noise_level_db": "",
                "global_peak_snr_db": "",
                "noise_mode": args.snr_noise_mode,
                "exclude_band_hz": f"{args.exclude_band_hz:.3f}",
                "image": "",
                "status": "",
                "status_note": "",
            }
            if parsed is None:
                skipped += 1
                writer.writerow(base | {"status": "skipped", "status_note": "invalid time range"})
                continue
            start, end = parsed
            candidates = find_candidate_files(args.data_root, start, end)
            selected = choose_middle_files(candidates, start, end, args.window_seconds)
            if len(selected) < args.window_seconds:
                skipped += 1
                writer.writerow(base | {"status": "skipped", "status_note": f"only {len(selected)} files found"})
                continue

            point_index, distance_m, distance_error_m, sampling_hz = nearest_point(selected[0].path, args.position_m)
            output = image_dir / image_name(record, point_index, selected)
            row = base | {
                "point_index": point_index,
                "distance_m": f"{distance_m:.3f}",
                "distance_error_m": f"{distance_error_m:.3f}",
                "selected_files": ";".join(str(item.path) for item in selected),
                "sampling_hz": sampling_hz,
                "image": str(output),
                "status_note": f"middle 10s window {selected[0].timestamp:%H:%M:%S}-{selected[-1].timestamp:%H:%M:%S}",
            }
            if args.dry_run:
                exported += 1
                writer.writerow(row | {"status": "dry_run"})
                continue

            target_frequency_hz = float(record.frequency_hz)
            result = export_image(
                export_python,
                export_script,
                selected,
                point_index,
                output,
                target_frequency_hz,
                args.snr_noise_mode,
                args.exclude_band_hz,
            )
            if result.returncode != 0:
                skipped += 1
                note = (result.stderr or result.stdout).strip().replace("\r", " ").replace("\n", " | ")
                writer.writerow(row | {"status": "failed_export", "status_note": note})
                continue
            snr, actual_sampling_hz = compute_snr(
                selected,
                point_index,
                target_frequency_hz,
                args.snr_noise_mode,
                args.exclude_band_hz,
            )
            if snr is None:
                skipped += 1
                writer.writerow(row | {"status": "failed_snr", "status_note": "no valid peak"})
                continue
            exported += 1
            writer.writerow(
                row
                | {
                    "sampling_hz": actual_sampling_hz,
                    "peak_line_snr_db": f"{snr['snr_db']:.2f}",
                    "peak_freq_hz": f"{snr['peak_freq']:.3f}",
                    "peak_value_db": f"{snr['peak_value_db']:.2f}",
                    "noise_level_db": f"{snr['noise_level_db']:.2f}",
                    "global_noise_level_db": f"{snr['noise_level_db']:.2f}" if args.snr_noise_mode == "global" else "",
                    "global_peak_snr_db": f"{snr['snr_db']:.2f}" if args.snr_noise_mode == "global" else "",
                    "noise_mode": args.snr_noise_mode,
                    "exclude_band_hz": f"{args.exclude_band_hz:.3f}",
                    "status": "exported",
                }
            )
            print(f"exported: {output}")

    print(f"records: {len(records)}, exported/planned: {exported}, skipped/failed: {skipped}")
    print(f"summary: {summary_path}")
    return 0 if exported > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
