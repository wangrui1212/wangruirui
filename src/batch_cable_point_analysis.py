#!/usr/bin/env python
"""Batch point-analysis export for the cable test workbook."""

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


DEFAULT_EXCEL = Path("outputs/cable_test/缆线测试记录整理.xlsx")
DEFAULT_DATA_ROOT = Path(r"E:\新方案das数据分析\DAS0624data-缆\20260624\15")
DEFAULT_OUTPUT_DIR = Path("outputs/cable_test/point_analysis")
DEFAULT_DATE = date(2026, 6, 24)
DEFAULT_POSITION_M = 1848.43
TIME_RE = re.compile(r"(\d{1,2}):(\d{2}):(\d{2})")
DAT_RE = re.compile(r"(?P<stamp>\d{14})\.\d+\.dat$", re.IGNORECASE)


@dataclass(frozen=True)
class CableRecord:
    row_number: int
    serial: str
    frequency_hz: str
    excel_peak_db: str
    excel_noise_db: str
    excel_snr_db: str
    time_range: str
    volume: str
    note: str


@dataclass(frozen=True)
class DatFile:
    path: Path
    timestamp: datetime


def text(value: object) -> str:
    return "" if value is None else str(value).strip()


def iter_records(excel_path: Path) -> list[CableRecord]:
    rows = read_sheet_rows(excel_path, "测试记录")
    records: list[CableRecord] = []
    for row in rows:
        row_number = int(row.get(0, 0) or 0)
        serial = text(row.get(1))
        if row_number < 4 or not serial or not serial.replace(".", "", 1).isdigit():
            continue
        records.append(
            CableRecord(
                row_number=row_number,
                serial=serial,
                frequency_hz=text(row.get(2)),
                excel_peak_db=text(row.get(3)),
                excel_noise_db=text(row.get(4)),
                excel_snr_db=text(row.get(5)),
                time_range=text(row.get(6)),
                volume=text(row.get(7)),
                note=text(row.get(9)),
            )
        )
    return records


def parse_time_range(value: str, test_date: date) -> tuple[datetime, datetime] | None:
    matches = list(TIME_RE.finditer(value))
    if len(matches) < 2:
        return None

    def as_time(match: re.Match[str]) -> time:
        return time(int(match.group(1)), int(match.group(2)), int(match.group(3)))

    start = datetime.combine(test_date, as_time(matches[0]))
    end = datetime.combine(test_date, as_time(matches[1]))
    if end < start:
        start, end = end, start
    if end <= start:
        return None
    return start, end


def parse_dat_timestamp(path: Path) -> datetime | None:
    match = DAT_RE.search(path.name)
    if not match:
        return None
    return datetime.strptime(match.group("stamp"), "%Y%m%d%H%M%S")


def minute_dir_candidates(data_root: Path, minute: datetime) -> list[Path]:
    candidates = [
        data_root / f"{minute:%M}",
        data_root / f"{minute:%H}" / f"{minute:%M}",
    ]
    return candidates


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


def nearest_point(path: Path, target_position_m: float) -> tuple[int, float, float]:
    header = read_header(path)
    raw_index = round((target_position_m - header.position_start_m) / header.point_interval_m)
    point_index = int(np.clip(raw_index, 0, header.point_count - 1))
    distance_m = header.position_start_m + point_index * header.point_interval_m
    return point_index, distance_m, abs(distance_m - target_position_m)


def sanitize(value: str) -> str:
    value = value.replace("%", "pct")
    value = re.sub(r"[^\w.-]+", "_", value, flags=re.UNICODE).strip("_")
    return value or "blank"


def image_name(record: CableRecord, point_index: int, start: datetime, end: datetime) -> str:
    freq = sanitize(record.frequency_hz)
    volume = sanitize(record.volume)
    return f"cable_freq_{freq}Hz_volume_{volume}_point_{point_index}_{start:%H%M%S}_{end:%H%M%S}.png"


def default_export_python(project_root: Path) -> Path:
    candidate = project_root / ".venv-build" / "Scripts" / "python.exe"
    return candidate if candidate.exists() else Path(sys.executable)


def export_image(export_python: Path, export_script: Path, files: list[DatFile], point_index: int, output: Path) -> subprocess.CompletedProcess[str]:
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
    ]
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    return subprocess.run(command, text=True, capture_output=True, env=env)


def compute_snr(files: list[DatFile], point_index: int) -> tuple[dict[str, float] | None, int]:
    matrix, header = load_concatenated_matrix([item.path for item in files])
    trace = matrix[point_index, :].copy()
    n = 1 << (trace.size.bit_length() - 1)
    samples = trace[:n] - np.mean(trace[:n])
    window_arr = np.hanning(n).astype(np.float32)
    spectrum = np.fft.rfft(samples * window_arr)
    freqs = np.fft.rfftfreq(n, d=1.0 / max(1, header.frequency_hz))
    magnitude = np.abs(spectrum) * 2.0 / n
    return calculate_peak_line_snr(freqs, magnitude), header.frequency_hz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export cable test point-analysis images and SNR summary.")
    parser.add_argument("--excel", type=Path, default=DEFAULT_EXCEL)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--date", default=DEFAULT_DATE.isoformat())
    parser.add_argument("--position-m", type=float, default=DEFAULT_POSITION_M)
    parser.add_argument("--window-seconds", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir = Path(__file__).resolve().parent
    project_root = source_dir.parent
    excel_path = args.excel if args.excel.is_absolute() else project_root / args.excel
    output_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    image_dir = output_dir / "images"
    summary_path = output_dir / "cable_point_snr_summary.csv"
    export_python = default_export_python(project_root)
    export_script = source_dir / "point_analysis_export.py"
    test_date = date.fromisoformat(args.date)

    records = iter_records(excel_path)
    if args.limit is not None:
        records = records[: args.limit]
    output_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "serial",
        "frequency_hz",
        "excel_peak_db",
        "excel_noise_db",
        "excel_snr_db",
        "time_range",
        "volume",
        "note",
        "point_index",
        "distance_m",
        "distance_error_m",
        "selected_files",
        "sampling_hz",
        "peak_line_snr_db",
        "peak_freq_hz",
        "peak_value_db",
        "noise_level_db",
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
            base = {
                "serial": record.serial,
                "frequency_hz": record.frequency_hz,
                "excel_peak_db": record.excel_peak_db,
                "excel_noise_db": record.excel_noise_db,
                "excel_snr_db": record.excel_snr_db,
                "time_range": record.time_range,
                "volume": record.volume,
                "note": record.note,
                "point_index": "",
                "distance_m": "",
                "distance_error_m": "",
                "selected_files": "",
                "sampling_hz": "",
                "peak_line_snr_db": "",
                "peak_freq_hz": "",
                "peak_value_db": "",
                "noise_level_db": "",
                "image": "",
                "status": "",
                "status_note": "",
            }
            parsed = parse_time_range(record.time_range, test_date)
            if parsed is None:
                skipped += 1
                writer.writerow(base | {"status": "skipped", "status_note": "invalid time range"})
                continue
            start, end = parsed
            files = choose_middle_files(find_candidate_files(args.data_root, start, end), start, end, args.window_seconds)
            if len(files) < args.window_seconds:
                skipped += 1
                writer.writerow(base | {"status": "skipped", "status_note": f"only {len(files)} files found"})
                continue
            point_index, distance_m, distance_error_m = nearest_point(files[0].path, args.position_m)
            output = image_dir / image_name(record, point_index, start, end)
            row = base | {
                "point_index": point_index,
                "distance_m": f"{distance_m:.3f}",
                "distance_error_m": f"{distance_error_m:.3f}",
                "selected_files": ";".join(str(item.path) for item in files),
                "image": str(output),
            }

            if args.dry_run:
                exported += 1
                writer.writerow(row | {"status": "dry_run"})
                continue

            result = export_image(export_python, export_script, files, point_index, output)
            if result.returncode != 0:
                skipped += 1
                note = (result.stderr or result.stdout).strip().replace("\r", " ").replace("\n", " | ")
                writer.writerow(row | {"status": "failed_export", "status_note": note})
                continue
            snr, sampling_hz = compute_snr(files, point_index)
            if snr is None:
                skipped += 1
                writer.writerow(row | {"status": "failed_snr", "status_note": "no valid peak", "sampling_hz": sampling_hz})
                continue
            exported += 1
            writer.writerow(
                row
                | {
                    "sampling_hz": sampling_hz,
                    "peak_line_snr_db": f"{snr['snr_db']:.2f}",
                    "peak_freq_hz": f"{snr['peak_freq']:.3f}",
                    "peak_value_db": f"{snr['peak_value_db']:.2f}",
                    "noise_level_db": f"{snr['noise_level_db']:.2f}",
                    "status": "exported",
                }
            )
            print(f"exported: {output}")

    print(f"records: {len(records)}, exported/planned: {exported}, skipped/failed: {skipped}")
    print(f"summary: {summary_path}")
    return 0 if exported > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
