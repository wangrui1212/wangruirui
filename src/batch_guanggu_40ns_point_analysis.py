#!/usr/bin/env python
"""Batch point-analysis export for Guanggu Interconnection 40ns data."""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from batch_point_analysis_from_excel import is_number_like, read_sheet_rows
from point_analysis_export import load_concatenated_matrix
from tdas_dat_viewer import calculate_peak_line_snr, read_header


DEFAULT_EXCEL = Path("outputs/png_to_excel/DAS测试记录表格_40ns_100ns_无EDFA_AOM频率.xlsx")
DEFAULT_SHEET = "光谷互联40ns"
DEFAULT_DATA_ROOT = Path(r"E:\新方案das数据分析\光谷互联数据")
DEFAULT_OUTPUT_DIR = Path("outputs/point_analysis_guanggu_40ns")
DEFAULT_DISPLAY_POINT = 819
DEFAULT_POSITION_M = 8202.0
DAT_RE = re.compile(r"(?P<stamp>\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})-out\.dat$", re.IGNORECASE)


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
    aom_mhz: str
    fiber_length_km: str
    time_text: str


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
        time_text = text(row.get(9))
        if row_number < 6 or not is_number_like(frequency) or not is_number_like(pzt) or not time_text:
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
                aom_mhz=fmt_number(row.get(7)),
                fiber_length_km=fmt_number(row.get(8)),
                time_text=time_text,
            )
        )
    return records


def parse_record_time_candidates(value: str) -> list[tuple[datetime, str]]:
    value = value.strip()
    candidates: list[tuple[datetime, str]] = []
    for fmt in ("%Y-%m-%d-%H-%M-%S", "%Y-%m-%d-%H-%M"):
        try:
            parsed = datetime.strptime(value, fmt)
            if fmt.endswith("%M"):
                parsed += timedelta(seconds=30)
                if parsed.hour < 10:
                    inferred = parsed.replace(hour=13, minute=parsed.hour, second=parsed.minute)
                    candidates.append((inferred, f"time_inferred from {value} to {inferred:%Y-%m-%d-%H-%M-%S}"))
            candidates.append((parsed, ""))
        except ValueError:
            pass
    if not candidates:
        raise ValueError(f"Unsupported time value: {value}")
    return candidates


def parse_dat_timestamp(path: Path) -> datetime | None:
    match = DAT_RE.search(path.name)
    if not match:
        return None
    return datetime.strptime(match.group("stamp"), "%Y-%m-%d-%H-%M-%S")


def find_dat_files(data_root: Path) -> list[DatFile]:
    files: list[DatFile] = []
    for path in data_root.rglob("*.dat"):
        timestamp = parse_dat_timestamp(path)
        if timestamp is not None:
            files.append(DatFile(path, timestamp))
    return sorted(files, key=lambda item: item.timestamp)


def choose_nearest_file(files: list[DatFile], target: datetime) -> DatFile:
    return min(files, key=lambda item: abs(item.timestamp - target))


def choose_best_time_and_file(files: list[DatFile], value: str) -> tuple[datetime, str, DatFile, float]:
    best: tuple[datetime, str, DatFile, float] | None = None
    for target, note in parse_record_time_candidates(value):
        selected = choose_nearest_file(files, target)
        delta = abs((selected.timestamp - target).total_seconds())
        if best is None or delta < best[3]:
            best = (target, note, selected, delta)
    assert best is not None
    return best


def excel_snr(record: TestRecord) -> str:
    try:
        return f"{float(record.excel_peak_db) - float(record.excel_noise_db):.2f}"
    except ValueError:
        return ""


def sanitize(value: str) -> str:
    return re.sub(r"[^\w.-]+", "_", value, flags=re.UNICODE).strip("_") or "blank"


def image_name(record: TestRecord, point_index: int, selected: DatFile) -> str:
    return (
        f"guanggu_freq_{sanitize(record.frequency_hz)}Hz_PZT_{sanitize(record.pzt_mvpp)}mVpp_"
        f"point_{point_index}_{selected.timestamp:%H%M%S}.png"
    )


def default_export_python(project_root: Path) -> Path:
    candidate = project_root / ".venv-build" / "Scripts" / "python.exe"
    return candidate if candidate.exists() else Path(sys.executable)


def export_image(
    export_python: Path,
    export_script: Path,
    dat_file: DatFile,
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
        str(dat_file.path),
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
    dat_file: DatFile,
    point_index: int,
    target_frequency_hz: float,
    snr_noise_mode: str,
    exclude_band_hz: float,
) -> tuple[dict[str, float] | None, int, float]:
    matrix, header = load_concatenated_matrix([dat_file.path])
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
    ), header.frequency_hz, header.acquisition_time_s


def point_distance(dat_file: DatFile, point_index: int) -> tuple[float, float, int]:
    header = read_header(dat_file.path)
    distance_m = header.position_start_m + point_index * header.point_interval_m
    return distance_m, abs(distance_m - DEFAULT_POSITION_M), header.point_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Guanggu 40ns point-analysis images and SNR summary.")
    parser.add_argument("--excel", type=Path, default=DEFAULT_EXCEL)
    parser.add_argument("--sheet", default=DEFAULT_SHEET)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--display-point", type=int, default=DEFAULT_DISPLAY_POINT)
    parser.add_argument("--position-m", type=float, default=DEFAULT_POSITION_M)
    parser.add_argument("--max-time-delta-s", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--snr-noise-mode", choices=["local", "global"], default="local")
    parser.add_argument("--exclude-band-hz", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir = Path(__file__).resolve().parent
    project_root = source_dir.parent
    excel_path = args.excel if args.excel.is_absolute() else project_root / args.excel
    output_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    image_dir = output_dir / "images"
    summary_path = output_dir / "point_snr_summary.csv"
    export_python = default_export_python(project_root)
    export_script = source_dir / "point_analysis_export.py"
    # point_analysis_export.py uses zero-based display point indices.
    point_index = args.display_point - 1

    records = iter_records(excel_path, args.sheet)
    if args.limit is not None:
        records = records[: args.limit]
    dat_files = find_dat_files(args.data_root)
    if not dat_files:
        raise RuntimeError(f"No Guanggu .dat files found under {args.data_root}")
    output_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "serial",
        "row_number",
        "device_type",
        "frequency_hz",
        "pzt_mvpp",
        "excel_peak_db",
        "excel_noise_db",
        "excel_snr_db",
        "sampling_hz_record",
        "pulse_width_ns",
        "aom_mhz",
        "fiber_length_km",
        "time_text",
        "target_time",
        "selected_file",
        "selected_file_time",
        "time_delta_s",
        "requested_display_point",
        "point_index",
        "target_position_m",
        "distance_m",
        "distance_error_m",
        "point_count",
        "sampling_hz",
        "acquisition_time_s",
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
            base = {
                "serial": record.serial,
                "row_number": record.row_number,
                "device_type": "guanggu_interconnection",
                "frequency_hz": record.frequency_hz,
                "pzt_mvpp": record.pzt_mvpp,
                "excel_peak_db": record.excel_peak_db,
                "excel_noise_db": record.excel_noise_db,
                "excel_snr_db": excel_snr(record),
                "sampling_hz_record": record.sampling_hz_record,
                "pulse_width_ns": record.pulse_width_ns,
                "aom_mhz": record.aom_mhz,
                "fiber_length_km": record.fiber_length_km,
                "time_text": record.time_text,
                "target_time": "",
                "selected_file": "",
                "selected_file_time": "",
                "time_delta_s": "",
                "requested_display_point": args.display_point,
                "point_index": point_index,
                "target_position_m": f"{args.position_m:.3f}",
                "distance_m": "",
                "distance_error_m": "",
                "point_count": "",
                "sampling_hz": "",
                "acquisition_time_s": "",
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
            try:
                target_time, inference_note, selected, time_delta_s = choose_best_time_and_file(dat_files, record.time_text)
            except ValueError as exc:
                skipped += 1
                writer.writerow(base | {"status": "skipped", "status_note": str(exc)})
                continue
            if time_delta_s > args.max_time_delta_s:
                skipped += 1
                writer.writerow(
                    base
                    | {
                        "target_time": f"{target_time:%Y-%m-%d %H:%M:%S}",
                        "selected_file": str(selected.path),
                        "selected_file_time": f"{selected.timestamp:%Y-%m-%d %H:%M:%S}",
                        "time_delta_s": f"{time_delta_s:.0f}",
                        "status": "skipped",
                        "status_note": f"nearest file is {time_delta_s:.0f}s away, over threshold {args.max_time_delta_s:.0f}s",
                    }
                )
                continue
            distance_m, distance_error_m, point_count = point_distance(selected, point_index)
            output = image_dir / image_name(record, point_index, selected)
            row = base | {
                "target_time": f"{target_time:%Y-%m-%d %H:%M:%S}",
                "selected_file": str(selected.path),
                "selected_file_time": f"{selected.timestamp:%Y-%m-%d %H:%M:%S}",
                "time_delta_s": f"{time_delta_s:.0f}",
                "distance_m": f"{distance_m:.3f}",
                "distance_error_m": f"{distance_error_m:.3f}",
                "point_count": point_count,
                "image": str(output),
                "status_note": inference_note or "single 10s Guanggu file nearest to record time",
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
            snr, sampling_hz, acquisition_time_s = compute_snr(
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
                    "sampling_hz": sampling_hz,
                    "acquisition_time_s": f"{acquisition_time_s:.2f}",
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

    print(f"device_type: guanggu_interconnection")
    print(f"records: {len(records)}, exported/planned: {exported}, skipped/failed: {skipped}")
    print(f"summary: {summary_path}")
    return 0 if exported > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
