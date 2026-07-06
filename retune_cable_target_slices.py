#!/usr/bin/env python
"""Retune selected cable-test slices around target frequencies and update outputs."""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from batch_cable_point_analysis import DatFile, find_candidate_files, parse_dat_timestamp
from point_analysis_export import load_concatenated_matrix
from tdas_dat_viewer import calculate_peak_line_snr


PROJECT_ROOT = Path(__file__).resolve().parent
SUMMARY = PROJECT_ROOT / "outputs/cable_test/point_analysis/cable_point_snr_summary.csv"
IMAGE_DIR = PROJECT_ROOT / "outputs/cable_test/point_analysis/images"
DATA_ROOT = Path(r"E:\新方案das数据分析\DAS0624data-缆\20260624\15")
EXPORT_SCRIPT = PROJECT_ROOT / "point_analysis_export.py"
PYTHON = PROJECT_ROOT / ".venv-build/Scripts/python.exe"
TARGETS = {
    "5": {"frequency": 100.0},
    "6": {"frequency": 200.0, "mode": "target", "search_half_width": 30.0},
    "7": {"frequency": 500.0, "fixed_start": datetime(2026, 6, 24, 15, 14, 44)},
}


def parse_time_range(value: str) -> tuple[datetime, datetime]:
    left, right = [part.strip() for part in value.split("-", 1)]
    test_date = "2026-06-24"
    return (
        datetime.fromisoformat(f"{test_date}T{left}"),
        datetime.fromisoformat(f"{test_date}T{right}"),
    )


def snr_for_files(
    files: list[DatFile],
    point_index: int,
    target_freq: float,
    mode: str = "auto",
    search_half_width: float = 5.0,
) -> dict[str, float] | None:
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
        mode=mode,
        target_freq_hz=target_freq,
        search_half_width_hz=search_half_width,
    )


def windows(files: list[DatFile], size: int = 10) -> list[list[DatFile]]:
    ordered = sorted(files, key=lambda item: item.timestamp)
    result: list[list[DatFile]] = []
    for index in range(0, max(0, len(ordered) - size + 1)):
        candidate = ordered[index : index + size]
        if (candidate[-1].timestamp - candidate[0].timestamp).total_seconds() <= size + 1:
            result.append(candidate)
    return result


def fixed_files(start: datetime, size: int = 10) -> list[DatFile]:
    end = start + timedelta(seconds=size - 1)
    files = find_candidate_files(DATA_ROOT, start, end)
    return files[:size]


def image_name(row: dict[str, str], target_freq: float, files: list[DatFile]) -> str:
    volume = row["volume"].replace("%", "pct")
    return (
        f"cable_freq_{int(target_freq)}Hz_volume_{volume}_point_{row['point_index']}_"
        f"auto_near_{int(target_freq)}Hz_{files[0].timestamp:%H%M%S}_{files[-1].timestamp:%H%M%S}.png"
    )


def export_image(
    row: dict[str, str],
    target_freq: float,
    files: list[DatFile],
    output: Path,
    mode: str = "auto",
    search_half_width: float = 5.0,
) -> None:
    command = [
        str(PYTHON),
        str(EXPORT_SCRIPT),
        *[str(item.path) for item in files],
        "--point",
        row["point_index"],
        "--output",
        str(output),
    ]
    if mode == "target":
        command.extend(
            [
                "--target-frequency",
                f"{target_freq:g}",
                "--search-half-width",
                f"{search_half_width:g}",
            ]
        )
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    result = subprocess.run(command, text=True, capture_output=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)


def main() -> int:
    with SUMMARY.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0].keys())

    for row in rows:
        target = TARGETS.get(row["serial"])
        if target is None:
            continue
        target_freq = float(target["frequency"])
        mode = str(target.get("mode", "auto"))
        search_half_width = float(target.get("search_half_width", 5.0))
        point_index = int(row["point_index"])
        start, end = parse_time_range(row["time_range"])
        if "fixed_start" in target:
            selected = fixed_files(target["fixed_start"])
        else:
            candidates = find_candidate_files(DATA_ROOT, start, end)
            scored = []
            for candidate in windows(candidates, 10):
                snr = snr_for_files(candidate, point_index, target_freq, mode, search_half_width)
                if snr is None:
                    continue
                freq_error = abs(snr["peak_freq"] - target_freq)
                if mode == "target" or freq_error <= 5.0:
                    scored.append((snr["snr_db"], -freq_error, candidate, snr))
            if not scored:
                for candidate in windows(candidates, 10):
                    snr = snr_for_files(candidate, point_index, target_freq, mode, search_half_width)
                    if snr is None:
                        continue
                    freq_error = abs(snr["peak_freq"] - target_freq)
                    scored.append((-freq_error, snr["snr_db"], candidate, snr))
                if not scored:
                    raise RuntimeError(f"No valid auto slice for serial {row['serial']}")
                scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
                _, _, selected, _ = scored[0]
            else:
                scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
                _, _, selected, _ = scored[0]

        if len(selected) < 10:
            raise RuntimeError(f"Only {len(selected)} files selected for serial {row['serial']}")
        snr = snr_for_files(selected, point_index, target_freq, mode, search_half_width)
        if snr is None:
            raise RuntimeError(f"No SNR for serial {row['serial']}")
        output = IMAGE_DIR / image_name(row, target_freq, selected)
        export_image(row, target_freq, selected, output, mode, search_half_width)
        row["selected_files"] = ";".join(str(item.path) for item in selected)
        row["peak_line_snr_db"] = f"{snr['snr_db']:.2f}"
        row["peak_freq_hz"] = f"{snr['peak_freq']:.3f}"
        row["peak_value_db"] = f"{snr['peak_value_db']:.2f}"
        row["noise_level_db"] = f"{snr['noise_level_db']:.2f}"
        row["image"] = str(output)
        freq_error = abs(float(snr["peak_freq"]) - target_freq)
        if mode == "target":
            row["status_note"] = (
                f"target slice {target_freq:g}+/-{search_half_width:g} Hz, "
                f"{selected[0].timestamp:%H:%M:%S}-{selected[-1].timestamp:%H:%M:%S}"
            )
        elif freq_error <= 5.0:
            row["status_note"] = f"auto slice near {target_freq:g} Hz, {selected[0].timestamp:%H:%M:%S}-{selected[-1].timestamp:%H:%M:%S}"
        else:
            row["status_note"] = (
                f"auto slice closest to {target_freq:g} Hz, but auto peak at {snr['peak_freq']:.3f} Hz; "
                f"{selected[0].timestamp:%H:%M:%S}-{selected[-1].timestamp:%H:%M:%S}"
            )
        print(row["serial"], row["frequency_hz"], row["status_note"], row["peak_line_snr_db"], row["peak_freq_hz"], output)

    with SUMMARY.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
