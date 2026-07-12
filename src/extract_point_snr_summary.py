#!/usr/bin/env python
"""Recompute the peak-line SNR shown in exported point-analysis images."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from point_analysis_export import load_concatenated_matrix
from batch_point_analysis_from_excel import is_number_like, read_sheet_rows
from tdas_dat_viewer import calculate_peak_line_snr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract peak-line SNR values for point-analysis exports.")
    parser.add_argument("--summary", type=Path, default=Path("outputs/point_analysis_batch/summary.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/point_analysis_batch/snr_summary.csv"))
    parser.add_argument("--excel", type=Path, default=Path("outputs/png_to_excel/DAS测试记录表格_40ns_100ns_无EDFA_AOM频率.xlsx"))
    parser.add_argument("--excel-sheet", default="光谷互联40ns")
    return parser.parse_args()


def compute_snr(files: list[Path], point: int) -> tuple[dict[str, float] | None, int]:
    matrix, header = load_concatenated_matrix(files)
    trace = matrix[point, :].copy()
    n = 1 << (trace.size.bit_length() - 1)
    samples = trace[:n] - np.mean(trace[:n])
    window_arr = np.hanning(n).astype(np.float32)
    spectrum = np.fft.rfft(samples * window_arr)
    freqs = np.fft.rfftfreq(n, d=1.0 / max(1, header.frequency_hz))
    magnitude = np.abs(spectrum) * 2.0 / n
    return calculate_peak_line_snr(freqs, magnitude), header.frequency_hz


def iter_excel_snr_rows(excel_path: Path, sheet_name: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in read_sheet_rows(excel_path, sheet_name):
        row_number = int(row.get(0, 0) or 0)
        frequency = row.get(1)
        pzt = row.get(2)
        peak = row.get(3)
        noise = row.get(4)
        sample_rate = row.get(5)
        if row_number < 6:
            continue
        if not all(is_number_like(value) for value in (frequency, pzt, peak, noise)):
            continue
        snr = float(peak) - float(noise)
        rows.append(
            {
                "sheet": sheet_name,
                "row": str(row_number),
                "frequency_hz": str(frequency),
                "pzt_mvpp": str(pzt),
                "sampling_hz": str(sample_rate) if is_number_like(sample_rate) else "",
                "point": "",
                "image": "",
                "snr_db": f"{snr:.2f}",
                "peak_freq_hz": str(frequency),
                "peak_value_db": f"{float(peak):.2f}",
                "noise_level_db": f"{float(noise):.2f}",
                "status": "ok_excel",
                "note": "SNR = 特征频率峰值 - 底噪水平",
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.summary.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    fieldnames = [
        "sheet",
        "row",
        "frequency_hz",
        "pzt_mvpp",
        "sampling_hz",
        "point",
        "image",
        "snr_db",
        "peak_freq_hz",
        "peak_value_db",
        "noise_level_db",
        "status",
        "note",
    ]
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if not row.get("status", "").startswith("exported"):
                continue
            files = [Path(item) for item in row.get("selected_files", "").split(";") if item]
            point = int(row["point"])
            base = {
                "sheet": row["sheet"],
                "row": row["row"],
                "frequency_hz": row["frequency_hz"],
                "pzt_mvpp": row["pzt_mvpp"],
                "sampling_hz": "",
                "point": row["point"],
                "image": row["output"],
                "snr_db": "",
                "peak_freq_hz": "",
                "peak_value_db": "",
                "noise_level_db": "",
                "status": "ok",
                "note": "",
            }
            try:
                snr, sampling_hz = compute_snr(files, point)
                base["sampling_hz"] = str(sampling_hz)
            except Exception as exc:  # noqa: BLE001 - write row-level failure into CSV.
                writer.writerow(base | {"status": "failed", "note": str(exc)})
                continue
            if snr is None:
                writer.writerow(base | {"status": "failed", "note": "no valid peak"})
                continue
            writer.writerow(
                base
                | {
                    "snr_db": f"{snr['snr_db']:.2f}",
                    "peak_freq_hz": f"{snr['peak_freq']:.3f}",
                    "peak_value_db": f"{snr['peak_value_db']:.2f}",
                    "noise_level_db": f"{snr['noise_level_db']:.2f}",
                }
            )

        for row in iter_excel_snr_rows(args.excel, args.excel_sheet):
            writer.writerow(row)

    print(f"wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
