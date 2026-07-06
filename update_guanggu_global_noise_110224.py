from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from batch_guanggu_40ns_point_analysis import DatFile, compute_snr, point_distance


PROJECT_ROOT = Path(__file__).resolve().parent
CSV_PATH = PROJECT_ROOT / "outputs" / "point_analysis_guanggu_40ns_global_noise" / "point_snr_summary.csv"
DAT_PATH = Path(r"E:\新方案das数据分析\光谷互联数据\2026-06-15-11-02-24-out.dat")
IMAGE_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "point_analysis_guanggu_40ns_global_noise"
    / "images"
    / "guanggu_freq_30Hz_PZT_200mVpp_point_818_110224.png"
)


def main() -> int:
    dat_file = DatFile(DAT_PATH, datetime(2026, 6, 15, 11, 2, 24))
    snr, sampling_hz, acquisition_time_s = compute_snr(dat_file, 818, 30.0, "global", 5.0)
    if snr is None:
        raise RuntimeError("failed to compute global-noise SNR for 11:02:24")
    distance_m, distance_error_m, point_count = point_distance(dat_file, 818)

    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    updated = 0
    for row in rows:
        if row.get("row_number") == "12" and row.get("frequency_hz") == "30" and row.get("pzt_mvpp") == "200":
            row.update(
                {
                    "selected_file": str(DAT_PATH),
                    "selected_file_time": "2026-06-15 11:02:24",
                    "time_delta_s": "6",
                    "distance_m": f"{distance_m:.3f}",
                    "distance_error_m": f"{distance_error_m:.3f}",
                    "point_count": str(point_count),
                    "sampling_hz": str(sampling_hz),
                    "acquisition_time_s": f"{acquisition_time_s:.2f}",
                    "peak_line_snr_db": f"{snr['snr_db']:.2f}",
                    "peak_freq_hz": f"{snr['peak_freq']:.3f}",
                    "peak_value_db": f"{snr['peak_value_db']:.2f}",
                    "noise_level_db": f"{snr['noise_level_db']:.2f}",
                    "global_noise_level_db": f"{snr['noise_level_db']:.2f}",
                    "global_peak_snr_db": f"{snr['snr_db']:.2f}",
                    "noise_mode": "global",
                    "exclude_band_hz": "5.000",
                    "image": str(IMAGE_PATH),
                    "status": "exported",
                    "status_note": "use specified Guanggu 10s file 2026-06-15 11:02:24 after manual review",
                }
            )
            updated += 1

    if updated != 1:
        raise RuntimeError(f"expected to update 1 row, updated {updated}")

    with CSV_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"updated=1 snr={snr['snr_db']:.2f} peak={snr['peak_freq']:.3f} "
        f"noise={snr['noise_level_db']:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
