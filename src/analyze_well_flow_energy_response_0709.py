#!/usr/bin/env python
"""Summarize spatial-band energy response for the 2026-07-09 well-flow deck."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from batch_well_flow_three_device_analysis import (
    Band,
    PointRange,
    compute_band_energy,
    files_for_record,
    index_data_files,
    read_records,
    select_point_indices,
    validate_headers,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = PROJECT_ROOT / "井中流动分析" / "井中流动分析_两组实验记录.xlsx"
DATA_ROOT = Path(r"E:\北京试验")
OUTPUT_DIR = PROJECT_ROOT / "井中流动分析" / "空间点位能量变化分析_自研补齐版"

DEVICES = {
    "silixia": "silixia",
    "OV": "ov",
    "自研_2U": "自研_2U",
}

SEGMENTS = {
    "silixia": [
        ("筒内光纤", 1048.0, 1060.0),
        ("筒外缆线", 1104.0, 1116.0),
    ],
    "OV": [
        ("筒内光纤", 320.0, 330.0),
        ("筒外缆线", 376.0, 392.0),
    ],
    "自研_2U": [
        ("筒外光缆", 348.0, 358.0),
    ],
}

BANDS = [
    Band("1~10Hz", 1.0, 10.0),
    Band("10~50Hz", 10.0, 50.0),
    Band("50~200Hz", 50.0, 200.0),
]


@dataclass
class SegmentRecord:
    experiment_group: str
    experiment_id: str
    device: str
    segment_name: str
    requested_range_m: str
    actual_range_m: str
    flow_lh: float
    band_label: str
    mean_energy_db: float
    median_energy_db: float
    p90_energy_db: float
    point_count: int
    time_window_count: int
    file_count: int


@dataclass
class SegmentSummary:
    device: str
    segment_name: str
    requested_range_m: str
    actual_range_m: str
    band_label: str
    first_group_dynamic_db: float
    first_group_corr_flow: float
    second_group_dynamic_db: float
    second_group_corr_flow: float
    overall_dynamic_db: float
    comment: str


def corrcoef(x: list[float], y: list[float]) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if float(np.std(x_arr)) <= 1e-12 or float(np.std(y_arr)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def summarize_group(rows: list[SegmentRecord]) -> tuple[float, float]:
    if not rows:
        return float("nan"), float("nan")
    values = [row.median_energy_db for row in rows]
    flows = [row.flow_lh for row in rows]
    return float(max(values) - min(values)), corrcoef(flows, values)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = read_records(WORKBOOK)
    files_by_device = {
        device: index_data_files(DATA_ROOT / subdir)
        for device, subdir in DEVICES.items()
    }

    detail_rows: list[SegmentRecord] = []
    for record in records:
        for device, ranges in SEGMENTS.items():
            files = files_for_record(files_by_device[device], record)
            if not files:
                continue
            paths = [item.path for item in files]
            header, headers = validate_headers(paths)
            for segment_name, low_m, high_m in ranges:
                requested = PointRange(f"{low_m:g}~{high_m:g}m", low_m, high_m)
                indices, positions, _warning = select_point_indices(header, requested)
                energy, centers = compute_band_energy(
                    paths,
                    header,
                    headers,
                    indices,
                    BANDS,
                    window_s=2.0,
                    overlap_percent=50.0,
                )
                actual_range = f"{float(positions[0]):.3f}-{float(positions[-1]):.3f}"
                for band_index, band in enumerate(BANDS):
                    values = energy[band_index]
                    finite = values[np.isfinite(values)]
                    if not finite.size:
                        continue
                    detail_rows.append(
                        SegmentRecord(
                            experiment_group=record.group,
                            experiment_id=record.seq,
                            device=device,
                            segment_name=segment_name,
                            requested_range_m=requested.text,
                            actual_range_m=actual_range,
                            flow_lh=float(record.flow_lh),
                            band_label=band.label,
                            mean_energy_db=float(np.mean(finite)),
                            median_energy_db=float(np.median(finite)),
                            p90_energy_db=float(np.percentile(finite, 90.0)),
                            point_count=int(indices.size),
                            time_window_count=int(centers.size),
                            file_count=len(files),
                        )
                    )

    detail_path = OUTPUT_DIR / "空间段频带能量_逐实验统计.csv"
    with detail_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SegmentRecord.__annotations__.keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in detail_rows)

    summaries: list[SegmentSummary] = []
    keys = sorted({(row.device, row.segment_name, row.requested_range_m, row.actual_range_m, row.band_label) for row in detail_rows})
    for device, segment_name, requested_range, actual_range, band_label in keys:
        rows = [
            row for row in detail_rows
            if row.device == device
            and row.segment_name == segment_name
            and row.requested_range_m == requested_range
            and row.actual_range_m == actual_range
            and row.band_label == band_label
        ]
        first = [row for row in rows if row.experiment_group == "第一组实验"]
        second = [row for row in rows if row.experiment_group == "第二组实验"]
        first_dyn, first_corr = summarize_group(first)
        second_dyn, second_corr = summarize_group(second)
        all_values = [row.median_energy_db for row in rows]
        overall_dyn = float(max(all_values) - min(all_values)) if all_values else float("nan")
        stronger = []
        if np.isfinite(first_dyn) and first_dyn >= 5.0:
            stronger.append("第一组随流量变化明显")
        if np.isfinite(second_dyn) and second_dyn >= 5.0:
            stronger.append("第二组随流量变化明显")
        if not stronger:
            stronger.append("变化较弱或需结合图面判断")
        summaries.append(
            SegmentSummary(
                device=device,
                segment_name=segment_name,
                requested_range_m=requested_range,
                actual_range_m=actual_range,
                band_label=band_label,
                first_group_dynamic_db=first_dyn,
                first_group_corr_flow=first_corr,
                second_group_dynamic_db=second_dyn,
                second_group_corr_flow=second_corr,
                overall_dynamic_db=overall_dyn,
                comment="；".join(stronger),
            )
        )

    summary_path = OUTPUT_DIR / "空间段频带能量_响应汇总.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SegmentSummary.__annotations__.keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in summaries)

    (OUTPUT_DIR / "空间段频带能量_响应汇总.json").write_text(
        json.dumps([asdict(row) for row in summaries], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"DETAIL {detail_path}")
    print(f"SUMMARY {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
