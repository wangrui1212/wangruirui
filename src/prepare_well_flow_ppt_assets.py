#!/usr/bin/env python
"""Prepare compressed image assets and a manifest for the well-flow PPT."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS_DIR = PROJECT_ROOT / "井中流动分析" / "三设备分析结果"
DEFAULT_ASSET_DIR = PROJECT_ROOT / "井中流动分析" / "ppt_assets"


BANDS = [
    ("0Hz", "0Hz"),
    ("1~10Hz", "1_10Hz"),
    ("10~25Hz", "10_25Hz"),
    ("25~40Hz", "25_40Hz"),
    ("40~55Hz", "40_55Hz"),
    ("55~70Hz", "55_70Hz"),
    ("70~85Hz", "70_85Hz"),
    ("85~100Hz", "85_100Hz"),
    ("100~115Hz", "100_115Hz"),
    ("115~130Hz", "115_130Hz"),
    ("130~145Hz", "130_145Hz"),
    ("145~160Hz", "145_160Hz"),
    ("160~175Hz", "160_175Hz"),
    ("175~190Hz", "175_190Hz"),
    ("190~200Hz", "190_200Hz"),
]


@dataclass
class ImageItem:
    label: str
    source: str
    asset: str


@dataclass
class DeviceItem:
    group: str
    seq: str
    device: str
    start: str
    end: str
    operation: str
    flow_lh: str
    requested_range_m: str
    actual_range_m: str
    selected_point_count: str
    file_count: str
    status: str
    message: str
    warnings: str
    output_folder: str
    phase: ImageItem | None = None
    bands: list[ImageItem] = field(default_factory=list)


def safe_filename(text: str) -> str:
    return re.sub(r'[\\/:*?"<>|\s]+', "_", str(text)).strip("_")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def compress_image(source: Path, target: Path, max_width: int, quality: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        rgb = image.convert("RGB")
        if rgb.width > max_width:
            new_height = int(round(rgb.height * max_width / rgb.width))
            rgb = rgb.resize((max_width, new_height), Image.Resampling.LANCZOS)
        rgb.save(target, format="JPEG", quality=quality, optimize=True, progressive=True)


def maybe_asset(source: Path, asset_dir: Path, relative_stem: Path, max_width: int, quality: int) -> str:
    target = asset_dir / relative_stem.with_suffix(".jpg")
    if not target.exists() or target.stat().st_mtime < source.stat().st_mtime:
        compress_image(source, target, max_width=max_width, quality=quality)
    return str(target)


def band_from_folder(folder: Path, label: str, filename_label: str) -> Path | None:
    candidates = sorted(folder.glob(f"*_{filename_label}_energy.png"))
    return candidates[0] if candidates else None


def build_manifest(analysis_dir: Path, asset_dir: Path) -> dict[str, object]:
    summary_csv = analysis_dir / "well_flow_three_device_summary.csv"
    rows = read_csv(summary_csv)
    devices: list[DeviceItem] = []

    for row in rows:
        output_folder = Path(row["output_folder"])
        device_key = safe_filename(f"{row['group']}_{row['seq']}_{row['device']}")
        item = DeviceItem(
            group=row["group"],
            seq=row["seq"],
            device=row["device"],
            start=row["start"],
            end=row["end"],
            operation=row["operation"],
            flow_lh=row["flow_lh"],
            requested_range_m=row["requested_range_m"],
            actual_range_m=row["actual_range_m"],
            selected_point_count=row["selected_point_count"],
            file_count=row["file_count"],
            status=row["status"],
            message=row["message"],
            warnings=row["warnings"],
            output_folder=str(output_folder),
        )

        phase_candidates = sorted((output_folder / "phase").glob("*_phase_heatmap.png"))
        if phase_candidates:
            phase_source = phase_candidates[0]
            phase_asset = maybe_asset(
                phase_source,
                asset_dir,
                Path("phase") / device_key / phase_source.stem,
                max_width=1500,
                quality=88,
            )
            item.phase = ImageItem(label="相位热力图", source=str(phase_source), asset=phase_asset)

        band_folder = output_folder / "band_energy"
        for label, filename_label in BANDS:
            band_source = band_from_folder(band_folder, label, filename_label)
            if band_source is None:
                continue
            band_asset = maybe_asset(
                band_source,
                asset_dir,
                Path("band_energy") / device_key / band_source.stem,
                max_width=1200,
                quality=84,
            )
            item.bands.append(ImageItem(label=label, source=str(band_source), asset=band_asset))
        devices.append(item)

    manifest = {
        "analysis_dir": str(analysis_dir),
        "asset_dir": str(asset_dir),
        "band_order": [label for label, _filename_label in BANDS],
        "devices": [asdict(item) for item in devices],
    }
    manifest_path = asset_dir / "well_flow_ppt_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare compressed well-flow PPT assets.")
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--asset-dir", type=Path, default=DEFAULT_ASSET_DIR)
    args = parser.parse_args()
    manifest = build_manifest(args.analysis_dir, args.asset_dir)
    devices = manifest["devices"]
    phase_count = sum(1 for item in devices if item["phase"])
    band_count = sum(len(item["bands"]) for item in devices)
    print(f"MANIFEST {args.asset_dir / 'well_flow_ppt_manifest.json'}")
    print(f"DEVICES {len(devices)}")
    print(f"PHASE_IMAGES {phase_count}")
    print(f"BAND_IMAGES {band_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
