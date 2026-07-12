#!/usr/bin/env python
"""Create compact, presentation-ready copies of the PSD analysis figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def safe_name(text: str) -> str:
    return "_".join("".join(char if char.isalnum() else " " for char in text).split())


def save_optimized(source: Path, target: Path, *, crop_plot: bool) -> None:
    with Image.open(source) as image:
        image = image.convert("RGB")
        if crop_plot:
            width, height = image.size
            image = image.crop(
                (
                    int(width * 0.035),
                    int(height * 0.115),
                    int(width * 0.972),
                    int(height * 0.885),
                )
            )
        max_width = 1600
        if image.width > max_width:
            new_height = int(round(image.height * max_width / image.width))
            image = image.resize((max_width, new_height), Image.Resampling.LANCZOS)
        target.parent.mkdir(parents=True, exist_ok=True)
        image.save(target, "JPEG", quality=88, optimize=True, progressive=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()

    rows = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    output_rows = []
    for row in rows:
        stem = safe_name(f"{row['experiment_type']}_{row['experiment_id']}_{row['device']}")
        phase_target = args.asset_dir / f"{stem}_phase.jpg"
        save_optimized(Path(row["phase_image"]), phase_target, crop_plot=False)
        band_targets = {}
        for band, source in row["band_images"].items():
            target = args.asset_dir / f"{stem}_{safe_name(band)}.jpg"
            save_optimized(Path(source), target, crop_plot=True)
            band_targets[band] = str(target)
        output = dict(row)
        output["phase_image"] = str(phase_target)
        output["band_images"] = band_targets
        output_rows.append(output)

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(output_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"assets={len(output_rows) * 5} manifest={args.output_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
