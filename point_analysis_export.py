#!/usr/bin/env python
"""Export TDAS/CDAS point-analysis window images from one or more .dat files.

This script reuses the reading and point-analysis logic from tdas_dat_viewer.py.
Multiple files are concatenated along the time/sample axis before extracting the
requested point trace.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets
from PyQt6 import QtGui

from tdas_dat_viewer import DatHeader, PointAnalysisWindow, read_header, read_signal_matrix


def expand_inputs(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.dat")))
        else:
            paths.append(path)
    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            unique_paths.append(path)
            seen.add(resolved)
    return sorted(unique_paths)


def validate_headers(paths: list[Path], headers: dict[Path, DatHeader]) -> DatHeader:
    if not paths:
        raise ValueError("未提供 .dat 文件")
    base = headers[paths[0]]
    total_samples = 0
    for path in paths:
        header = headers[path]
        if header.protocol != base.protocol:
            raise ValueError(f"协议类型不同，无法拼接: {path.name}={header.protocol}, {paths[0].name}={base.protocol}")
        if header.point_count != base.point_count:
            raise ValueError(f"点位数不同，无法拼接: {path.name}={header.point_count}, {paths[0].name}={base.point_count}")
        if abs(header.position_start_m - base.position_start_m) > 1e-6:
            raise ValueError(
                f"起始位置不同，无法拼接: {path.name}={header.position_start_m:.9g} m, "
                f"{paths[0].name}={base.position_start_m:.9g} m"
            )
        if abs(header.position_end_m - base.position_end_m) > 1e-6:
            raise ValueError(
                f"结束位置不同，无法拼接: {path.name}={header.position_end_m:.9g} m, "
                f"{paths[0].name}={base.position_end_m:.9g} m"
            )
        if abs(header.point_interval_m - base.point_interval_m) > 1e-6:
            raise ValueError(
                f"点位间隔不同，无法拼接: {path.name}={header.point_interval_m:.9g} m, "
                f"{paths[0].name}={base.point_interval_m:.9g} m"
            )
        if header.frequency_hz != base.frequency_hz:
            raise ValueError(
                f"采样率不同，无法拼接: {path.name}={header.frequency_hz} Hz, "
                f"{paths[0].name}={base.frequency_hz} Hz"
            )
        if header.protocol == "CDAS" and header.channel != base.channel:
            raise ValueError(f"通道号不同，无法拼接: {path.name}=通道{header.channel}, {paths[0].name}=通道{base.channel}")
        total_samples += header.sample_count
    return replace(base, sample_count=total_samples, acquisition_time_s=total_samples / max(1, base.frequency_hz))


def load_concatenated_matrix(paths: list[Path]) -> tuple[np.ndarray, DatHeader]:
    headers = {path: read_header(path) for path in paths}
    combined_header = validate_headers(paths, headers)
    matrices = [read_signal_matrix(path, headers[path]) for path in paths]
    matrix = matrices[0] if len(matrices) == 1 else np.concatenate(matrices, axis=1)
    return matrix, combined_header


def configure_chinese_font(app: QtWidgets.QApplication) -> str | None:
    """Load a CJK-capable font explicitly for offscreen Qt rendering."""
    candidates = [
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\Deng.ttf"),
    ]
    for font_path in candidates:
        if not font_path.exists():
            continue
        font_id = QtGui.QFontDatabase.addApplicationFont(str(font_path))
        if font_id < 0:
            continue
        families = QtGui.QFontDatabase.applicationFontFamilies(font_id)
        if not families:
            continue
        font = QtGui.QFont(families[0], 10)
        app.setFont(font)
        return families[0]
    return None


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="拼接一个或多个 TDAS/CDAS .dat 文件，并导出指定点位的点位分析窗口图片。"
    )
    parser.add_argument("files", nargs="+", help="一个或多个 .dat 文件；也可传入目录，目录下 *.dat 会按文件名排序拼接。")
    parser.add_argument("--point", type=int, required=True, help="点位编号。默认按 viewer 显示点位/0-based 索引理解，例如 --point 799 表示显示点位799，也就是第800个点。")
    parser.add_argument("--one-based", action="store_true", help="将 --point 解释为自然序号/1-based，例如 --point 800 等价于默认模式下 --point 799。")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="输出图片路径，支持 .png/.jpg。默认输出到当前目录 point_analysis_point<点位>.png。",
    )
    parser.add_argument("--width", type=int, default=1200, help="导出窗口宽度，默认 1200。")
    parser.add_argument("--height", type=int, default=850, help="导出窗口高度，默认 850。")
    parser.add_argument("--target-frequency", type=float, default=None, help="使用目标频率模式计算/显示峰值谱线SNR。")
    parser.add_argument("--search-half-width", type=float, default=5.0, help="目标频率搜索半宽，默认 5 Hz。")
    parser.add_argument("--noise-window", type=float, default=20.0, help="SNR局部噪声窗口，默认 20 Hz。")
    parser.add_argument("--guard-band", type=float, default=1.0, help="SNR保护带宽，默认 1 Hz。")
    parser.add_argument("--snr-noise-mode", choices=["local", "global"], default="local", help="SNR底噪口径，默认 local。")
    parser.add_argument("--exclude-band-hz", type=float, default=5.0, help="全局底噪排除信号频率总宽，默认 5 Hz。")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    paths = expand_inputs(args.files)
    if not paths:
        raise SystemExit("未找到可读取的 .dat 文件")
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise SystemExit("以下文件不存在:\n" + "\n".join(missing))

    point_index = args.point - 1 if args.one_based else args.point
    if point_index < 0:
        raise SystemExit("点位编号必须为非负索引，或使用 --one-based 指定大于0的自然序号")

    matrix, header = load_concatenated_matrix(paths)
    if point_index >= header.point_count:
        raise SystemExit(f"点位超出范围: 输入索引={point_index}, 点位数={header.point_count}")

    distance_m = header.position_start_m + point_index * header.point_interval_m
    trace = matrix[point_index, :].copy()

    output = args.output or Path(f"point_analysis_point{args.point}.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "0")
    pg.setConfigOption("background", "w")
    pg.setConfigOption("foreground", "k")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    configure_chinese_font(app)
    window = PointAnalysisWindow(trace, header, point_index, distance_m)
    window.snr_noise_mode = args.snr_noise_mode
    window.snr_exclude_band_hz = float(args.exclude_band_hz)
    if args.target_frequency is not None:
        window.snr_mode_combo.setCurrentIndex(1)
        window.snr_target_freq.setValue(float(args.target_frequency))
        window.snr_search_half_width.setValue(float(args.search_half_width))
        window.snr_noise_window.setValue(float(args.noise_window))
        window.snr_guard_band.setValue(float(args.guard_band))
        window.update_fft_plot()
    point_label = str(point_index)
    window.setWindowTitle(f"点位分析 - #{point_label} / {distance_m:.3f} m")
    title_label = window.findChild(QtWidgets.QLabel)
    if title_label is not None:
        title_label.setText(
            f"点位: {point_label}    位置: {distance_m:.3f} m    "
            f"采样率: {header.frequency_hz} Hz    采样数: {header.sample_count}"
        )
    window.resize(args.width, args.height)
    window.show()
    app.processEvents()
    QtCore.QThread.msleep(300)
    app.processEvents()
    pixmap = window.grab()
    if not pixmap.save(str(output)):
        raise SystemExit(f"图片保存失败: {output}")
    window.close()

    print(f"已拼接文件数: {len(paths)}")
    print(f"数据矩阵: {matrix.shape[0]} 点位 x {matrix.shape[1]} 采样")
    print(f"导出点位: 输入={args.point}, 内部索引={point_index}, 位置={distance_m:.3f} m")
    print(f"输出图片: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
