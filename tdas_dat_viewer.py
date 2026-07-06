#!/usr/bin/env python
"""PyQtGraph viewer for Changfei CDAS/TDAS .dat files."""

from __future__ import annotations

import struct
import sys
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
import re

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets


HEADER_SIZE = 36
HEADER_STRUCT = struct.Struct(">4sHHHfI H B B H f f H H")
GUANGGU_HEADER_FLOATS = 64
GUANGGU_HEADER_SIZE = GUANGGU_HEADER_FLOATS * 4
DISPLAY_MAX_TIME_BINS = 4096


@dataclass(frozen=True)
class DatHeader:
    protocol: str
    system_id: str
    year: int
    month: int
    day: int
    hour: int
    minute: int
    second: float
    length: int
    global_alarm: int
    acquisition_status: int
    channel: int
    frequency_hz: int
    acquisition_time_s: float
    point_interval_m: float
    point_count: int
    sample_count: int
    position_start_m: float = 0.0
    raw_value_note: str = "raw_int16 / 100"
    header_size: int = HEADER_SIZE
    bytes_per_sample: int = 2
    timestamp_text: str | None = None
    extra_info: dict[str, object] = field(default_factory=dict)

    @property
    def timestamp(self) -> str:
        if self.timestamp_text:
            return self.timestamp_text
        sec = int(self.second)
        ms = int(round((self.second - sec) * 1000))
        try:
            base = datetime(self.year, self.month, self.day, self.hour, self.minute, sec)
            return base.strftime("%Y-%m-%d %H:%M:%S") + f".{ms:03d}"
        except ValueError:
            return (
                f"{self.year:04d}-{self.month:02d}-{self.day:02d} "
                f"{self.hour:02d}:{self.minute:02d}:{self.second:06.3f}"
            )

    @property
    def expected_data_bytes(self) -> int:
        return self.point_count * self.sample_count * self.bytes_per_sample

    @property
    def distance_span_m(self) -> float:
        if self.point_count <= 1:
            return 0.0
        return (self.point_count - 1) * self.point_interval_m

    @property
    def position_end_m(self) -> float:
        return self.position_start_m + self.distance_span_m

    @property
    def time_span_s(self) -> float:
        if self.acquisition_time_s > 0:
            return self.acquisition_time_s
        return self.sample_count / max(1, self.frequency_hz)


def read_cdas_header(path: Path) -> DatHeader:
    with path.open("rb") as f:
        raw = f.read(HEADER_SIZE)
    if len(raw) < HEADER_SIZE:
        raise ValueError(f"{path} 文件太小，无法读取 CDAS 帧头")

    (
        system_id,
        year,
        month_day,
        hour_minute,
        second,
        length,
        global_alarm,
        acquisition_status,
        channel,
        frequency_hz,
        acquisition_time_s,
        point_interval_m,
        point_count,
        sample_count,
    ) = HEADER_STRUCT.unpack(raw)

    system_text = system_id.decode("ascii", errors="replace")
    if system_text != "CDAS":
        raise ValueError(f"{path} 不是 CDAS 数据帧")

    return DatHeader(
        protocol="CDAS",
        system_id=system_text,
        year=year,
        month=(month_day >> 8) & 0xFF,
        day=month_day & 0xFF,
        hour=(hour_minute >> 8) & 0xFF,
        minute=hour_minute & 0xFF,
        second=second,
        length=length,
        global_alarm=global_alarm,
        acquisition_status=acquisition_status,
        channel=channel,
        frequency_hz=frequency_hz,
        acquisition_time_s=acquisition_time_s,
        point_interval_m=point_interval_m,
        point_count=point_count,
        sample_count=sample_count,
    )


def parse_guanggu_timestamp(path: Path, header_values: np.ndarray) -> str:
    match = re.match(r"(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-out\.dat$", path.name, re.IGNORECASE)
    if match:
        year, month, day, hour, minute, second = (int(value) for value in match.groups())
        try:
            return datetime(year, month, day, hour, minute, second).strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass
    raw_time = header_values[:6]
    return "头部GPS时间: " + ", ".join(f"{float(value):.6g}" for value in raw_time)


def read_guanggu_header(path: Path) -> DatHeader:
    file_size = path.stat().st_size
    if file_size < GUANGGU_HEADER_SIZE or file_size % 4 != 0:
        raise ValueError(f"{path} 不是有效的光谷互联 float32 DAT 文件")

    header_values = np.fromfile(path, dtype="<f4", count=GUANGGU_HEADER_FLOATS)
    if header_values.size != GUANGGU_HEADER_FLOATS:
        raise ValueError(f"{path} 文件太小，无法读取光谷互联头信息")

    frequency_hz = int(round(float(header_values[10])))
    start_distance_m = float(header_values[14])
    stop_distance_m = float(header_values[15])
    sensor_number = int(round(float(header_values[16])))
    if frequency_hz <= 0:
        raise ValueError(f"{path.name} 光谷互联采样率无效: {header_values[10]}")
    if sensor_number <= 0:
        raise ValueError(f"{path.name} 光谷互联点位数无效: {header_values[16]}")

    data_float_count = file_size // 4 - GUANGGU_HEADER_FLOATS
    if data_float_count <= 0 or data_float_count % sensor_number != 0:
        raise ValueError(
            f"{path.name} 光谷互联数据长度不匹配: 数据float数={data_float_count}, 点位数={sensor_number}"
        )

    sample_count = data_float_count // sensor_number
    acquisition_time_s = sample_count / frequency_hz
    if sensor_number > 1:
        point_interval_m = (stop_distance_m - start_distance_m) / (sensor_number - 1)
    else:
        point_interval_m = 0.0

    year = int(round(float(header_values[0]))) if np.isfinite(header_values[0]) else 0
    month = int(round(float(header_values[1]))) if np.isfinite(header_values[1]) else 0
    day = int(round(float(header_values[2]))) if np.isfinite(header_values[2]) else 0
    hour = int(round(float(header_values[3]))) if np.isfinite(header_values[3]) else 0
    minute = int(round(float(header_values[4]))) if np.isfinite(header_values[4]) else 0
    second = float(header_values[5]) if np.isfinite(header_values[5]) else 0.0

    acquisition_modes = {1: "CTNE", 2: "PTRI", 3: "WTRI"}
    fibre_types = {1: "SMF", 2: "MMF", 3: "MSF"}
    extra_info: dict[str, object] = {
        "采集模式": acquisition_modes.get(int(round(float(header_values[6]))), float(header_values[6])),
        "光纤类型": fibre_types.get(int(round(float(header_values[7]))), float(header_values[7])),
        "物理光纤长度(m)": float(header_values[8]),
        "折射率": float(header_values[9]),
        "脉宽(ns)": float(header_values[11]),
        "标距": float(header_values[12]),
        "空间分辨率(m)": float(header_values[13]),
        "头部连续采集时间(s)": float(header_values[17]),
        "P": float(header_values[18]),
        "TimeDecimation": float(header_values[19]),
        "Number": float(header_values[20]),
        "Window": float(header_values[21]),
        "Cutoff": float(header_values[22]),
        "Gain1": float(header_values[23]),
        "Gain2": float(header_values[24]),
        "OpticalPower": float(header_values[25]),
        "ScanThreshold": float(header_values[26]),
        "TriggerLevel(ns)": float(header_values[27]),
        "AcquisitionTime(s)": float(header_values[28]),
        "TriggerInterval(s)": float(header_values[29]),
    }

    return DatHeader(
        protocol="光谷互联",
        system_id="GUANGGU",
        year=year,
        month=month,
        day=day,
        hour=hour,
        minute=minute,
        second=second,
        length=GUANGGU_HEADER_SIZE,
        global_alarm=0,
        acquisition_status=1,
        channel=0,
        frequency_hz=frequency_hz,
        acquisition_time_s=acquisition_time_s,
        point_interval_m=point_interval_m,
        point_count=sensor_number,
        sample_count=sample_count,
        position_start_m=start_distance_m,
        raw_value_note="float32 原值",
        header_size=GUANGGU_HEADER_SIZE,
        bytes_per_sample=4,
        timestamp_text=parse_guanggu_timestamp(path, header_values),
        extra_info=extra_info,
    )


def read_header(path: Path) -> DatHeader:
    with path.open("rb") as f:
        magic = f.read(4)
    if magic == b"CDAS":
        return read_cdas_header(path)
    return read_guanggu_header(path)


def read_signal_matrix(path: Path, header: DatHeader) -> np.ndarray:
    count = header.point_count * header.sample_count
    if header.protocol == "光谷互联":
        data = np.fromfile(path, dtype="<f4", offset=header.header_size, count=count)
        if data.size != count:
            raise ValueError(f"{path.name} 数据长度不匹配")
        return data.reshape(header.sample_count, header.point_count).T.astype(np.float32, copy=False)

    data = np.fromfile(path, dtype=">i2", offset=header.header_size, count=count)
    if data.size != header.point_count * header.sample_count:
        raise ValueError(f"{path.name} 数据长度不匹配")
    return data.reshape(header.point_count, header.sample_count).astype(np.float32) / 100.0


def make_display_matrix_max_abs(matrix: np.ndarray, max_time_bins: int = DISPLAY_MAX_TIME_BINS) -> np.ndarray:
    """Aggregate only the heatmap display with signed max-abs over time bins."""
    point_count, sample_count = matrix.shape
    if sample_count <= max_time_bins:
        return matrix

    edges = np.linspace(0, sample_count, max_time_bins + 1, dtype=np.int64)
    display = np.empty((point_count, max_time_bins), dtype=np.float32)
    row_indices = np.arange(point_count)
    for bin_index in range(max_time_bins):
        start = int(edges[bin_index])
        stop = int(edges[bin_index + 1])
        if stop <= start:
            stop = min(sample_count, start + 1)
        block = matrix[:, start:stop]
        max_indices = np.argmax(np.abs(block), axis=1)
        display[:, bin_index] = block[row_indices, max_indices]
    return display


def make_frequency_spectrum_matrix(matrix: np.ndarray, sample_rate_hz: int) -> tuple[np.ndarray, np.ndarray]:
    sample_count = matrix.shape[1]
    n = 1 << (sample_count.bit_length() - 1)
    if n < 2:
        raise ValueError("采样数不足，无法计算 FFT")
    samples = matrix[:, :n].astype(np.float32, copy=False)
    samples = samples - np.mean(samples, axis=1, keepdims=True)
    window = np.hanning(n).astype(np.float32)
    spectrum = np.fft.rfft(samples * window[np.newaxis, :], axis=1)
    magnitude = (np.abs(spectrum) * 2.0 / n).astype(np.float32)
    freqs = np.fft.rfftfreq(n, d=1.0 / max(1, sample_rate_hz)).astype(np.float32)
    return freqs, magnitude


def aggregate_positive_matrix_max(
    matrix: np.ndarray,
    freqs: np.ndarray,
    max_bins: int = DISPLAY_MAX_TIME_BINS,
) -> tuple[np.ndarray, np.ndarray]:
    point_count, bin_count = matrix.shape
    if bin_count <= max_bins:
        return matrix, freqs
    edges = np.linspace(0, bin_count, max_bins + 1, dtype=np.int64)
    display = np.empty((point_count, max_bins), dtype=np.float32)
    display_freqs = np.empty(max_bins, dtype=np.float32)
    for index in range(max_bins):
        start = int(edges[index])
        stop = int(edges[index + 1])
        if stop <= start:
            stop = min(bin_count, start + 1)
        display[:, index] = np.max(matrix[:, start:stop], axis=1)
        display_freqs[index] = float((freqs[start] + freqs[stop - 1]) / 2.0)
    return display, display_freqs


def estimate_local_noise_floor_db(freqs: np.ndarray, spectrum_db: np.ndarray, window_hz: float) -> np.ndarray:
    """Estimate a curved spectral noise floor without scipy."""
    count = min(freqs.size, spectrum_db.size)
    if count == 0:
        return np.array([], dtype=np.float64)

    freqs = np.asarray(freqs[:count], dtype=np.float64)
    values = np.asarray(spectrum_db[:count], dtype=np.float64)
    finite = np.isfinite(freqs) & np.isfinite(values)
    if not np.any(finite):
        return np.zeros(count, dtype=np.float64)

    freq_span = float(np.nanmax(freqs[finite]) - np.nanmin(freqs[finite]))
    if freq_span <= 0:
        floor_value = float(np.nanmedian(values[finite]))
        return np.full(count, floor_value, dtype=np.float64)

    window_hz = max(float(window_hz), freq_span / 200.0)
    bin_count = int(np.clip(np.ceil(freq_span / window_hz * 4.0), 32, 800))
    edges = np.linspace(float(np.nanmin(freqs[finite])), float(np.nanmax(freqs[finite])), bin_count + 1)
    centers: list[float] = []
    floor_values: list[float] = []
    for index in range(bin_count):
        left = edges[index]
        right = edges[index + 1]
        if index == bin_count - 1:
            mask = finite & (freqs >= left) & (freqs <= right)
        else:
            mask = finite & (freqs >= left) & (freqs < right)
        if not np.any(mask):
            continue
        centers.append(float((left + right) / 2.0))
        floor_values.append(float(np.nanpercentile(values[mask], 35.0)))

    if not centers:
        floor_value = float(np.nanmedian(values[finite]))
        return np.full(count, floor_value, dtype=np.float64)
    if len(centers) == 1:
        return np.full(count, floor_values[0], dtype=np.float64)

    floor = np.interp(freqs, np.asarray(centers), np.asarray(floor_values), left=floor_values[0], right=floor_values[-1])
    return floor.astype(np.float64, copy=False)


def calculate_peak_line_snr(
    freqs: np.ndarray,
    magnitude: np.ndarray,
    mode: str = "auto",
    target_freq_hz: float = 100.0,
    search_half_width_hz: float = 5.0,
    noise_window_hz: float = 20.0,
    guard_band_hz: float = 1.0,
    noise_mode: str = "local",
    exclude_band_hz: float = 5.0,
) -> dict[str, float] | None:
    """Find the spectral line with the highest local peak-line SNR."""
    if freqs.size == 0 or magnitude.size == 0:
        return None
    count = min(freqs.size, magnitude.size)
    freqs = np.asarray(freqs[:count], dtype=np.float64)
    magnitude = np.asarray(magnitude[:count], dtype=np.float64)
    spectrum_db = 20.0 * np.log10(np.maximum(magnitude, 1e-12))
    finite = np.isfinite(freqs) & np.isfinite(spectrum_db)
    if not np.any(finite):
        return None

    noise_floor_db = estimate_local_noise_floor_db(freqs, spectrum_db, noise_window_hz)
    score = spectrum_db - noise_floor_db
    safe_spectrum_db = np.where(finite, spectrum_db, -np.inf)
    if mode == "target":
        search_half_width_hz = max(0.0, float(search_half_width_hz))
        candidate_mask = (
            finite
            & (freqs >= float(target_freq_hz) - search_half_width_hz)
            & (freqs <= float(target_freq_hz) + search_half_width_hz)
        )
        mode_text = f"目标频率 {target_freq_hz:.3f}±{search_half_width_hz:.3f} Hz"
    else:
        candidate_mask = finite & (freqs > 0.0)
        mode_text = "自动检测"

    if not np.any(candidate_mask):
        return None

    noise_window_hz = max(float(noise_window_hz), 0.0)
    guard_band_hz = max(float(guard_band_hz), 0.0)
    exclude_half_width_hz = max(float(exclude_band_hz), 0.0) / 2.0

    def global_noise_level_db() -> float:
        if mode == "target":
            center_freq = float(target_freq_hz)
        else:
            center_freq = float(freqs[int(candidate_indices[best_candidate_position])])
        noise_mask = (
            finite
            & (freqs > 0.0)
            & ~((freqs >= center_freq - exclude_half_width_hz) & (freqs <= center_freq + exclude_half_width_hz))
        )
        noise_values = spectrum_db[noise_mask]
        noise_values = noise_values[np.isfinite(noise_values)]
        if noise_values.size:
            return float(np.nanmean(noise_values))
        fallback = spectrum_db[finite & (freqs > 0.0)]
        fallback = fallback[np.isfinite(fallback)]
        if fallback.size:
            return float(np.nanmean(fallback))
        return float(np.nanmean(spectrum_db[finite]))

    def local_noise_level_db(peak_index: int) -> float:
        peak_freq = float(freqs[peak_index])
        side_values: list[np.ndarray] = []
        left_start = int(np.searchsorted(freqs, peak_freq - noise_window_hz, side="left"))
        left_stop = int(np.searchsorted(freqs, peak_freq - guard_band_hz, side="right"))
        right_start = int(np.searchsorted(freqs, peak_freq + guard_band_hz, side="left"))
        right_stop = int(np.searchsorted(freqs, peak_freq + noise_window_hz, side="right"))
        if left_stop > left_start:
            left_values = spectrum_db[left_start:left_stop]
            left_values = left_values[np.isfinite(left_values)]
            if left_values.size:
                side_values.append(left_values)
        if right_stop > right_start:
            right_values = spectrum_db[right_start:right_stop]
            right_values = right_values[np.isfinite(right_values)]
            if right_values.size:
                side_values.append(right_values)
        if side_values:
            return float(np.nanmedian(np.concatenate(side_values)))
        return float(noise_floor_db[peak_index])

    local_peaks = np.array([], dtype=np.int64)
    if count >= 3:
        local_peak_mask = (
            candidate_mask[1:-1]
            & (safe_spectrum_db[1:-1] >= safe_spectrum_db[:-2])
            & (safe_spectrum_db[1:-1] >= safe_spectrum_db[2:])
        )
        local_peaks = np.where(local_peak_mask)[0] + 1

    if local_peaks.size:
        candidate_indices = local_peaks
    else:
        candidate_indices = np.where(candidate_mask)[0]

    candidate_snr_values = []
    for candidate_index in candidate_indices:
        peak_value = float(spectrum_db[candidate_index])
        if noise_mode == "global":
            if mode == "target":
                center_freq = float(target_freq_hz)
            else:
                center_freq = float(freqs[int(candidate_index)])
            noise_mask = (
                finite
                & (freqs > 0.0)
                & ~((freqs >= center_freq - exclude_half_width_hz) & (freqs <= center_freq + exclude_half_width_hz))
            )
            noise_values = spectrum_db[noise_mask]
            noise_values = noise_values[np.isfinite(noise_values)]
            noise_value = float(np.nanmean(noise_values)) if noise_values.size else local_noise_level_db(int(candidate_index))
        else:
            noise_value = local_noise_level_db(int(candidate_index))
        candidate_snr_values.append(peak_value - noise_value)
    candidate_snr = np.asarray(candidate_snr_values, dtype=np.float64)
    finite_snr = np.isfinite(candidate_snr)
    if not np.any(finite_snr):
        return None
    best_candidate_position = int(np.argmax(np.where(finite_snr, candidate_snr, -np.inf)))
    peak_index = int(candidate_indices[best_candidate_position])

    peak_value_db = float(spectrum_db[peak_index])
    peak_score_db = float(score[peak_index])
    if not np.isfinite(peak_value_db):
        return None

    peak_freq = float(freqs[peak_index])
    if noise_mode == "global":
        noise_level_db = global_noise_level_db()
        noise_mode_text = f"全局底噪，排除总宽{float(exclude_band_hz):.3g} Hz"
    else:
        noise_level_db = local_noise_level_db(peak_index)
        noise_mode_text = "局部底噪"

    return {
        "peak_index": float(peak_index),
        "peak_freq": peak_freq,
        "peak_value_db": peak_value_db,
        "noise_level_db": noise_level_db,
        "noise_floor_db": float(noise_floor_db[peak_index]),
        "score_db": peak_score_db,
        "snr_db": peak_value_db - noise_level_db,
        "mode_text": mode_text,
        "noise_mode": noise_mode,
        "noise_mode_text": noise_mode_text,
        "exclude_band_hz": float(exclude_band_hz),
    }


def find_dat_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.dat"))


def format_header(path: Path, header: DatHeader, matrix: np.ndarray | None = None, color_limit: float | None = None) -> str:
    file_size = path.stat().st_size
    actual_data_bytes = max(0, file_size - header.header_size)
    status_text = "正常采集" if header.acquisition_status == 1 else "停止采集"
    lines = [
        f"文件名: {path.name}",
        f"完整路径: {path}",
        f"协议类型: {header.protocol}",
        f"系统标识: {header.system_id}",
        f"采集时间: {header.timestamp}",
        f"采样率: {header.frequency_hz} Hz",
        f"采集时长: {header.time_span_s:.6g} s",
        f"点位间隔: {header.point_interval_m:.6g} m",
        f"点位数: {header.point_count}",
        f"采样数: {header.sample_count}",
        f"位置范围: {header.position_start_m:.3f} ~ {header.position_end_m:.3f} m",
        f"文件大小: {file_size} bytes",
        f"数据区大小: {actual_data_bytes} bytes",
        f"期望数据区大小: {header.expected_data_bytes} bytes",
        f"真实值换算: {header.raw_value_note}",
    ]
    if header.protocol == "CDAS":
        lines[5:5] = [
            f"通道号: {header.channel}",
            f"采集状态: {header.acquisition_status} ({status_text})",
        ]
        lines.extend(
            [
                f"全局告警: 0x{header.global_alarm:04X}",
                f"帧长度字段: {header.length} bytes",
            ]
        )
    if header.extra_info:
        lines.append("协议扩展信息:")
        for key, value in header.extra_info.items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.6g}")
            else:
                lines.append(f"  {key}: {value}")
    if matrix is not None:
        lines.extend(
            [
                f"数据最小值: {float(np.nanmin(matrix)):.2f}",
                f"数据最大值: {float(np.nanmax(matrix)):.2f}",
                f"数据平均值: {float(np.nanmean(matrix)):.2f}",
            ]
        )
    if color_limit is not None:
        lines.append(f"色条范围: {-color_limit:.2f} ~ {color_limit:.2f}")
    return "\n".join(lines)


def format_display_info(
    paths: list[Path],
    header: DatHeader,
    matrix: np.ndarray | None = None,
    color_limit: float | None = None,
    display_matrix: np.ndarray | None = None,
) -> str:
    if len(paths) == 1:
        return format_header(paths[0], header, matrix, color_limit)

    total_size = sum(path.stat().st_size for path in paths)
    lines = [
        f"文件数量: {len(paths)}",
        f"协议类型: {header.protocol}",
        f"起始文件: {paths[0].name}",
        f"结束文件: {paths[-1].name}",
        f"起始时间: {read_header(paths[0]).timestamp}",
        f"结束时间: {read_header(paths[-1]).timestamp}",
        f"采样率: {header.frequency_hz} Hz",
        f"总采集时长: {header.time_span_s:.6g} s",
        f"点位间隔: {header.point_interval_m:.6g} m",
        f"点位数: {header.point_count}",
        f"总采样数: {header.sample_count}",
        f"位置范围: {header.position_start_m:.3f} ~ {header.position_end_m:.3f} m",
        f"文件总大小: {total_size} bytes",
        f"真实值换算: {header.raw_value_note}",
    ]
    if header.protocol == "CDAS":
        lines.insert(6, f"通道号: {header.channel}")
    if matrix is not None:
        lines.extend(
            [
                f"数据最小值: {float(np.nanmin(matrix)):.2f}",
                f"数据最大值: {float(np.nanmax(matrix)):.2f}",
                f"数据平均值: {float(np.nanmean(matrix)):.2f}",
            ]
        )
    if display_matrix is not None and matrix is not None:
        if display_matrix.shape == matrix.shape:
            lines.append("热力图显示: 原始采样点")
        else:
            lines.append(
                f"热力图显示: 时间轴最大绝对值聚合 "
                f"{matrix.shape[1]} -> {display_matrix.shape[1]} 行"
            )
            lines.append("点位分析/FFT: 使用原始未聚合数据")
    if color_limit is not None:
        lines.append(f"色条范围: {-color_limit:.2f} ~ {color_limit:.2f}")
    return "\n".join(lines)


class PointAnalysisWindow(QtWidgets.QMainWindow):
    def __init__(self, trace: np.ndarray, header: DatHeader, point_index: int, distance_m: float) -> None:
        super().__init__()
        self.setWindowTitle(f"点位分析 - #{point_index} / {distance_m:.3f} m")
        self.resize(1000, 720)

        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        self.setCentralWidget(widget)

        title = QtWidgets.QLabel(
            f"点位: {point_index}    位置: {distance_m:.3f} m    "
            f"采样率: {header.frequency_hz} Hz    采样数: {header.sample_count}"
        )
        layout.addWidget(title)
        controls = QtWidgets.QHBoxLayout()
        layout.addLayout(controls)
        self.fft_mode_combo = QtWidgets.QComboBox()
        self.fft_mode_combo.addItems(["dB谱(幅值)", "功率谱(线性)"])
        controls.addWidget(QtWidgets.QLabel("FFT谱类型"))
        controls.addWidget(self.fft_mode_combo)
        controls.addStretch(1)

        snr_controls = QtWidgets.QHBoxLayout()
        layout.addLayout(snr_controls)
        self.snr_mode_combo = QtWidgets.QComboBox()
        self.snr_mode_combo.addItems(["自动检测", "目标频率"])
        self.snr_target_freq = QtWidgets.QDoubleSpinBox()
        self.snr_search_half_width = QtWidgets.QDoubleSpinBox()
        self.snr_noise_window = QtWidgets.QDoubleSpinBox()
        self.snr_guard_band = QtWidgets.QDoubleSpinBox()
        for spin in (
            self.snr_target_freq,
            self.snr_search_half_width,
            self.snr_noise_window,
            self.snr_guard_band,
        ):
            spin.setRange(0.0, 1_000_000.0)
            spin.setDecimals(3)
            spin.setSingleStep(1.0)
            spin.setKeyboardTracking(False)
        self.snr_target_freq.setValue(100.0)
        self.snr_search_half_width.setValue(5.0)
        self.snr_noise_window.setValue(20.0)
        self.snr_guard_band.setValue(1.0)
        snr_controls.addWidget(QtWidgets.QLabel("峰值模式"))
        snr_controls.addWidget(self.snr_mode_combo)
        snr_controls.addSpacing(12)
        snr_controls.addWidget(QtWidgets.QLabel("目标频率(Hz)"))
        snr_controls.addWidget(self.snr_target_freq)
        snr_controls.addWidget(QtWidgets.QLabel("搜索半宽(Hz)"))
        snr_controls.addWidget(self.snr_search_half_width)
        snr_controls.addWidget(QtWidgets.QLabel("噪声窗口(Hz)"))
        snr_controls.addWidget(self.snr_noise_window)
        snr_controls.addWidget(QtWidgets.QLabel("保护带宽(Hz)"))
        snr_controls.addWidget(self.snr_guard_band)
        snr_controls.addStretch(1)

        phase_range_controls = QtWidgets.QHBoxLayout()
        layout.addLayout(phase_range_controls)
        self.phase_x_min = QtWidgets.QDoubleSpinBox()
        self.phase_x_max = QtWidgets.QDoubleSpinBox()
        self.phase_y_min = QtWidgets.QDoubleSpinBox()
        self.phase_y_max = QtWidgets.QDoubleSpinBox()
        for spin in (self.phase_x_min, self.phase_x_max, self.phase_y_min, self.phase_y_max):
            spin.setRange(-1_000_000_000.0, 1_000_000_000.0)
            spin.setDecimals(6)
            spin.setSingleStep(0.1)
            spin.setKeyboardTracking(False)
        phase_apply_btn = QtWidgets.QPushButton("应用相位坐标")
        phase_full_btn = QtWidgets.QPushButton("相位显示全部")
        phase_range_controls.addWidget(QtWidgets.QLabel("相位X时间(s):"))
        phase_range_controls.addWidget(self.phase_x_min)
        phase_range_controls.addWidget(QtWidgets.QLabel("到"))
        phase_range_controls.addWidget(self.phase_x_max)
        phase_range_controls.addSpacing(12)
        phase_range_controls.addWidget(QtWidgets.QLabel("相位Y:"))
        phase_range_controls.addWidget(self.phase_y_min)
        phase_range_controls.addWidget(QtWidgets.QLabel("到"))
        phase_range_controls.addWidget(self.phase_y_max)
        phase_range_controls.addWidget(phase_apply_btn)
        phase_range_controls.addWidget(phase_full_btn)
        phase_range_controls.addStretch(1)

        self.time_axis = np.arange(trace.size, dtype=np.float32) / max(1, header.frequency_hz)
        self.trace = trace
        self.time_plot = pg.PlotWidget(title="相位数据")
        self.time_plot.setLabel("bottom", "Time", units="s")
        self.time_plot.setLabel("left", "Value")
        self.time_plot.plot(self.time_axis, trace, pen=pg.mkPen("#1769aa", width=1))
        self.time_plot.showGrid(x=True, y=True, alpha=0.25)
        layout.addWidget(self.time_plot, stretch=1)
        phase_apply_btn.clicked.connect(self.apply_phase_axis_range)
        phase_full_btn.clicked.connect(self.show_full_phase_range)
        for spin in (self.phase_x_min, self.phase_x_max, self.phase_y_min, self.phase_y_max):
            spin.editingFinished.connect(self.apply_phase_axis_range)
        self.time_plot.getViewBox().sigRangeChanged.connect(self.on_phase_view_range_changed)

        n = 1 << (trace.size.bit_length() - 1)
        samples = trace[:n] - np.mean(trace[:n])
        window_arr = np.hanning(n).astype(np.float32)
        spectrum = np.fft.rfft(samples * window_arr)
        self.freqs = np.fft.rfftfreq(n, d=1.0 / max(1, header.frequency_hz))
        self.magnitude = np.abs(spectrum) * 2.0 / n

        self.freq_plot = pg.PlotWidget(title="FFT频率谱")
        self.freq_plot.setLabel("bottom", "Frequency", units="Hz")
        self.freq_curve = self.freq_plot.plot(pen=pg.mkPen("#b21f35", width=1))
        self.freq_plot.showGrid(x=True, y=True, alpha=0.25)
        layout.addWidget(self.freq_plot, stretch=1)
        self.snr_label = QtWidgets.QLabel("峰值谱线SNR: --")
        layout.addWidget(self.snr_label)
        self.fft_mode_combo.currentIndexChanged.connect(self.update_fft_plot)
        self.snr_mode_combo.currentIndexChanged.connect(self.update_fft_plot)
        for spin in (
            self.snr_target_freq,
            self.snr_search_half_width,
            self.snr_noise_window,
            self.snr_guard_band,
        ):
            spin.valueChanged.connect(self.update_fft_plot)
        self.show_full_phase_range()
        self.update_fft_plot()

    def set_phase_range_controls(self, x0: float, x1: float, y0: float, y1: float) -> None:
        for spin, value in (
            (self.phase_x_min, x0),
            (self.phase_x_max, x1),
            (self.phase_y_min, y0),
            (self.phase_y_max, y1),
        ):
            spin.blockSignals(True)
            spin.setValue(float(value))
            spin.blockSignals(False)

    def on_phase_view_range_changed(self) -> None:
        (x0, x1), (y0, y1) = self.time_plot.getViewBox().viewRange()
        self.set_phase_range_controls(x0, x1, y0, y1)

    def apply_phase_axis_range(self) -> None:
        x0 = float(self.phase_x_min.value())
        x1 = float(self.phase_x_max.value())
        y0 = float(self.phase_y_min.value())
        y1 = float(self.phase_y_max.value())
        if x1 <= x0 or y1 <= y0:
            return
        self.time_plot.setXRange(x0, x1, padding=0.0)
        self.time_plot.setYRange(y0, y1, padding=0.0)

    def show_full_phase_range(self) -> None:
        if self.time_axis.size:
            x0 = float(self.time_axis[0])
            x1 = float(self.time_axis[-1])
        else:
            x0, x1 = 0.0, 1.0
        if x1 <= x0:
            x1 = x0 + 1.0 / 1000.0
        finite_trace = self.trace[np.isfinite(self.trace)]
        if finite_trace.size:
            y0 = float(np.min(finite_trace))
            y1 = float(np.max(finite_trace))
        else:
            y0, y1 = -1.0, 1.0
        if y1 <= y0:
            pad = max(1e-6, abs(y0) * 0.05)
            y0 -= pad
            y1 += pad
        else:
            pad = (y1 - y0) * 0.05
            y0 -= pad
            y1 += pad
        self.set_phase_range_controls(x0, x1, y0, y1)
        self.time_plot.setXRange(x0, x1, padding=0.0)
        self.time_plot.setYRange(y0, y1, padding=0.0)

    def update_fft_plot(self) -> None:
        if self.fft_mode_combo.currentIndex() == 0:
            y = 20.0 * np.log10(np.maximum(self.magnitude, 1e-12))
            self.freq_plot.setTitle("FFT频率谱 - dB谱(幅值)")
            self.freq_plot.setLabel("left", "Amplitude", units="dB")
        else:
            y = np.square(self.magnitude)
            self.freq_plot.setTitle("FFT频率谱 - 功率谱(线性)")
            self.freq_plot.setLabel("left", "Power")
        self.freq_curve.setData(self.freqs, y)
        mode = "target" if self.snr_mode_combo.currentIndex() == 1 else "auto"
        snr = calculate_peak_line_snr(
            self.freqs,
            self.magnitude,
            mode=mode,
            target_freq_hz=float(self.snr_target_freq.value()),
            search_half_width_hz=float(self.snr_search_half_width.value()),
            noise_window_hz=float(self.snr_noise_window.value()),
            guard_band_hz=float(self.snr_guard_band.value()),
            noise_mode=getattr(self, "snr_noise_mode", "local"),
            exclude_band_hz=float(getattr(self, "snr_exclude_band_hz", 5.0)),
        )
        if snr is None:
            self.snr_label.setText("峰值谱线SNR: 未找到有效特征峰")
        else:
            self.snr_label.setText(
                "峰值谱线SNR: "
                f"{snr['snr_db']:.4g} dB    "
                f"模式 {snr['mode_text']}    "
                "选峰依据 峰值谱线SNR最大    "
                f"特征峰 {snr['peak_freq']:.3f} Hz / {snr['peak_value_db']:.4g} dB    "
                f"局部噪声 {snr['noise_level_db']:.4g} dB    "
                f"相对底噪突出度 {snr['score_db']:.4g} dB"
            )
        if snr is not None:
            self.snr_label.setText(
                "峰值谱线SNR: "
                f"{snr['snr_db']:.4g} dB    "
                f"模式 {snr['mode_text']}    "
                f"底噪口径 {snr.get('noise_mode_text', '局部底噪')}    "
                f"特征峰 {snr['peak_freq']:.3f} Hz / {snr['peak_value_db']:.4g} dB    "
                f"底噪 {snr['noise_level_db']:.4g} dB"
            )
        self.freq_plot.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)


class FrequencyProfileWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
        positions: np.ndarray,
        profile: np.ndarray,
        selected_freq_hz: float,
        clicked_freq_hz: float,
        mode_index: int,
    ) -> None:
        super().__init__()
        self.setWindowTitle(f"频率空间分布 - {selected_freq_hz:.3f} Hz")
        self.resize(900, 560)

        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        self.setCentralWidget(widget)

        if mode_index == 0:
            y = 20.0 * np.log10(np.maximum(profile, 1e-12))
            title = f"频率空间分布 - dB谱(幅值) - {selected_freq_hz:.3f} Hz"
            y_label = ("Amplitude", "dB")
        else:
            y = np.square(profile)
            title = f"频率空间分布 - 功率谱(线性) - {selected_freq_hz:.3f} Hz"
            y_label = ("Power", None)

        info = QtWidgets.QLabel(
            f"点击频率: {clicked_freq_hz:.3f} Hz    "
            f"实际FFT频点: {selected_freq_hz:.3f} Hz    "
            f"空间点数: {positions.size}"
        )
        layout.addWidget(info)

        plot = pg.PlotWidget(title=title)
        plot.setLabel("bottom", "Position", units="m")
        if y_label[1]:
            plot.setLabel("left", y_label[0], units=y_label[1])
        else:
            plot.setLabel("left", y_label[0])
        plot.plot(positions, y, pen=pg.mkPen("#1769aa", width=2), symbol="o", symbolSize=4)
        plot.showGrid(x=True, y=True, alpha=0.25)
        layout.addWidget(plot, stretch=1)


class SpectrumHeatmapWindow(QtWidgets.QMainWindow):
    def __init__(self, matrix: np.ndarray, header: DatHeader, title: str) -> None:
        super().__init__()
        self.setWindowTitle(title)
        self.resize(1100, 760)
        self.header = header
        self.profile_windows: list[QtWidgets.QMainWindow] = []

        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        self.setCentralWidget(widget)

        controls = QtWidgets.QHBoxLayout()
        layout.addLayout(controls)
        range_controls = QtWidgets.QHBoxLayout()
        layout.addLayout(range_controls)

        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["dB谱(幅值)", "功率谱(线性)"])
        self.level_min = QtWidgets.QDoubleSpinBox()
        self.level_max = QtWidgets.QDoubleSpinBox()
        for spin in (self.level_min, self.level_max):
            spin.setRange(-300.0, 1_000_000.0)
            spin.setDecimals(4)
            spin.setSingleStep(1.0)
            spin.setKeyboardTracking(False)
        controls.addWidget(QtWidgets.QLabel("谱类型"))
        controls.addWidget(self.mode_combo)
        controls.addSpacing(12)
        controls.addWidget(QtWidgets.QLabel("色条下限"))
        controls.addWidget(self.level_min)
        controls.addWidget(QtWidgets.QLabel("上限"))
        controls.addWidget(self.level_max)
        controls.addStretch(1)

        self.x_min = QtWidgets.QDoubleSpinBox()
        self.x_max = QtWidgets.QDoubleSpinBox()
        self.y_min = QtWidgets.QDoubleSpinBox()
        self.y_max = QtWidgets.QDoubleSpinBox()
        for spin in (self.x_min, self.x_max, self.y_min, self.y_max):
            spin.setRange(-1_000_000.0, 1_000_000.0)
            spin.setDecimals(3)
            spin.setSingleStep(1.0)
            spin.setKeyboardTracking(False)
        apply_range_btn = QtWidgets.QPushButton("应用坐标范围")
        full_range_btn = QtWidgets.QPushButton("显示全部")
        range_controls.addWidget(QtWidgets.QLabel("X位置(m):"))
        range_controls.addWidget(self.x_min)
        range_controls.addWidget(QtWidgets.QLabel("到"))
        range_controls.addWidget(self.x_max)
        range_controls.addSpacing(12)
        range_controls.addWidget(QtWidgets.QLabel("Y频率(Hz):"))
        range_controls.addWidget(self.y_min)
        range_controls.addWidget(QtWidgets.QLabel("到"))
        range_controls.addWidget(self.y_max)
        range_controls.addWidget(apply_range_btn)
        range_controls.addWidget(full_range_btn)
        range_controls.addStretch(1)

        graphics = pg.GraphicsLayoutWidget()
        layout.addWidget(graphics, stretch=1)
        self.plot = graphics.addPlot(row=0, col=0)
        self.plot.setLabel("bottom", "Position", units="m")
        self.plot.setLabel("left", "Frequency", units="Hz")
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.invertY(False)

        freqs, spectrum = make_frequency_spectrum_matrix(matrix, header.frequency_hz)
        self.freqs = freqs
        self.amplitude_full = spectrum
        self.display_amplitude, self.display_freqs = aggregate_positive_matrix_max(spectrum, freqs)
        self.current_display = self.display_amplitude
        self.freq_step = self._estimate_freq_step(self.display_freqs)
        self.freq_axis_min = float(self.display_freqs[0] - self.freq_step / 2.0) if self.display_freqs.size else 0.0
        self.freq_axis_max = float(self.display_freqs[-1] + self.freq_step / 2.0) if self.display_freqs.size else 0.0
        self.x_center_min = header.position_start_m
        self.x_center_max = header.position_end_m
        self.x_axis_min = header.position_start_m - header.point_interval_m / 2.0
        self.x_axis_max = header.position_end_m + header.point_interval_m / 2.0
        self.image_item = pg.ImageItem()
        self.plot.addItem(self.image_item)
        self.image_item.setLookupTable(pg.colormap.get("turbo").getLookupTable(0.0, 1.0, 256))
        self.frequency_marker = pg.InfiniteLine(
            angle=0,
            movable=False,
            pen=pg.mkPen("#1f4d78", width=2, style=QtCore.Qt.PenStyle.DashLine),
        )
        self.frequency_marker.hide()
        self.plot.addItem(self.frequency_marker)
        self.image_rect = QtCore.QRectF(
            self.x_axis_min,
            self.freq_axis_min,
            self.x_axis_max - self.x_axis_min,
            self.freq_axis_max - self.freq_axis_min,
        )
        self.refresh_timer = QtCore.QTimer(self)
        self.refresh_timer.setSingleShot(True)
        self.refresh_timer.timeout.connect(self.refresh_spectrum_for_view)
        self.color_bar = pg.ColorBarItem(values=(0.0, 1.0), colorMap=pg.colormap.get("turbo"))
        self.color_bar.setImageItem(self.image_item, insert_in=self.plot)

        self.mode_combo.currentIndexChanged.connect(lambda _index: self.update_spectrum_mode(reset_levels=True))
        self.level_min.valueChanged.connect(self.update_levels)
        self.level_max.valueChanged.connect(self.update_levels)
        apply_range_btn.clicked.connect(self.apply_axis_range)
        full_range_btn.clicked.connect(self.show_full_range)
        for spin in (self.x_min, self.x_max, self.y_min, self.y_max):
            spin.editingFinished.connect(self.apply_axis_range)
        self.plot.vb.sigRangeChanged.connect(self.on_view_range_changed)
        self.plot.scene().sigMouseClicked.connect(self.on_spectrum_clicked)

        self.update_spectrum_mode()
        self.show_full_range()

        note = QtWidgets.QLabel(
            f"FFT使用原始未聚合数据: {matrix.shape[0]} 点位 x {matrix.shape[1]} 采样；"
            f"频谱显示矩阵: {self.display_amplitude.shape[0]} 点位 x {self.display_amplitude.shape[1]} 频率点；"
            f"频率范围: {self.freq_axis_min:.3f} ~ {self.freq_axis_max:.3f} Hz；"
            "点击热力图可绘制该频率的空间分布曲线"
        )
        self.profile_status = note
        layout.addWidget(note)

    @staticmethod
    def _estimate_freq_step(freqs: np.ndarray) -> float:
        if freqs.size >= 2:
            diffs = np.diff(freqs.astype(np.float64))
            positive = diffs[diffs > 0]
            if positive.size:
                return float(np.median(positive))
        return 1.0

    def set_range_controls(self, x0: float, x1: float, y0: float, y1: float) -> None:
        for spin, value in (
            (self.x_min, x0),
            (self.x_max, x1),
            (self.y_min, y0),
            (self.y_max, y1),
        ):
            spin.blockSignals(True)
            spin.setValue(float(value))
            spin.blockSignals(False)

    def on_view_range_changed(self) -> None:
        (x0, x1), (y0, y1) = self.plot.vb.viewRange()
        self.set_range_controls(x0, x1, y0, y1)
        self.refresh_timer.start(80)

    def apply_axis_range(self) -> None:
        x0 = float(self.x_min.value())
        x1 = float(self.x_max.value())
        y0 = float(self.y_min.value())
        y1 = float(self.y_max.value())
        if x1 <= x0 or y1 <= y0:
            return
        self.plot.setXRange(x0, x1, padding=0.0)
        self.plot.setYRange(y0, y1, padding=0.0)

    def show_full_range(self) -> None:
        x0 = self.x_center_min
        x1 = self.x_center_max
        y0 = 0.0
        y1 = float(self.freqs[-1]) if self.freqs.size else 0.0
        self.set_range_controls(x0, x1, y0, y1)
        self.plot.setXRange(x0, x1, padding=0.0)
        self.plot.setYRange(y0, y1, padding=0.0)
        self.refresh_timer.start(0)

    def make_visible_peak_amplitude(self) -> tuple[np.ndarray, np.ndarray]:
        if self.freqs.size == 0:
            return self.amplitude_full, self.freqs
        _x_range, y_range = self.plot.vb.viewRange()
        y0, y1 = sorted(float(v) for v in y_range)
        y0 = max(0.0, y0)
        y1 = min(float(self.freqs[-1]), y1)
        if y1 <= y0:
            y0, y1 = 0.0, float(self.freqs[-1])

        visible = np.where((self.freqs >= y0) & (self.freqs <= y1))[0]
        if visible.size == 0:
            nearest = int(np.argmin(np.abs(self.freqs - (y0 + y1) / 2.0)))
            visible = np.array([nearest], dtype=np.int64)

        start = int(visible[0])
        stop = int(visible[-1]) + 1
        amplitude = self.amplitude_full[:, start:stop]
        freqs = self.freqs[start:stop]
        target_bins = max(64, int(self.plot.vb.height()) if self.plot.vb.height() > 0 else 800)
        if amplitude.shape[1] <= target_bins:
            return amplitude, freqs
        return aggregate_positive_matrix_max(amplitude, freqs, target_bins)

    def refresh_spectrum_for_view(self) -> None:
        self.display_amplitude, self.display_freqs = self.make_visible_peak_amplitude()
        self.freq_step = self._estimate_freq_step(self.display_freqs)
        self.freq_axis_min = float(self.display_freqs[0] - self.freq_step / 2.0) if self.display_freqs.size else 0.0
        self.freq_axis_max = float(self.display_freqs[-1] + self.freq_step / 2.0) if self.display_freqs.size else 0.0
        self.image_rect = QtCore.QRectF(
            self.x_axis_min,
            self.freq_axis_min,
            self.x_axis_max - self.x_axis_min,
            self.freq_axis_max - self.freq_axis_min,
        )
        self.update_spectrum_mode(reset_levels=False)

    def update_spectrum_mode(self, reset_levels: bool = True) -> None:
        if self.mode_combo.currentIndex() == 0:
            self.current_display = 20.0 * np.log10(np.maximum(self.display_amplitude, 1e-12))
            self.plot.setTitle("频谱热力图 - dB谱(幅值)")
            low = float(np.nanpercentile(self.current_display, 1.0))
            high = float(np.nanpercentile(self.current_display, 99.5))
        else:
            self.current_display = np.square(self.display_amplitude)
            self.plot.setTitle("频谱热力图 - 功率谱(线性)")
            low = 0.0
            high = float(np.nanpercentile(self.current_display, 99.5))
        if not np.isfinite(low):
            low = 0.0
        if not np.isfinite(high) or high <= low:
            high = low + 1.0
        if reset_levels:
            self.level_min.blockSignals(True)
            self.level_max.blockSignals(True)
            self.level_min.setValue(low)
            self.level_max.setValue(high)
            self.level_min.blockSignals(False)
            self.level_max.blockSignals(False)
        self.image_item.setImage(self.current_display.T, autoLevels=False)
        self.image_item.setRect(self.image_rect)
        self.update_levels()

    def update_levels(self) -> None:
        low = float(self.level_min.value())
        high = float(self.level_max.value())
        if high <= low:
            return
        self.image_item.setLevels((low, high))
        self.color_bar.setLevels((low, high))

    def on_spectrum_clicked(self, event: object) -> None:
        if self.freqs.size == 0 or self.amplitude_full.size == 0:
            return
        if not hasattr(event, "scenePos"):
            return
        pos = self.plot.vb.mapSceneToView(event.scenePos())
        clicked_freq_hz = float(pos.y())
        if clicked_freq_hz < 0.0 or clicked_freq_hz > float(self.freqs[-1]):
            return

        freq_index = int(np.argmin(np.abs(self.freqs - clicked_freq_hz)))
        selected_freq_hz = float(self.freqs[freq_index])
        profile = self.amplitude_full[:, freq_index].astype(np.float32, copy=True)
        positions = (
            self.header.position_start_m
            + np.arange(self.header.point_count, dtype=np.float32) * self.header.point_interval_m
        )

        self.frequency_marker.setValue(selected_freq_hz)
        self.frequency_marker.show()
        self.profile_status.setText(
            f"已选择频率: {selected_freq_hz:.3f} Hz "
            f"(点击 {clicked_freq_hz:.3f} Hz, 最近FFT频点, 空间点数 {positions.size})"
        )

        window = FrequencyProfileWindow(
            positions,
            profile,
            selected_freq_hz,
            clicked_freq_hz,
            self.mode_combo.currentIndex(),
        )
        self.profile_windows.append(window)
        window.show()


class DatViewer(QtWidgets.QMainWindow):
    def __init__(self, start_path: Path | None = None) -> None:
        super().__init__()
        self.setWindowTitle("CDAS/TDAS DAT Viewer - PyQtGraph")
        self.resize(1400, 860)

        self.files: list[Path] = []
        self.headers: dict[Path, DatHeader] = {}
        self.current_path: Path | None = None
        self.current_paths: list[Path] = []
        self.current_header: DatHeader | None = None
        self.current_matrix: np.ndarray | None = None
        self.current_display_matrix: np.ndarray | None = None
        self.analysis_windows: list[QtWidgets.QMainWindow] = []

        self._build_ui()
        if start_path is not None:
            self.load_path(start_path)
        else:
            self.status.setText("请选择文件夹或DAT文件后开始读取")
            self.info.setPlainText("请选择文件夹或DAT文件后开始读取。")

    def _build_ui(self) -> None:
        pg.setConfigOptions(imageAxisOrder="row-major", background="w", foreground="k", antialias=True)

        central = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(central)
        self.setCentralWidget(central)

        toolbar = QtWidgets.QHBoxLayout()
        root.addLayout(toolbar)
        rangebar = QtWidgets.QHBoxLayout()
        root.addLayout(rangebar)

        self.path_edit = QtWidgets.QLineEdit()
        self.path_edit.setPlaceholderText("请选择文件夹或DAT文件")
        folder_btn = QtWidgets.QPushButton("选择文件夹")
        file_btn = QtWidgets.QPushButton("选择DAT文件")
        reload_btn = QtWidgets.QPushButton("重新读取")
        show_selected_btn = QtWidgets.QPushButton("显示选中文件")
        spectrum_btn = QtWidgets.QPushButton("频谱热力图")
        self.color_limit = QtWidgets.QDoubleSpinBox()
        self.color_limit.setRange(0.01, 100.0)
        self.color_limit.setDecimals(2)
        self.color_limit.setSingleStep(0.1)
        self.color_limit.setValue(1.58)
        self.x_min = QtWidgets.QDoubleSpinBox()
        self.x_max = QtWidgets.QDoubleSpinBox()
        self.y_min = QtWidgets.QDoubleSpinBox()
        self.y_max = QtWidgets.QDoubleSpinBox()
        for spin in (self.x_min, self.x_max, self.y_min, self.y_max):
            spin.setRange(-1_000_000.0, 1_000_000.0)
            spin.setDecimals(3)
            spin.setSingleStep(1.0)
            spin.setKeyboardTracking(False)
        apply_range_btn = QtWidgets.QPushButton("应用坐标范围")
        full_range_btn = QtWidgets.QPushButton("显示全部")
        x_select_btn = QtWidgets.QPushButton("X轴框选")
        x_apply_btn = QtWidgets.QPushButton("应用X选择")

        toolbar.addWidget(folder_btn)
        toolbar.addWidget(self.path_edit, stretch=1)
        toolbar.addWidget(file_btn)
        toolbar.addWidget(reload_btn)
        toolbar.addWidget(show_selected_btn)
        toolbar.addWidget(spectrum_btn)
        toolbar.addWidget(QtWidgets.QLabel("色条范围±"))
        toolbar.addWidget(self.color_limit)
        rangebar.addWidget(QtWidgets.QLabel("X位置(m):"))
        rangebar.addWidget(self.x_min)
        rangebar.addWidget(QtWidgets.QLabel("到"))
        rangebar.addWidget(self.x_max)
        rangebar.addSpacing(12)
        rangebar.addWidget(QtWidgets.QLabel("Y时间(s):"))
        rangebar.addWidget(self.y_min)
        rangebar.addWidget(QtWidgets.QLabel("到"))
        rangebar.addWidget(self.y_max)
        rangebar.addWidget(apply_range_btn)
        rangebar.addWidget(full_range_btn)
        rangebar.addWidget(x_select_btn)
        rangebar.addWidget(x_apply_btn)
        rangebar.addStretch(1)

        folder_btn.clicked.connect(self.choose_folder)
        file_btn.clicked.connect(self.choose_file)
        reload_btn.clicked.connect(self.reload_current_path)
        show_selected_btn.clicked.connect(self.show_selected_files)
        spectrum_btn.clicked.connect(self.show_spectrum_heatmap)
        self.color_limit.valueChanged.connect(self.update_levels_only)
        apply_range_btn.clicked.connect(self.apply_axis_range)
        full_range_btn.clicked.connect(self.show_full_range)
        x_select_btn.clicked.connect(self.toggle_x_selector)
        x_apply_btn.clicked.connect(self.apply_x_selector)
        for spin in (self.x_min, self.x_max, self.y_min, self.y_max):
            spin.editingFinished.connect(self.apply_axis_range)

        splitter = QtWidgets.QSplitter()
        splitter.setOrientation(QtCore.Qt.Orientation.Horizontal)
        root.addWidget(splitter, stretch=1)

        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        self.file_list = QtWidgets.QListWidget()
        self.file_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        left_layout.addWidget(QtWidgets.QLabel("文件名"))
        left_layout.addWidget(self.file_list)
        splitter.addWidget(left_panel)

        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([300, 1100])

        graphics = pg.GraphicsLayoutWidget()
        right_layout.addWidget(graphics, stretch=1)

        self.plot = graphics.addPlot(row=0, col=0)
        self.plot.setLabel("bottom", "Position", units="m")
        self.plot.setLabel("left", "Time", units="s")
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.invertY(True)
        self.image_item = pg.ImageItem()
        self.plot.addItem(self.image_item)
        self.marker = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("k", width=2, style=QtCore.Qt.PenStyle.DashLine))
        self.marker.hide()
        self.plot.addItem(self.marker)
        self.x_selector = pg.LinearRegionItem(
            orientation="vertical",
            brush=pg.mkBrush(60, 120, 220, 45),
            pen=pg.mkPen(40, 80, 180, width=1),
            hoverBrush=pg.mkBrush(60, 120, 220, 70),
            movable=True,
        )
        self.x_selector.hide()
        self.plot.addItem(self.x_selector)
        self.x_selector.sigRegionChanged.connect(self.on_x_selector_changed)

        self.color_bar = pg.ColorBarItem(values=(-1.58, 1.58), colorMap=pg.colormap.get("turbo"))
        self.color_bar.setImageItem(self.image_item, insert_in=self.plot)
        self.image_item.setLookupTable(pg.colormap.get("turbo").getLookupTable(0.0, 1.0, 256))

        self.info = QtWidgets.QPlainTextEdit()
        self.info.setReadOnly(True)
        self.info.setMinimumHeight(170)
        right_layout.addWidget(self.info)

        self.status = QtWidgets.QLabel("Ready")
        root.addWidget(self.status)

        self.file_list.currentRowChanged.connect(self.on_file_selected)
        self.plot.scene().sigMouseClicked.connect(self.on_plot_clicked)
        self.plot.vb.sigRangeChanged.connect(self.on_view_range_changed)

    def choose_folder(self) -> None:
        selected = QtWidgets.QFileDialog.getExistingDirectory(self, "选择数据文件夹", self.dialog_start_path())
        if selected:
            self.load_path(Path(selected))

    def choose_file(self) -> None:
        selected, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择DAT文件",
            self.dialog_start_path(),
            "DAT files (*.dat);;All files (*.*)",
        )
        if selected:
            self.load_path(Path(selected))

    def dialog_start_path(self) -> str:
        text = self.path_edit.text().strip()
        if text:
            return text
        return str(Path(".").resolve())

    def reload_current_path(self) -> None:
        text = self.path_edit.text().strip()
        if not text:
            self.status.setText("请先选择文件夹或DAT文件")
            return
        self.load_path(Path(text))

    def load_path(self, path: Path) -> None:
        self.path_edit.setText(str(path))
        self.status.setText("正在扫描DAT文件...")
        QtWidgets.QApplication.processEvents()

        self.files = find_dat_files(path)
        self.headers = {}
        self.file_list.clear()
        for file_path in self.files:
            try:
                self.headers[file_path] = read_header(file_path)
                self.file_list.addItem(file_path.name)
            except Exception as exc:
                self.file_list.addItem(f"{file_path.name} - {exc}")

        self.status.setText(f"找到 {len(self.files)} 个DAT文件")
        if self.files:
            self.file_list.setCurrentRow(0)

    def on_file_selected(self, row: int) -> None:
        if row < 0 or row >= len(self.files):
            return
        path = self.files[row]
        self.display_paths([path])

    def show_selected_files(self) -> None:
        rows = sorted(index.row() for index in self.file_list.selectedIndexes())
        paths = [self.files[row] for row in rows if 0 <= row < len(self.files)]
        if not paths:
            row = self.file_list.currentRow()
            if 0 <= row < len(self.files):
                paths = [self.files[row]]
        if paths:
            self.display_paths(paths)

    def validate_multi_headers(self, paths: list[Path]) -> DatHeader:
        base = self.headers.get(paths[0])
        if base is None:
            raise ValueError(f"{paths[0].name} 文件头无效")
        total_samples = 0
        for path in paths:
            header = self.headers.get(path)
            if header is None:
                raise ValueError(f"{path.name} 文件头无效")
            if header.protocol != base.protocol:
                raise ValueError(
                    f"协议类型不同，无法拼接: {path.name}={header.protocol}, "
                    f"{paths[0].name}={base.protocol}"
                )
            if header.point_count != base.point_count:
                raise ValueError(
                    f"点位数不同，无法拼接: {path.name}={header.point_count}, "
                    f"{paths[0].name}={base.point_count}"
                )
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
                    f"采样率不同，采样方式不同，无法拼接: {path.name}={header.frequency_hz} Hz, "
                    f"{paths[0].name}={base.frequency_hz} Hz"
                )
            if header.protocol == "CDAS" and header.channel != base.channel:
                raise ValueError(
                    f"通道号不同，无法拼接: {path.name}=通道{header.channel}, "
                    f"{paths[0].name}=通道{base.channel}"
                )
            total_samples += header.sample_count

        return replace(
            base,
            sample_count=total_samples,
            acquisition_time_s=total_samples / max(1, base.frequency_hz),
        )

    def display_paths(self, paths: list[Path]) -> None:
        if not paths:
            return
        paths = sorted(paths, key=lambda p: self.files.index(p) if p in self.files else str(p))
        try:
            header = self.validate_multi_headers(paths)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "无法显示多文件", str(exc))
            self.status.setText("多文件参数不一致")
            return

        first_path = paths[0]

        try:
            if len(paths) == 1:
                self.status.setText(f"正在读取 {first_path.name} ...")
            else:
                self.status.setText(f"正在读取并拼接 {len(paths)} 个文件 ...")
            QtWidgets.QApplication.processEvents()
            matrices = []
            for index, file_path in enumerate(paths, start=1):
                file_header = self.headers[file_path]
                self.status.setText(f"正在读取 {index}/{len(paths)}: {file_path.name}")
                QtWidgets.QApplication.processEvents()
                matrices.append(read_signal_matrix(file_path, file_header))
            matrix = matrices[0] if len(matrices) == 1 else np.concatenate(matrices, axis=1)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "读取失败", str(exc))
            self.status.setText("读取失败")
            return

        self.current_path = first_path
        self.current_paths = paths
        self.current_header = header
        self.current_matrix = matrix
        self.current_display_matrix = make_display_matrix_max_abs(matrix)

        image = self.current_display_matrix.T
        self.image_item.setImage(image, autoLevels=False)
        x_width = max(header.distance_span_m, header.point_interval_m)
        rect = QtCore.QRectF(header.position_start_m, 0.0, x_width, header.time_span_s)
        self.image_item.setRect(rect)
        self.set_range_controls(header.position_start_m, header.position_start_m + x_width, 0.0, header.time_span_s)
        self.show_full_range()
        self.marker.hide()
        self.x_selector.hide()
        self.update_levels_only()
        self.info.setPlainText(
            format_display_info(paths, header, matrix, self.color_limit.value(), self.current_display_matrix)
        )
        if len(paths) == 1:
            self.status.setText(f"已显示 {first_path.name}")
        else:
            self.status.setText(f"已拼接显示 {len(paths)} 个文件，总时长 {header.time_span_s:.3f} s")

    def set_range_controls(self, x0: float, x1: float, y0: float, y1: float) -> None:
        for spin, value in (
            (self.x_min, x0),
            (self.x_max, x1),
            (self.y_min, y0),
            (self.y_max, y1),
        ):
            spin.blockSignals(True)
            spin.setValue(float(value))
            spin.blockSignals(False)

    def on_view_range_changed(self) -> None:
        if self.current_header is None:
            return
        (x0, x1), (y0, y1) = self.plot.vb.viewRange()
        self.set_range_controls(x0, x1, y0, y1)

    def toggle_x_selector(self) -> None:
        if self.current_header is None:
            return
        if self.x_selector.isVisible():
            self.x_selector.hide()
            self.status.setText("已关闭X轴框选")
            return
        (x0, x1), _ = self.plot.vb.viewRange()
        span = x1 - x0
        if span <= 0:
            return
        self.x_selector.setRegion((x0 + span * 0.25, x0 + span * 0.75))
        self.x_selector.show()
        self.status.setText("拖动蓝色区域或左右边界选择X轴范围，然后点“应用X选择”")

    def on_x_selector_changed(self) -> None:
        if not self.x_selector.isVisible():
            return
        x0, x1 = sorted(float(v) for v in self.x_selector.getRegion())
        self.x_min.blockSignals(True)
        self.x_max.blockSignals(True)
        self.x_min.setValue(x0)
        self.x_max.setValue(x1)
        self.x_min.blockSignals(False)
        self.x_max.blockSignals(False)

    def apply_x_selector(self) -> None:
        if self.current_header is None or not self.x_selector.isVisible():
            return
        x0, x1 = sorted(float(v) for v in self.x_selector.getRegion())
        if x1 <= x0:
            self.status.setText("X轴选择范围无效")
            return
        self.plot.setXRange(x0, x1, padding=0.0)
        self.x_selector.hide()
        self.status.setText(f"已应用X轴选择范围: {x0:.3f} ~ {x1:.3f} m")

    def apply_axis_range(self) -> None:
        if self.current_header is None:
            return
        x0 = float(self.x_min.value())
        x1 = float(self.x_max.value())
        y0 = float(self.y_min.value())
        y1 = float(self.y_max.value())
        if x1 <= x0 or y1 <= y0:
            self.status.setText("坐标范围无效：最大值必须大于最小值")
            return
        self.plot.setXRange(x0, x1, padding=0.0)
        self.plot.setYRange(y0, y1, padding=0.0)
        self.status.setText(f"已应用坐标范围 X: {x0:.3f}~{x1:.3f} m, Y: {y0:.3f}~{y1:.3f} s")

    def show_full_range(self) -> None:
        if self.current_header is None:
            return
        header = self.current_header
        x0 = header.position_start_m
        x1 = header.position_start_m + max(header.distance_span_m, header.point_interval_m)
        y0 = 0.0
        y1 = header.time_span_s
        self.set_range_controls(x0, x1, y0, y1)
        self.plot.setXRange(x0, x1, padding=0.02)
        self.plot.setYRange(y0, y1, padding=0.02)
        self.x_selector.hide()
        self.status.setText("已显示全部范围")

    def update_levels_only(self) -> None:
        limit = float(self.color_limit.value())
        self.image_item.setLevels((-limit, limit))
        self.color_bar.setLevels((-limit, limit))
        if self.current_path and self.current_header and self.current_matrix is not None:
            paths = self.current_paths or [self.current_path]
            self.info.setPlainText(
                format_display_info(
                    paths,
                    self.current_header,
                    self.current_matrix,
                    limit,
                    self.current_display_matrix,
                )
            )
            self.status.setText(f"已更新色条范围: {-limit:.2f} ~ {limit:.2f}")

    def show_spectrum_heatmap(self) -> None:
        if self.current_header is None or self.current_matrix is None:
            QtWidgets.QMessageBox.information(self, "提示", "请先显示一个或多个DAT文件")
            return
        paths = self.current_paths or ([self.current_path] if self.current_path else [])
        if len(paths) <= 1:
            title = "频谱热力图"
        else:
            title = f"频谱热力图 - {len(paths)} 个文件"
        try:
            self.status.setText("正在计算频谱热力图...")
            QtWidgets.QApplication.processEvents()
            window = SpectrumHeatmapWindow(self.current_matrix, self.current_header, title)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "频谱热力图失败", str(exc))
            self.status.setText("频谱热力图计算失败")
            return
        self.analysis_windows.append(window)
        window.show()
        self.status.setText("已打开频谱热力图")

    def on_plot_clicked(self, event: object) -> None:
        if self.current_header is None or self.current_matrix is None:
            return
        if not hasattr(event, "scenePos"):
            return
        pos = self.plot.vb.mapSceneToView(event.scenePos())
        x = float(pos.x())
        y = float(pos.y())
        header = self.current_header
        x_min = header.position_start_m
        x_max = header.position_start_m + max(header.distance_span_m, header.point_interval_m)
        if x < x_min or x > x_max:
            return
        if y < 0 or y > header.time_span_s:
            return

        point_index = int(round((x - header.position_start_m) / max(header.point_interval_m, 1e-12)))
        point_index = max(0, min(header.point_count - 1, point_index))
        distance_m = header.position_start_m + point_index * header.point_interval_m
        self.marker.setValue(distance_m)
        self.marker.show()

        trace = self.current_matrix[point_index, :].copy()
        window = PointAnalysisWindow(trace, header, point_index, distance_m)
        self.analysis_windows.append(window)
        window.show()


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    start_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    viewer = DatViewer(start_path)
    viewer.showMaximized()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
