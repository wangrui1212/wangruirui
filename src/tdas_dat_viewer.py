#!/usr/bin/env python
"""PyQtGraph viewer for Changfei CDAS/TDAS .dat files."""

from __future__ import annotations

import struct
import sys
import gc
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
import re

VENDOR_DIR = Path(__file__).resolve().parent / "_vendor"
if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets
from scipy import signal
from scipy.ndimage import median_filter

try:
    import pyqtgraph.opengl as gl
except Exception:
    gl = None


HEADER_SIZE = 36
HEADER_STRUCT = struct.Struct(">4sHHHfI H B B H f f H H")
GUANGGU_HEADER_FLOATS = 64
GUANGGU_HEADER_SIZE = GUANGGU_HEADER_FLOATS * 4
TDMS_LEAD_IN_SIZE = 28
TDMS_TO_C_META_DATA = 0x02
TDMS_TO_C_RAW_DATA = 0x08
SILIXA_NORMALIZED_MAX = 3.14
DISPLAY_MAX_TIME_BINS = 4096
FFT_WINDOW_OPTIONS = (
    "Rectangular",
    "Hann",
    "Hamming",
    "Blackman",
    "Blackman-Harris",
    "Flat Top",
)
SPECTRUM_MODE_OPTIONS = (
    ("Magnitude (Peak)", "magnitude-peak"),
    ("Magnitude (RMS)", "magnitude-rms"),
    ("Power spectrum", "power"),
    ("Power Spectral Density (PSD)", "psd"),
    ("Amplitude Spectral Density (ASD)", "asd"),
)
DEFAULT_WAVELENGTH_NM = 1550.0
DEFAULT_REFRACTIVE_INDEX = 1.4862
DEFAULT_PHOTOELASTIC_COEFFICIENT = 0.78
DEFAULT_GAUGE_LENGTH_M = 10.0


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
        if self.point_count <= 0:
            return 0.0
        # The heatmap contains one spatial cell for every channel.  Map each
        # cell to one complete point interval, rather than squeezing N cells
        # into only N - 1 intervals.
        return self.point_count * self.point_interval_m

    @property
    def position_end_m(self) -> float:
        return self.position_start_m + self.distance_span_m

    @property
    def time_span_s(self) -> float:
        if self.acquisition_time_s > 0:
            return self.acquisition_time_s
        return self.sample_count / max(1, self.frequency_hz)


@dataclass(frozen=True)
class SpatialPsdResult:
    freqs_hz: np.ndarray
    mean_psd: np.ndarray
    median_psd: np.ndarray
    requested_start_m: float
    requested_end_m: float
    actual_start_m: float
    actual_end_m: float
    selected_channel_count: int
    valid_channel_count: int
    window_samples: int
    hop_samples: int
    window_count: int
    window_seconds: float
    overlap_percent: float
    window_name: str
    sample_rate_hz: int
    warnings: tuple[str, ...] = ()


@dataclass
class SpatialPsdCurveRecord:
    source_key: tuple[str, ...]
    statistic: str
    statistic_label: str
    freqs_hz: np.ndarray
    psd_linear: np.ndarray
    color: str
    source_label: str
    short_label: str
    metadata: dict[str, object]


@dataclass(frozen=True)
class SpectralFlatteningSettings:
    """Parameters for detecting and removing narrow-band interference."""

    first_frequency_hz: float = 200.0
    threshold_db: float = 8.0
    q_factor: float = 50.0
    median_size: int = 9


@dataclass(frozen=True)
class ChannelReplacement:
    """One inclusive, zero-based source-to-target channel mapping."""

    source_path: Path
    target_start: int
    target_end: int
    source_start: int
    source_end: int


@dataclass
class ProcessingResult:
    matrix: np.ndarray
    detected_frequencies_hz: dict[int, np.ndarray]
    source_previews: dict[int, tuple[np.ndarray, DatHeader, np.ndarray]]


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


def _tdms_read_u32(data: bytes, offset: int) -> tuple[int, int]:
    return struct.unpack_from("<I", data, offset)[0], offset + 4


def _tdms_read_u64(data: bytes, offset: int) -> tuple[int, int]:
    return struct.unpack_from("<Q", data, offset)[0], offset + 8


def _tdms_read_i64(data: bytes, offset: int) -> tuple[int, int]:
    return struct.unpack_from("<q", data, offset)[0], offset + 8


def _tdms_read_string(data: bytes, offset: int) -> tuple[str, int]:
    length, offset = _tdms_read_u32(data, offset)
    text = data[offset : offset + length].decode("utf-8", errors="replace")
    return text, offset + length


def _tdms_read_property_value(data: bytes, offset: int, data_type: int) -> tuple[object, int]:
    if data_type == 0x20:
        return _tdms_read_string(data, offset)
    if data_type == 0x44:
        fraction, offset = _tdms_read_u64(data, offset)
        seconds, offset = _tdms_read_i64(data, offset)
        return (seconds, fraction), offset
    formats = {
        1: "<b",
        2: "<h",
        3: "<i",
        4: "<q",
        5: "<B",
        6: "<H",
        7: "<I",
        8: "<Q",
        9: "<f",
        10: "<d",
        0x21: "<?",
    }
    fmt = formats.get(data_type)
    if fmt is None:
        raise ValueError(f"不支持的 TDMS 属性类型: 0x{data_type:X}")
    return struct.unpack_from(fmt, data, offset)[0], offset + struct.calcsize(fmt)


def _parse_tdms_root_and_raw_objects(path: Path) -> tuple[dict[str, object], list[dict[str, object]], int]:
    with path.open("rb") as file:
        lead_in = file.read(TDMS_LEAD_IN_SIZE)
        if len(lead_in) != TDMS_LEAD_IN_SIZE:
            raise ValueError(f"{path.name} TDMS 文件太小，无法读取 lead-in")
        tag, toc_mask, version, next_segment_offset, raw_data_offset = struct.unpack("<4sIIQQ", lead_in)
        if tag != b"TDSm":
            raise ValueError(f"{path.name} 不是 TDMS 文件")
        if (toc_mask & TDMS_TO_C_META_DATA) == 0 or (toc_mask & TDMS_TO_C_RAW_DATA) == 0:
            raise ValueError(f"{path.name} TDMS 缺少元数据或原始数据")
        if next_segment_offset == 0xFFFFFFFFFFFFFFFF:
            raise ValueError(f"{path.name} TDMS 不支持未确定长度的段")
        if next_segment_offset + TDMS_LEAD_IN_SIZE != path.stat().st_size:
            raise ValueError(f"{path.name} TDMS 暂只支持单段 defragmented 文件")
        metadata = file.read(raw_data_offset)

    offset = 0
    object_count, offset = _tdms_read_u32(metadata, offset)
    root_properties: dict[str, object] = {}
    raw_objects: list[dict[str, object]] = []

    for object_index in range(object_count):
        object_path, offset = _tdms_read_string(metadata, offset)
        raw_index_length, offset = _tdms_read_u32(metadata, offset)
        raw_info: dict[str, object] | None = None
        if raw_index_length == 0xFFFFFFFF:
            raw_info = None
        elif raw_index_length == 0:
            raw_info = None
        else:
            raw_start = offset
            # Silixa TDMS files store raw_index_length including this length field.
            payload_size = raw_index_length - 4
            if payload_size < 16:
                raise ValueError(f"{path.name} TDMS raw index 长度无效: {raw_index_length}")
            data_type, offset = _tdms_read_u32(metadata, offset)
            dimension, offset = _tdms_read_u32(metadata, offset)
            value_count, offset = _tdms_read_u64(metadata, offset)
            raw_info = {
                "object_path": object_path,
                "data_type": data_type,
                "dimension": dimension,
                "value_count": value_count,
            }
            offset = raw_start + payload_size

        property_count, offset = _tdms_read_u32(metadata, offset)
        properties: dict[str, object] = {}
        for _ in range(property_count):
            name, offset = _tdms_read_string(metadata, offset)
            data_type, offset = _tdms_read_u32(metadata, offset)
            value, offset = _tdms_read_property_value(metadata, offset, data_type)
            properties[name] = value

        if object_index == 0:
            root_properties = properties
        if raw_info is not None:
            raw_info["properties"] = properties
            raw_objects.append(raw_info)

    return root_properties, raw_objects, TDMS_LEAD_IN_SIZE + raw_data_offset


def parse_silixa_timestamp(path: Path, properties: dict[str, object]) -> str:
    timestamp = properties.get("ISO8601 Timestamp")
    if isinstance(timestamp, str) and timestamp:
        return timestamp
    match = re.search(r"_(\d{8})_(\d{6})\.(\d+)\.tdms$", path.name, re.IGNORECASE)
    if match:
        date_text, time_text, fraction_text = match.groups()
        try:
            base = datetime.strptime(date_text + time_text, "%Y%m%d%H%M%S")
            milliseconds = int(round(float("0." + fraction_text) * 1000.0))
            return base.strftime("%Y-%m-%d %H:%M:%S") + f".{milliseconds:03d}"
        except ValueError:
            pass
    return path.stem


def read_silixa_tdms_header(path: Path) -> DatHeader:
    properties, raw_objects, raw_start = _parse_tdms_root_and_raw_objects(path)
    if not raw_objects:
        raise ValueError(f"{path.name} TDMS 未发现原始数据通道")

    data_types = {int(obj["data_type"]) for obj in raw_objects}
    dimensions = {int(obj["dimension"]) for obj in raw_objects}
    if data_types != {2}:
        raise ValueError(f"{path.name} TDMS 暂只支持 int16 原始数据，当前类型: {sorted(data_types)}")
    if dimensions != {1}:
        raise ValueError(f"{path.name} TDMS 暂只支持一维通道，当前维度: {sorted(dimensions)}")

    frequency_hz = int(round(float(properties.get("SamplingFrequency[Hz]", 0.0))))
    point_interval_m = float(properties.get("SpatialResolution[m]", 0.0))
    position_start_m = float(properties.get("Start Distance (m)", 0.0))
    measure_length_m = float(properties.get("MeasureLength[m]", 0.0))
    point_count = len(raw_objects)
    if frequency_hz <= 0:
        raise ValueError(f"{path.name} TDMS 采样率无效")
    if point_interval_m <= 0:
        raise ValueError(f"{path.name} TDMS 点位间隔无效")
    if point_count <= 0:
        raise ValueError(f"{path.name} TDMS 点位数无效")

    expected_points = int(round(measure_length_m / point_interval_m)) if measure_length_m > 0 else point_count
    if expected_points > 0 and expected_points != point_count:
        raise ValueError(
            f"{path.name} TDMS 点位数与 MeasureLength/SpatialResolution 不一致: "
            f"通道数={point_count}, 元数据推导={expected_points}"
        )

    file_size = path.stat().st_size
    raw_bytes = file_size - raw_start
    if raw_bytes <= 0 or raw_bytes % (2 * point_count) != 0:
        raise ValueError(f"{path.name} TDMS 数据区大小无法整除点位数: raw_bytes={raw_bytes}, 点位数={point_count}")
    sample_count = raw_bytes // (2 * point_count)
    acquisition_time_s = sample_count / frequency_hz

    extra_info: dict[str, object] = {}
    for key in (
        "Acoustic Output",
        "Diagnostic Output",
        "GaugeLength",
        "MeasureLength[m]",
        "SpatialResolution[m]",
        "Start Distance (m)",
        "Stop Distance (m)",
        "Unit Calibration (nm)",
        "Precise Sampling Frequency (Hz)",
        "Time Decimation",
        "ISO8601 Timestamp",
        "iDASVersion",
    ):
        if key in properties:
            extra_info[key] = properties[key]
    extra_info["TDMS raw start"] = raw_start
    extra_info["TDMS raw channels"] = point_count
    extra_info["TDMS raw layout"] = "time-major int16"

    return DatHeader(
        protocol="Silixa TDMS",
        system_id="SILIXA",
        year=0,
        month=0,
        day=0,
        hour=0,
        minute=0,
        second=0.0,
        length=raw_start,
        global_alarm=0,
        acquisition_status=1,
        channel=0,
        frequency_hz=frequency_hz,
        acquisition_time_s=acquisition_time_s,
        point_interval_m=point_interval_m,
        point_count=point_count,
        sample_count=sample_count,
        position_start_m=position_start_m,
        raw_value_note="Silixa int16 raw counts",
        header_size=raw_start,
        bytes_per_sample=2,
        timestamp_text=parse_silixa_timestamp(path, properties),
        extra_info=extra_info,
    )


def read_header(path: Path) -> DatHeader:
    with path.open("rb") as f:
        magic = f.read(4)
    if magic == b"CDAS":
        return read_cdas_header(path)
    if magic == b"TDSm":
        return read_silixa_tdms_header(path)
    return read_guanggu_header(path)


def read_signal_matrix(path: Path, header: DatHeader) -> np.ndarray:
    count = header.point_count * header.sample_count
    if header.protocol == "Silixa TDMS":
        data = np.fromfile(path, dtype="<i2", offset=header.header_size, count=count)
        if data.size != count:
            raise ValueError(f"{path.name} 数据长度不匹配")
        return data.reshape(header.sample_count, header.point_count).T.astype(np.float32, copy=False)

    if header.protocol == "光谷互联":
        data = np.fromfile(path, dtype="<f4", offset=header.header_size, count=count)
        if data.size != count:
            raise ValueError(f"{path.name} 数据长度不匹配")
        return data.reshape(header.sample_count, header.point_count).T.astype(np.float32, copy=False)

    data = np.fromfile(path, dtype=">i2", offset=header.header_size, count=count)
    if data.size != header.point_count * header.sample_count:
        raise ValueError(f"{path.name} 数据长度不匹配")
    return data.reshape(header.point_count, header.sample_count).astype(np.float32) / 100.0


def read_signal_matrix_slice_into(
    path: Path,
    header: DatHeader,
    first_sample: int,
    last_sample: int,
    output: np.ndarray,
    point_first: int = 0,
    point_last: int | None = None,
) -> None:
    """Read only a time slice directly into a float32 ``(point, time)`` buffer.

    Report generation commonly needs a few seconds from a large acquisition
    file.  Reading the whole file and slicing it afterwards needlessly holds a
    second, often multi-gigabyte matrix in RAM.  The source layout differs by
    device: Silixa/Guanggu are time-major, while CDAS is point-major.  This
    helper retains those layout details here and lets callers allocate exactly
    one final output matrix.
    """
    first_sample = int(first_sample)
    last_sample = int(last_sample)
    point_first = int(point_first)
    point_last = int(header.point_count) if point_last is None else int(point_last)
    expected_shape = (point_last - point_first, max(0, last_sample - first_sample))
    if first_sample < 0 or last_sample > header.sample_count or last_sample <= first_sample:
        raise ValueError(f"{path.name} requested sample slice is out of range: {first_sample}:{last_sample}")
    if point_first < 0 or point_last > header.point_count or point_last <= point_first:
        raise ValueError(f"{path.name} requested point slice is out of range: {point_first}:{point_last}")
    if output.dtype != np.float32 or output.shape != expected_shape:
        raise ValueError(f"output shape/dtype must be {expected_shape} float32, got {output.shape} {output.dtype}")

    points = int(header.point_count)
    samples = last_sample - first_sample
    if header.protocol == "Silixa TDMS":
        if point_first != 0 or point_last != points:
            raw = np.memmap(path, dtype="<i2", mode="r", offset=int(header.header_size), shape=(int(header.sample_count), points))
            try:
                np.copyto(output, raw[first_sample:last_sample, point_first:point_last].T, casting="unsafe")
            finally:
                del raw
            return
        offset = int(header.header_size) + first_sample * points * 2
        raw = np.fromfile(path, dtype="<i2", offset=offset, count=samples * points)
        if raw.size != samples * points:
            raise ValueError(f"{path.name} data length does not match requested slice")
        np.copyto(output, raw.reshape(samples, points).T, casting="unsafe")
        return
    if header.protocol == "光谷互联":
        if point_first != 0 or point_last != points:
            raw = np.memmap(path, dtype="<f4", mode="r", offset=int(header.header_size), shape=(int(header.sample_count), points))
            try:
                np.copyto(output, raw[first_sample:last_sample, point_first:point_last].T, casting="unsafe")
            finally:
                del raw
            return
        offset = int(header.header_size) + first_sample * points * 4
        raw = np.fromfile(path, dtype="<f4", offset=offset, count=samples * points)
        if raw.size != samples * points:
            raise ValueError(f"{path.name} data length does not match requested slice")
        np.copyto(output, raw.reshape(samples, points).T, casting="unsafe")
        return

    # CDAS stores the complete time trace for each point consecutively.  A
    # memmap keeps that file-backed instead of materialising all points; the
    # selected columns are converted straight into the caller's output buffer.
    raw = np.memmap(
        path,
        dtype=">i2",
        mode="r",
        offset=int(header.header_size),
        shape=(points, int(header.sample_count)),
    )
    try:
        np.divide(raw[point_first:point_last, first_sample:last_sample], 100.0, out=output, casting="unsafe")
    finally:
        del raw


def read_signal_matrix_indices_into(
    path: Path,
    header: DatHeader,
    sample_indices: np.ndarray,
    output: np.ndarray,
) -> None:
    """Read selected time samples directly into a ``(point, sample)`` buffer.

    This is intentionally used by the report boundary-preview path only.  It
    permits a multi-minute, high-rate file to be represented by uniformly
    distributed samples without ever allocating the complete time matrix.
    """
    indices = np.asarray(sample_indices, dtype=np.int64).reshape(-1)
    expected_shape = (int(header.point_count), int(indices.size))
    if indices.size == 0 or np.any(indices < 0) or np.any(indices >= header.sample_count):
        raise ValueError(f"{path.name} requested sample indices are out of range")
    if output.dtype != np.float32 or output.shape != expected_shape:
        raise ValueError(f"output shape/dtype must be {expected_shape} float32, got {output.shape} {output.dtype}")
    points = int(header.point_count)
    if header.protocol == "Silixa TDMS":
        raw = np.memmap(path, dtype="<i2", mode="r", offset=int(header.header_size), shape=(int(header.sample_count), points))
        try:
            np.copyto(output, raw[indices].T, casting="unsafe")
        finally:
            del raw
        return
    if header.protocol == "光谷互联":
        raw = np.memmap(path, dtype="<f4", mode="r", offset=int(header.header_size), shape=(int(header.sample_count), points))
        try:
            np.copyto(output, raw[indices].T, casting="unsafe")
        finally:
            del raw
        return
    raw = np.memmap(path, dtype=">i2", mode="r", offset=int(header.header_size), shape=(points, int(header.sample_count)))
    try:
        np.divide(raw[:, indices], 100.0, out=output, casting="unsafe")
    finally:
        del raw


def _validate_flattening_settings(settings: SpectralFlatteningSettings, sample_rate_hz: int) -> None:
    if sample_rate_hz <= 0:
        raise ValueError("采样率必须大于 0")
    if not 0.0 <= settings.first_frequency_hz < sample_rate_hz / 2.0:
        raise ValueError("最低检测频率必须小于 Nyquist 频率")
    if settings.threshold_db < 0.0:
        raise ValueError("峰值阈值不能小于 0 dB")
    if settings.q_factor <= 0.0:
        raise ValueError("陷波器 Q 值必须大于 0")
    if settings.median_size < 3 or settings.median_size % 2 == 0:
        raise ValueError("中值窗口必须是不小于 3 的奇数")


def detect_interference_frequencies(
    trace: np.ndarray,
    sample_rate_hz: int,
    settings: SpectralFlatteningSettings = SpectralFlatteningSettings(),
) -> np.ndarray:
    """Return narrow-band peaks above a local median spectrum background.

    The FFT here is diagnostic only.  The returned frequencies are subsequently
    removed from the time-domain signal with zero-phase notch filters.
    """
    _validate_flattening_settings(settings, sample_rate_hz)
    samples = np.asarray(trace, dtype=np.float64).reshape(-1)
    if samples.size < max(16, settings.median_size) or not np.all(np.isfinite(samples)):
        return np.array([], dtype=np.float64)
    samples = samples - np.mean(samples)
    # Windowing prevents a strong, bin-aligned component below the configured
    # range from producing numerical/leakage side lobes that look like dozens
    # of independent narrow-band interferers.
    spectrum = np.fft.rfft(samples * np.hanning(samples.size))
    freqs = np.fft.rfftfreq(samples.size, d=1.0 / sample_rate_hz)
    magnitude_db = 20.0 * np.log10(np.maximum(np.abs(spectrum), 1e-15))
    baseline_db = median_filter(magnitude_db, size=settings.median_size, mode="nearest")
    excess_db = magnitude_db - baseline_db
    peaks, properties = signal.find_peaks(excess_db, height=settings.threshold_db)
    if peaks.size == 0:
        return np.array([], dtype=np.float64)
    valid = (
        (freqs[peaks] >= settings.first_frequency_hz)
        & (freqs[peaks] > 0.0)
        & (freqs[peaks] < sample_rate_hz / 2.0)
        & np.isfinite(properties["peak_heights"])
    )
    return freqs[peaks[valid]].astype(np.float64, copy=False)


def flatten_spectral_trace(
    trace: np.ndarray,
    sample_rate_hz: int,
    settings: SpectralFlatteningSettings = SpectralFlatteningSettings(),
) -> tuple[np.ndarray, np.ndarray]:
    """Apply detected zero-phase notch filters and return data plus peak list."""
    values = np.asarray(trace, dtype=np.float32).reshape(-1)
    peaks = detect_interference_frequencies(values, sample_rate_hz, settings)
    return apply_notch_frequencies(values, sample_rate_hz, peaks, settings.q_factor), peaks


def apply_notch_frequencies(
    trace: np.ndarray,
    sample_rate_hz: int,
    frequencies_hz: np.ndarray,
    q_factor: float,
) -> np.ndarray:
    """Apply an already-designed common notch list to one time trace."""
    values = np.asarray(trace, dtype=np.float32).reshape(-1)
    peaks = np.asarray(frequencies_hz, dtype=np.float64).reshape(-1)
    if peaks.size == 0:
        return values.copy()
    if values.size <= 12:
        return values.copy()

    mean_value = float(np.mean(values))
    filtered = values.astype(np.float64, copy=True) - mean_value
    for frequency_hz in peaks:
        try:
            b, a = signal.iirnotch(float(frequency_hz), q_factor, fs=sample_rate_hz)
            filtered = signal.sosfiltfilt(signal.tf2sos(b, a), filtered)
        except ValueError:
            # Frequencies too close to Nyquist can be rejected by scipy; they
            # are intentionally skipped instead of making the entire job fail.
            continue
    return (filtered + mean_value).astype(np.float32)


def flatten_spectral_channels(
    matrix: np.ndarray,
    sample_rate_hz: int,
    first_channel: int,
    last_channel: int,
    settings: SpectralFlatteningSettings = SpectralFlatteningSettings(),
    reference_channel: int | None = None,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Filter an inclusive zero-based channel interval without changing others."""
    source = np.asarray(matrix, dtype=np.float32)
    if source.ndim != 2:
        raise ValueError("相位数据必须是 通道 x 时间 的二维矩阵")
    if first_channel < 0 or last_channel < first_channel or last_channel >= source.shape[0]:
        raise ValueError("滤波通道范围无效")
    if reference_channel is not None and not 0 <= reference_channel < source.shape[0]:
        raise ValueError("参考通道范围无效")
    result = source.copy()
    detected: dict[int, np.ndarray] = {}
    common_peaks: np.ndarray | None = None
    if reference_channel is not None:
        common_peaks = detect_interference_frequencies(source[reference_channel], sample_rate_hz, settings)
    for channel_index in range(first_channel, last_channel + 1):
        if common_peaks is None:
            result[channel_index], detected[channel_index] = flatten_spectral_trace(source[channel_index], sample_rate_hz, settings)
        else:
            result[channel_index] = apply_notch_frequencies(source[channel_index], sample_rate_hz, common_peaks, settings.q_factor)
            detected[channel_index] = common_peaks.copy()
    return result, detected


def validate_replacement_headers(target: DatHeader, source: DatHeader) -> None:
    if source.protocol != target.protocol:
        raise ValueError(f"来源协议必须与目标相同: {source.protocol} != {target.protocol}")
    target_duration = target.sample_count / max(1, target.frequency_hz)
    source_duration = source.sample_count / max(1, source.frequency_hz)
    tolerance = max(1e-9, 1e-9 * max(target_duration, source_duration))
    if abs(target_duration - source_duration) > tolerance:
        raise ValueError(
            f"来源与目标时长必须一致: 目标 {target_duration:.9g} s, 来源 {source_duration:.9g} s"
        )


def resample_matrix_to_target(
    source_matrix: np.ndarray,
    source_header: DatHeader,
    target_header: DatHeader,
) -> np.ndarray:
    """Band-limited, polyphase resample onto the exact target time grid."""
    validate_replacement_headers(target_header, source_header)
    source = np.asarray(source_matrix, dtype=np.float32)
    if source.ndim != 2 or source.shape[1] != source_header.sample_count:
        raise ValueError("来源数据形状与文件头不一致")
    if source_header.frequency_hz == target_header.frequency_hz:
        result = source.copy()
    else:
        divisor = int(np.gcd(int(source_header.frequency_hz), int(target_header.frequency_hz)))
        up = int(target_header.frequency_hz) // divisor
        down = int(source_header.frequency_hz) // divisor
        result = signal.resample_poly(source, up, down, axis=1, padtype="line").astype(np.float32)
    if result.shape[1] != target_header.sample_count:
        raise ValueError(
            f"重采样后样本数不匹配: {result.shape[1]} != {target_header.sample_count}；"
            "请确认两个文件的时长与采样率"
        )
    return result


def resample_matrix_to_rate(
    source_matrix: np.ndarray,
    source_rate_hz: int,
    target_rate_hz: int,
) -> np.ndarray:
    """Band-limited resampling without imposing a target duration."""
    source = np.asarray(source_matrix, dtype=np.float32)
    if source.ndim != 2 or source_rate_hz <= 0 or target_rate_hz <= 0:
        raise ValueError("重采样数据或采样率无效")
    if source_rate_hz == target_rate_hz:
        return source.copy()
    divisor = int(np.gcd(int(source_rate_hz), int(target_rate_hz)))
    return signal.resample_poly(
        source,
        int(target_rate_hz) // divisor,
        int(source_rate_hz) // divisor,
        axis=1,
        padtype="line",
    ).astype(np.float32)


def apply_channel_replacements(
    target_matrix: np.ndarray,
    target_header: DatHeader,
    filter_start: int,
    filter_end: int,
    settings: SpectralFlatteningSettings,
    replacements: list[ChannelReplacement],
    source_data: dict[Path, tuple[np.ndarray, DatHeader]],
) -> ProcessingResult:
    """Filter target/source segments, resample sources, then replace channels."""
    result, detected = flatten_spectral_channels(
        target_matrix, target_header.frequency_hz, filter_start, filter_end, settings
    )
    previews: dict[int, tuple[np.ndarray, DatHeader, np.ndarray]] = {}
    for index, mapping in enumerate(replacements):
        if mapping.source_path not in source_data:
            raise ValueError(f"未加载来源文件: {mapping.source_path.name}")
        source_matrix, source_header = source_data[mapping.source_path]
        validate_replacement_headers(target_header, source_header)
        if (
            mapping.target_start < 0
            or mapping.target_end < mapping.target_start
            or mapping.target_end >= target_header.point_count
            or mapping.source_start < 0
            or mapping.source_end < mapping.source_start
            or mapping.source_end >= source_header.point_count
        ):
            raise ValueError(f"第 {index + 1} 条替换映射的通道范围无效")
        target_count = mapping.target_end - mapping.target_start + 1
        source_count = mapping.source_end - mapping.source_start + 1
        if target_count != source_count:
            raise ValueError(f"第 {index + 1} 条替换映射的目标与来源通道数必须相等")

        source_block = source_matrix[mapping.source_start : mapping.source_end + 1]
        filtered_block, _ = flatten_spectral_channels(
            source_block,
            source_header.frequency_hz,
            0,
            source_block.shape[0] - 1,
            settings,
        )
        resampled_block = resample_matrix_to_target(
            filtered_block, replace(source_header, point_count=filtered_block.shape[0]),
            replace(target_header, point_count=filtered_block.shape[0]),
        )
        result[mapping.target_start : mapping.target_end + 1] = resampled_block
        previews[index] = (filtered_block, source_header, resampled_block)
    return ProcessingResult(result, detected, previews)


def export_processed_signal(
    target_path: Path,
    target_header: DatHeader,
    matrix: np.ndarray,
    output_path: Path,
) -> None:
    """Write a processed matrix while preserving the target file protocol/header."""
    values = np.asarray(matrix, dtype=np.float32)
    expected_shape = (target_header.point_count, target_header.sample_count)
    if values.shape != expected_shape:
        raise ValueError(f"导出矩阵形状必须为 {expected_shape}，当前为 {values.shape}")
    if output_path.resolve() == target_path.resolve():
        raise ValueError("处理结果不能覆盖原始文件")
    with target_path.open("rb") as source_file:
        header_bytes = source_file.read(target_header.header_size)
    if len(header_bytes) != target_header.header_size:
        raise ValueError("无法读取目标文件头")

    if target_header.protocol == "CDAS":
        scaled = np.rint(values.astype(np.float64) * 100.0)
        dtype = np.dtype(">i2")
        payload = scaled
    elif target_header.protocol == "光谷互联":
        scaled = values
        dtype = np.dtype("<f4")
        payload = scaled.T
    elif target_header.protocol == "Silixa TDMS":
        scaled = np.rint(values.astype(np.float64))
        dtype = np.dtype("<i2")
        payload = scaled.T
    else:
        raise ValueError(f"不支持导出协议: {target_header.protocol}")
    if np.issubdtype(dtype, np.integer):
        limits = np.iinfo(dtype)
        if np.nanmin(scaled) < limits.min or np.nanmax(scaled) > limits.max:
            raise ValueError(f"处理结果超出 {dtype} 的可写范围 [{limits.min}, {limits.max}]")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as output_file:
        output_file.write(header_bytes)
        np.asarray(payload, dtype=dtype).tofile(output_file)

    exported_header = read_header(output_path)
    if (
        exported_header.protocol != target_header.protocol
        or exported_header.frequency_hz != target_header.frequency_hz
        or exported_header.point_count != target_header.point_count
        or exported_header.sample_count != target_header.sample_count
    ):
        raise ValueError("导出文件头校验失败")
    exported_matrix = read_signal_matrix(output_path, exported_header)
    if exported_matrix.shape != values.shape:
        raise ValueError("导出数据形状校验失败")


def normalize_silixa_matrix(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """Scale Silixa raw counts so max absolute value maps to 3.14."""
    if matrix.size == 0:
        return matrix.astype(np.float32, copy=True), 0.0
    max_abs = float(np.nanmax(np.abs(matrix)))
    if not np.isfinite(max_abs) or max_abs <= 0.0:
        return matrix.astype(np.float32, copy=True), 0.0
    normalized = matrix.astype(np.float32, copy=False) * (SILIXA_NORMALIZED_MAX / max_abs)
    return normalized.astype(np.float32, copy=False), max_abs


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


def make_stft_window(name: str, sample_count: int) -> np.ndarray:
    if sample_count < 1:
        raise ValueError("窗函数长度必须至少为 1")
    key = name.strip().lower().replace("_", "-")
    if key == "hamming":
        return np.hamming(sample_count).astype(np.float32)
    if key == "blackman":
        return np.blackman(sample_count).astype(np.float32)
    if key in {"blackman-harris", "blackmanharris"}:
        return signal.windows.blackmanharris(sample_count).astype(np.float32)
    if key in {"flat top", "flat-top", "flattop"}:
        return signal.windows.flattop(sample_count).astype(np.float32)
    if key == "rectangular":
        return np.ones(sample_count, dtype=np.float32)
    return np.hanning(sample_count).astype(np.float32)


@dataclass(frozen=True)
class OneSidedSpectrum:
    """Calibrated one-sided spectra for a real-valued, windowed trace."""

    freqs_hz: np.ndarray
    magnitude_peak: np.ndarray
    magnitude_rms: np.ndarray
    power: np.ndarray
    psd: np.ndarray
    asd: np.ndarray
    sample_count: int


def phase_to_microstrain_per_rad(
    wavelength_nm: float = DEFAULT_WAVELENGTH_NM,
    refractive_index: float = DEFAULT_REFRACTIVE_INDEX,
    photoelastic_coefficient: float = DEFAULT_PHOTOELASTIC_COEFFICIENT,
    gauge_length_m: float = DEFAULT_GAUGE_LENGTH_M,
) -> float:
    """Return the DAS phase-to-strain factor in microstrain per radian.

    ``ε = Δφ λ / (4π n ξ G)``; this helper expresses ε in με.  With the
    defaults it returns 0.010640204 με/rad (10640.204 pε/rad).
    """
    values = (wavelength_nm, refractive_index, photoelastic_coefficient, gauge_length_m)
    if not all(np.isfinite(value) and value > 0.0 for value in values):
        raise ValueError("波长、折射率、光弹性系数和标距必须大于 0")
    return float(wavelength_nm) * 1e-3 / (
        4.0 * np.pi * float(refractive_index) * float(photoelastic_coefficient) * float(gauge_length_m)
    )


def phase_to_picostrain_per_rad(
    wavelength_nm: float = DEFAULT_WAVELENGTH_NM,
    refractive_index: float = DEFAULT_REFRACTIVE_INDEX,
    photoelastic_coefficient: float = DEFAULT_PHOTOELASTIC_COEFFICIENT,
    gauge_length_m: float = DEFAULT_GAUGE_LENGTH_M,
) -> float:
    """Return the phase-to-strain factor in picostrain per radian."""
    return phase_to_microstrain_per_rad(
        wavelength_nm,
        refractive_index,
        photoelastic_coefficient,
        gauge_length_m,
    ) * 1e6


def gauge_length_from_header(
    header: DatHeader,
    default_m: float = DEFAULT_GAUGE_LENGTH_M,
) -> tuple[float, str]:
    """Read gauge length from protocol metadata, with the CDAS spacing convention."""
    for key, raw_value in header.extra_info.items():
        normalized_key = str(key).strip().lower().replace(" ", "")
        if normalized_key not in {"gaugelength", "gaugelength[m]", "标距", "标距(m)"}:
            continue
        try:
            if isinstance(raw_value, str):
                match = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", raw_value)
                value = float(match.group()) if match else float("nan")
            else:
                value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value) and value > 0.0:
            return value, f"头部 {key}"
    # CDAS's current compact header does not expose a separate gauge-length
    # field.  Per the acquisition convention, its point interval is the gauge
    # length used for phase-to-strain conversion.
    if header.protocol == "CDAS" and np.isfinite(header.point_interval_m) and header.point_interval_m > 0.0:
        return float(header.point_interval_m), "CDAS 头部点位间隔"
    return float(default_m), f"头部未提供，默认 {default_m:g} m"


def compute_one_sided_spectrum(
    samples: np.ndarray,
    sample_rate_hz: float,
    window_name: str = "Hann",
) -> OneSidedSpectrum:
    """Return calibrated one-sided Peak/RMS/Power/PSD/ASD spectra.

    The input is de-meaned before applying the selected symmetric window.  Peak
    magnitude is corrected by the window coherent gain.  Power and PSD use the
    window-energy normalization, while ASD is ``sqrt(PSD)``.
    """
    values = np.asarray(samples, dtype=np.float64).reshape(-1)
    sample_count = int(values.size)
    if sample_count < 2:
        raise ValueError("采样点不足，至少需要 2 个点才能计算频谱")
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
        raise ValueError("采样率必须大于 0 Hz")

    values = values - np.mean(values)
    window = make_stft_window(window_name, sample_count).astype(np.float64, copy=False)
    coherent_gain = float(np.mean(window))
    window_power = float(np.sum(window * window))
    if not np.isfinite(coherent_gain) or coherent_gain <= 0.0:
        raise ValueError("窗函数相干增益无效，无法计算幅值谱")
    if not np.isfinite(window_power) or window_power <= 0.0:
        raise ValueError("窗函数能量为 0，无法计算功率谱")

    spectrum = np.fft.rfft(values * window)
    freqs_hz = np.fft.rfftfreq(sample_count, d=1.0 / float(sample_rate_hz))
    magnitude_peak = np.abs(spectrum) / (sample_count * coherent_gain)
    psd = np.square(np.abs(spectrum)) / (float(sample_rate_hz) * window_power)

    # One-sided spectra contain the energy of both positive and negative
    # frequency components.  DC and (for even N) Nyquist have no counterpart.
    if magnitude_peak.size > 1:
        if sample_count % 2 == 0:
            magnitude_peak[1:-1] *= 2.0
            psd[1:-1] *= 2.0
        else:
            magnitude_peak[1:] *= 2.0
            psd[1:] *= 2.0

    magnitude_rms = magnitude_peak.copy()
    if magnitude_rms.size > 1:
        if sample_count % 2 == 0:
            magnitude_rms[1:-1] /= np.sqrt(2.0)
        else:
            magnitude_rms[1:] /= np.sqrt(2.0)

    frequency_resolution_hz = float(sample_rate_hz) / sample_count
    power = psd * frequency_resolution_hz
    asd = np.sqrt(np.maximum(psd, 0.0))
    return OneSidedSpectrum(
        freqs_hz=freqs_hz.astype(np.float32),
        magnitude_peak=magnitude_peak.astype(np.float32),
        magnitude_rms=magnitude_rms.astype(np.float32),
        power=power.astype(np.float32),
        psd=psd.astype(np.float32),
        asd=asd.astype(np.float32),
        sample_count=sample_count,
    )


def make_stft_psd_cache(
    matrix: np.ndarray,
    sample_rate_hz: int,
    window_s: float = 1.0,
    overlap_percent: float = 50.0,
    window_name: str = "Hann",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if matrix.ndim != 2:
        raise ValueError("数据矩阵必须是 点位 x 采样 的二维数组")
    sample_rate_hz = max(1, int(sample_rate_hz))
    window_s = float(window_s)
    overlap_percent = float(overlap_percent)
    if not np.isfinite(window_s) or window_s <= 0.0:
        raise ValueError("窗长必须大于 0 s")
    if not np.isfinite(overlap_percent) or overlap_percent < 0.0 or overlap_percent >= 100.0:
        raise ValueError("重叠率必须满足 0 <= overlap < 100%")

    point_count, sample_count = matrix.shape
    window_samples = int(round(window_s * sample_rate_hz))
    if window_samples < 2:
        raise ValueError("窗长对应采样点数不足，至少需要 2 个采样点")
    if sample_count < window_samples:
        raise ValueError(
            f"数据长度不足以容纳一个 STFT 窗：数据 {sample_count} 点，窗长 {window_samples} 点"
        )
    hop_samples = int(round(window_samples * (1.0 - overlap_percent / 100.0)))
    hop_samples = max(1, hop_samples)

    freqs = np.fft.rfftfreq(window_samples, d=1.0 / sample_rate_hz).astype(np.float32)
    starts = np.arange(0, sample_count - window_samples + 1, hop_samples, dtype=np.int64)
    centers_s = (starts.astype(np.float64) + window_samples / 2.0) / sample_rate_hz
    window_arr = make_stft_window(window_name, window_samples)
    window_power = float(np.sum(window_arr * window_arr))
    if window_power <= 0.0:
        raise ValueError("窗函数能量为 0，无法计算频段能量")

    psd_cache = np.empty((point_count, starts.size, freqs.size), dtype=np.float32)
    for index, start in enumerate(starts):
        stop = int(start) + window_samples
        block = matrix[:, int(start) : stop].astype(np.float32, copy=False)
        block = block - np.mean(block, axis=1, keepdims=True)
        spectrum = np.fft.rfft(block * window_arr[np.newaxis, :], axis=1)
        psd = (np.abs(spectrum) ** 2) / (sample_rate_hz * window_power)
        if psd.shape[1] > 1:
            if window_samples % 2 == 0:
                if psd.shape[1] > 2:
                    psd[:, 1:-1] *= 2.0
            else:
                psd[:, 1:] *= 2.0
        psd_cache[:, index, :] = psd.astype(np.float32)

    return psd_cache, freqs, centers_s.astype(np.float32)


def compute_spatial_segment_psd(
    matrix: np.ndarray,
    header: DatHeader,
    start_m: float,
    end_m: float,
    window_s: float = 2.0,
    overlap_percent: float = 50.0,
    window_name: str = "Hann",
) -> SpatialPsdResult:
    if matrix.ndim != 2:
        raise ValueError("数据矩阵必须是 点位 x 采样 的二维数组")
    sample_rate_hz = int(header.frequency_hz)
    if sample_rate_hz <= 0:
        raise ValueError("采样率必须大于 0 Hz")
    point_count, sample_count = matrix.shape
    if point_count <= 0 or sample_count <= 0:
        raise ValueError("数据矩阵为空，无法计算空间段 PSD")
    if point_count != int(header.point_count):
        raise ValueError(f"数据点位数与文件头不一致：矩阵 {point_count}，文件头 {header.point_count}")

    start_m = float(start_m)
    end_m = float(end_m)
    if not np.isfinite(start_m) or not np.isfinite(end_m):
        raise ValueError("空间范围必须是有限数值")
    if start_m > end_m:
        raise ValueError("空间范围无效：起始位置不能大于结束位置")

    dx = float(header.point_interval_m)
    if point_count > 1 and (not np.isfinite(dx) or dx <= 0.0):
        raise ValueError(f"点位间隔无效：{dx:g} m")
    positions = header.position_start_m + np.arange(point_count, dtype=np.float64) * dx
    tolerance = max(abs(dx), 1.0) * 1e-9
    selected_mask = (positions >= start_m - tolerance) & (positions <= end_m + tolerance)
    selected_indices = np.flatnonzero(selected_mask)
    if selected_indices.size == 0:
        nearest_index = int(np.argmin(np.minimum(np.abs(positions - start_m), np.abs(positions - end_m))))
        raise ValueError(
            f"空间范围 {start_m:.3f}~{end_m:.3f} m 内没有实际通道；"
            f"当前数据范围 {positions[0]:.3f}~{positions[-1]:.3f} m，"
            f"最近通道约 {float(positions[nearest_index]):.3f} m"
        )

    window_s = float(window_s)
    overlap_percent = float(overlap_percent)
    if not np.isfinite(window_s) or window_s <= 0.0:
        raise ValueError("Welch 窗长必须大于 0 s")
    if not np.isfinite(overlap_percent) or overlap_percent < 0.0 or overlap_percent >= 100.0:
        raise ValueError("Welch 重叠率必须满足 0 <= overlap < 100%")
    window_samples = int(round(window_s * sample_rate_hz))
    if window_samples < 2:
        raise ValueError("Welch 窗长对应采样点数不足，至少需要 2 个采样点")
    if window_samples > sample_count:
        data_seconds = sample_count / sample_rate_hz
        raise ValueError(
            f"当前数据时长 {data_seconds:.3f} s，不足以容纳 {window_s:.3f} s 的 Welch 窗；"
            "请缩短窗长或拼接更多文件。"
        )
    hop_samples = max(1, int(round(window_samples * (1.0 - overlap_percent / 100.0))))
    starts = np.arange(0, sample_count - window_samples + 1, hop_samples, dtype=np.int64)
    if starts.size == 0:
        raise ValueError("当前参数下没有可用 Welch 时间窗")

    selected = matrix[selected_indices, :].astype(np.float32, copy=False)
    finite_channel_mask = np.all(np.isfinite(selected), axis=1)
    warnings: list[str] = []
    invalid_count = int(selected.shape[0] - np.count_nonzero(finite_channel_mask))
    if invalid_count:
        warnings.append(f"已排除 {invalid_count} 个包含 NaN/Inf 的通道")
    selected = selected[finite_channel_mask, :]
    valid_positions = positions[selected_indices][finite_channel_mask]
    if selected.shape[0] == 0:
        raise ValueError("空间范围内所有通道都包含 NaN/Inf，无法计算 PSD")

    window_arr = make_stft_window(window_name, window_samples).astype(np.float32, copy=False)
    window_power = float(np.sum(window_arr * window_arr))
    if window_power <= 0.0:
        raise ValueError("窗函数能量为 0，无法计算 PSD")

    freqs = np.fft.rfftfreq(window_samples, d=1.0 / sample_rate_hz).astype(np.float32)
    channel_psd_sum = np.zeros((selected.shape[0], freqs.size), dtype=np.float64)
    scale = sample_rate_hz * window_power
    for start in starts:
        stop = int(start) + window_samples
        block = selected[:, int(start) : stop].astype(np.float32, copy=False)
        block = block - np.mean(block, axis=1, keepdims=True)
        spectrum = np.fft.rfft(block * window_arr[np.newaxis, :], axis=1)
        psd = (np.abs(spectrum) ** 2) / scale
        if psd.shape[1] > 1:
            if window_samples % 2 == 0:
                if psd.shape[1] > 2:
                    psd[:, 1:-1] *= 2.0
            else:
                psd[:, 1:] *= 2.0
        channel_psd_sum += psd

    channel_psd = channel_psd_sum / float(starts.size)
    mean_psd = np.mean(channel_psd, axis=0).astype(np.float32)
    median_psd = np.median(channel_psd, axis=0).astype(np.float32)
    if not np.any(mean_psd > 0.0) and not np.any(median_psd > 0.0):
        warnings.append("空间段 PSD 全为 0，dB 显示会落在 -240 dB 地板")

    return SpatialPsdResult(
        freqs_hz=freqs,
        mean_psd=mean_psd,
        median_psd=median_psd,
        requested_start_m=start_m,
        requested_end_m=end_m,
        actual_start_m=float(valid_positions[0]),
        actual_end_m=float(valid_positions[-1]),
        selected_channel_count=int(selected_indices.size),
        valid_channel_count=int(selected.shape[0]),
        window_samples=window_samples,
        hop_samples=hop_samples,
        window_count=int(starts.size),
        window_seconds=window_s,
        overlap_percent=overlap_percent,
        window_name=window_name,
        sample_rate_hz=sample_rate_hz,
        warnings=tuple(warnings),
    )


def psd_to_db(psd_linear: np.ndarray) -> np.ndarray:
    return (10.0 * np.log10(np.maximum(np.asarray(psd_linear, dtype=np.float32), 1e-24))).astype(np.float32)


def integrate_band_energy(
    psd_cache: np.ndarray,
    freqs: np.ndarray,
    f_low_hz: float,
    f_high_hz: float,
) -> np.ndarray:
    if psd_cache.ndim != 3:
        raise ValueError("PSD缓存必须是 点位 x 时间窗 x 频率点 的三维数组")
    if freqs.size == 0 or psd_cache.shape[2] == 0:
        raise ValueError("PSD缓存为空，无法积分频段能量")
    f_low_hz = float(f_low_hz)
    f_high_hz = float(f_high_hz)
    nyquist_hz = float(freqs[-1])
    if not np.isfinite(f_low_hz) or not np.isfinite(f_high_hz) or f_low_hz < 0.0 or f_low_hz >= f_high_hz:
        raise ValueError("频段范围无效：需要 0 <= 频段下限 < 频段上限")
    if f_high_hz > nyquist_hz:
        raise ValueError(f"频段上限不能超过 Nyquist 频率 {nyquist_hz:.3f} Hz")
    band_mask = (freqs >= f_low_hz) & (freqs <= f_high_hz)
    if not np.any(band_mask):
        raise ValueError("指定频段内没有 FFT 频点，请调整频段或窗长")
    df = float(freqs[1] - freqs[0]) if freqs.size >= 2 else 1.0
    band_energy = np.sum(psd_cache[:, :, band_mask], axis=2) * df
    return (10.0 * np.log10(np.maximum(band_energy, 1e-24))).astype(np.float32)


def make_band_energy_heatmap(
    matrix: np.ndarray,
    sample_rate_hz: int,
    f_low_hz: float,
    f_high_hz: float,
    window_s: float = 1.0,
    overlap_percent: float = 50.0,
    window_name: str = "Hann",
) -> tuple[np.ndarray, np.ndarray]:
    psd_cache, freqs, centers_s = make_stft_psd_cache(
        matrix,
        sample_rate_hz,
        window_s,
        overlap_percent,
        window_name,
    )
    heatmap = integrate_band_energy(psd_cache, freqs, f_low_hz, f_high_hz)
    return heatmap, centers_s.astype(np.float32)


def percentile_color_levels(
    values: np.ndarray,
    low_percentile: float,
    high_percentile: float,
    default_low: float,
    default_high: float,
) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return default_low, default_high
    low = float(np.nanpercentile(finite, low_percentile))
    high = float(np.nanpercentile(finite, high_percentile))
    if not np.isfinite(low):
        low = default_low
    if not np.isfinite(high) or high <= low:
        high = low + 1.0
    return low, high


def robust_adaptive_color_levels(
    values: np.ndarray,
    low_percentile: float = 2.0,
    high_percentile: float = 99.5,
    floor_db: float | None = -200.0,
    default_low: float = -120.0,
    default_high: float = 0.0,
) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return default_low, default_high
    valid = finite
    if floor_db is not None:
        above_floor = finite[finite > float(floor_db)]
        if above_floor.size >= max(16, int(finite.size * 0.01)):
            valid = above_floor
    low = float(np.nanpercentile(valid, low_percentile))
    high = float(np.nanpercentile(valid, high_percentile))
    if not np.isfinite(low):
        low = default_low
    if not np.isfinite(high) or high <= low:
        high = low + 1.0
    return low, high


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
    patterns = ("*.dat", "*.DAT", "*.tdms", "*.TDMS")
    files = {item for pattern in patterns for item in path.rglob(pattern)}
    return sorted(files)


def format_header(
    path: Path,
    header: DatHeader,
    matrix: np.ndarray | None = None,
    color_limit: float | None = None,
    value_mode_note: str | None = None,
) -> str:
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
    if value_mode_note:
        lines.append(value_mode_note)
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
    value_mode_note: str | None = None,
) -> str:
    if len(paths) == 1:
        return format_header(paths[0], header, matrix, color_limit, value_mode_note)

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
    if value_mode_note:
        lines.append(value_mode_note)
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


class SpatialPsdWorkerSignals(QtCore.QObject):
    finished = QtCore.pyqtSignal(object, object)
    failed = QtCore.pyqtSignal(object, str)


class SpatialPsdWorker(QtCore.QRunnable):
    def __init__(
        self,
        payload: dict[str, object],
        matrix: np.ndarray,
        header: DatHeader,
        start_m: float,
        end_m: float,
        window_s: float,
        overlap_percent: float,
        window_name: str,
    ) -> None:
        super().__init__()
        self.signals = SpatialPsdWorkerSignals()
        self.payload = payload
        self.matrix = matrix
        self.header = header
        self.start_m = start_m
        self.end_m = end_m
        self.window_s = window_s
        self.overlap_percent = overlap_percent
        self.window_name = window_name

    @QtCore.pyqtSlot()
    def run(self) -> None:
        try:
            result = compute_spatial_segment_psd(
                self.matrix,
                self.header,
                self.start_m,
                self.end_m,
                self.window_s,
                self.overlap_percent,
                self.window_name,
            )
        except Exception as exc:
            self.signals.failed.emit(self.payload, str(exc))
            return
        self.signals.finished.emit(self.payload, result)


class SpatialPsdComparisonWindow(QtWidgets.QMainWindow):
    COLOR_PALETTE = (
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#9467bd",
        "#ff7f0e",
        "#17becf",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
    )

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("空间段功率谱密度（PSD）对比")
        self.resize(1180, 760)
        self.thread_pool = QtCore.QThreadPool.globalInstance()
        self.current_matrix: np.ndarray | None = None
        self.current_header: DatHeader | None = None
        self.current_payload: dict[str, object] | None = None
        self.records: dict[tuple[tuple[str, ...], str], SpatialPsdCurveRecord] = {}
        self.source_order: list[tuple[str, ...]] = []
        self.source_colors: dict[tuple[str, ...], str] = {}
        self.legend: pg.LegendItem | None = None
        self.active_job_count = 0

        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        self.setCentralWidget(widget)

        self.source_label = QtWidgets.QLabel("当前数据源: 尚未载入")
        self.source_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.source_label)
        self.warning_label = QtWidgets.QLabel("")
        self.warning_label.setStyleSheet("color: #9a3412;")
        self.warning_label.setWordWrap(True)
        layout.addWidget(self.warning_label)

        controls = QtWidgets.QGridLayout()
        layout.addLayout(controls)
        self.start_spin = QtWidgets.QDoubleSpinBox()
        self.end_spin = QtWidgets.QDoubleSpinBox()
        for spin in (self.start_spin, self.end_spin):
            spin.setRange(-1_000_000_000.0, 1_000_000_000.0)
            spin.setDecimals(6)
            spin.setSingleStep(1.0)
            spin.setKeyboardTracking(False)
        self.window_seconds = QtWidgets.QDoubleSpinBox()
        self.window_seconds.setRange(0.0001, 100_000.0)
        self.window_seconds.setDecimals(4)
        self.window_seconds.setSingleStep(0.5)
        self.window_seconds.setValue(2.0)
        self.overlap_percent = QtWidgets.QDoubleSpinBox()
        self.overlap_percent.setRange(0.0, 99.999)
        self.overlap_percent.setDecimals(3)
        self.overlap_percent.setSingleStep(5.0)
        self.overlap_percent.setValue(50.0)
        self.window_combo = QtWidgets.QComboBox()
        self.window_combo.addItems(["Hann", "Hamming", "Blackman", "Rectangular"])
        self.mean_check = QtWidgets.QCheckBox("空间平均 PSD 曲线")
        self.mean_check.setChecked(True)
        self.median_check = QtWidgets.QCheckBox("空间中位数 PSD 曲线")
        self.median_check.setChecked(True)
        self.y_mode_combo = QtWidgets.QComboBox()
        self.y_mode_combo.addItems(["dB", "线性"])
        self.plot_btn = QtWidgets.QPushButton("绘制/更新")
        self.delete_btn = QtWidgets.QPushButton("删除选中")

        controls.addWidget(QtWidgets.QLabel("空间起点(m)"), 0, 0)
        controls.addWidget(self.start_spin, 0, 1)
        controls.addWidget(QtWidgets.QLabel("空间终点(m)"), 0, 2)
        controls.addWidget(self.end_spin, 0, 3)
        controls.addWidget(QtWidgets.QLabel("Welch窗长(s)"), 0, 4)
        controls.addWidget(self.window_seconds, 0, 5)
        controls.addWidget(QtWidgets.QLabel("重叠率(%)"), 1, 0)
        controls.addWidget(self.overlap_percent, 1, 1)
        controls.addWidget(QtWidgets.QLabel("窗函数"), 1, 2)
        controls.addWidget(self.window_combo, 1, 3)
        controls.addWidget(QtWidgets.QLabel("纵轴"), 1, 4)
        controls.addWidget(self.y_mode_combo, 1, 5)
        controls.addWidget(self.mean_check, 2, 0, 1, 2)
        controls.addWidget(self.median_check, 2, 2, 1, 2)
        controls.addWidget(self.plot_btn, 2, 4)
        controls.addWidget(self.delete_btn, 2, 5)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        layout.addWidget(splitter, stretch=1)
        self.plot = pg.PlotWidget(title="空间段 PSD 曲线对比")
        self.plot.setLabel("bottom", "Frequency", units="Hz")
        self.plot.setLabel("left", "PSD", units="dB re 1 数据值²/Hz")
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        splitter.addWidget(self.plot)

        self.table = QtWidgets.QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["#", "数据源", "统计", "空间范围", "通道", "Welch", "数值模式"])
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        splitter.addWidget(self.table)
        splitter.setSizes([520, 180])

        self.status = QtWidgets.QLabel("历史曲线: 0")
        layout.addWidget(self.status)

        self.plot_btn.clicked.connect(self.start_compute)
        self.delete_btn.clicked.connect(self.delete_selected)
        self.y_mode_combo.currentIndexChanged.connect(self.rebuild_plot)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        event.ignore()
        self.hide()

    @staticmethod
    def source_key_from_paths(paths: list[Path]) -> tuple[str, ...]:
        return tuple(str(path.resolve()) for path in paths)

    @staticmethod
    def source_label_from_paths(paths: list[Path]) -> str:
        if not paths:
            return "未命名数据源"
        if len(paths) == 1:
            return paths[0].name
        return f"{paths[0].name} ... {paths[-1].name} ({len(paths)}个文件)"

    def set_source(
        self,
        matrix: np.ndarray,
        header: DatHeader,
        paths: list[Path],
        value_mode_note: str | None,
        silixa_normalized: bool,
        silixa_normalization_scale: float | None,
    ) -> None:
        key = self.source_key_from_paths(paths)
        label = self.source_label_from_paths(paths)
        value_mode = value_mode_note or "当前数据值"
        self.current_matrix = matrix
        self.current_header = header
        self.current_payload = {
            "source_key": key,
            "source_label": label,
            "paths": tuple(str(path.resolve()) for path in paths),
            "protocol": header.protocol,
            "sample_rate_hz": int(header.frequency_hz),
            "value_mode": value_mode,
            "silixa_normalized": bool(silixa_normalized),
            "silixa_normalization_scale": silixa_normalization_scale,
            "computed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        x0 = float(header.position_start_m)
        x1 = float(header.position_end_m)
        if x1 < x0:
            x0, x1 = x1, x0
        if self.start_spin.value() < x0 or self.start_spin.value() > x1:
            self.start_spin.setValue(x0)
        if self.end_spin.value() < x0 or self.end_spin.value() > x1:
            self.end_spin.setValue(x1)
        self.start_spin.setRange(x0, x1)
        self.end_spin.setRange(x0, x1)
        self.source_label.setText(
            f"当前数据源: {label} | {header.protocol} | {header.frequency_hz} Hz | "
            f"{header.position_start_m:.3f}~{header.position_end_m:.3f} m | {header.time_span_s:.3f} s"
        )
        self.update_compare_warning()

    def source_color(self, key: tuple[str, ...]) -> str:
        if key not in self.source_colors:
            self.source_colors[key] = self.COLOR_PALETTE[len(self.source_colors) % len(self.COLOR_PALETTE)]
        return self.source_colors[key]

    def source_number(self, key: tuple[str, ...]) -> int:
        if key not in self.source_order:
            self.source_order.append(key)
        return self.source_order.index(key) + 1

    def start_compute(self) -> None:
        if self.current_matrix is None or self.current_header is None or self.current_payload is None:
            QtWidgets.QMessageBox.information(self, "空间段 PSD 曲线对比", "请先在主窗口显示一个或多个数据文件")
            return
        include_mean = self.mean_check.isChecked()
        include_median = self.median_check.isChecked()
        if not include_mean and not include_median:
            QtWidgets.QMessageBox.information(self, "空间段 PSD 曲线对比", "请至少选择均值或中位数曲线")
            return
        payload = dict(self.current_payload)
        payload["include_mean"] = include_mean
        payload["include_median"] = include_median
        payload["window_seconds"] = float(self.window_seconds.value())
        payload["overlap_percent"] = float(self.overlap_percent.value())
        payload["window_name"] = self.window_combo.currentText()
        matrix_snapshot = np.array(self.current_matrix, dtype=np.float32, copy=True)
        worker = SpatialPsdWorker(
            payload,
            matrix_snapshot,
            self.current_header,
            float(self.start_spin.value()),
            float(self.end_spin.value()),
            float(self.window_seconds.value()),
            float(self.overlap_percent.value()),
            self.window_combo.currentText(),
        )
        worker.signals.finished.connect(self.on_compute_finished)
        worker.signals.failed.connect(self.on_compute_failed)
        self.active_job_count += 1
        self.plot_btn.setEnabled(False)
        self.status.setText("正在计算空间段 PSD ...")
        self.thread_pool.start(worker)

    def on_compute_failed(self, _payload: object, message: str) -> None:
        self.active_job_count = max(0, self.active_job_count - 1)
        self.plot_btn.setEnabled(self.active_job_count == 0)
        QtWidgets.QMessageBox.warning(self, "空间段 PSD 曲线对比", message)
        self.status.setText(f"计算失败: {message}")

    def on_compute_finished(self, payload_obj: object, result_obj: object) -> None:
        self.active_job_count = max(0, self.active_job_count - 1)
        self.plot_btn.setEnabled(self.active_job_count == 0)
        payload = dict(payload_obj)
        result = result_obj
        if not isinstance(result, SpatialPsdResult):
            self.status.setText("计算失败: 返回结果类型无效")
            return
        key = payload["source_key"]
        if not isinstance(key, tuple):
            key = tuple(key)
        color = self.source_color(key)
        source_index = self.source_number(key)
        self.records.pop((key, "mean"), None)
        self.records.pop((key, "median"), None)

        metadata = {
            **payload,
            "actual_start_m": result.actual_start_m,
            "actual_end_m": result.actual_end_m,
            "selected_channel_count": result.selected_channel_count,
            "valid_channel_count": result.valid_channel_count,
            "window_count": result.window_count,
            "window_samples": result.window_samples,
            "hop_samples": result.hop_samples,
            "warnings": result.warnings,
        }
        if payload.get("include_mean"):
            self.records[(key, "mean")] = SpatialPsdCurveRecord(
                source_key=key,
                statistic="mean",
                statistic_label="空间平均",
                freqs_hz=result.freqs_hz,
                psd_linear=result.mean_psd,
                color=color,
                source_label=str(payload["source_label"]),
                short_label=f"#{source_index} {payload['source_label']} | 均值",
                metadata=metadata,
            )
        if payload.get("include_median"):
            self.records[(key, "median")] = SpatialPsdCurveRecord(
                source_key=key,
                statistic="median",
                statistic_label="空间中位数",
                freqs_hz=result.freqs_hz,
                psd_linear=result.median_psd,
                color=color,
                source_label=str(payload["source_label"]),
                short_label=f"#{source_index} {payload['source_label']} | 中位数",
                metadata=metadata,
            )
        self.rebuild_plot()
        self.update_compare_warning()
        warning_text = "；".join(result.warnings)
        suffix = f"；{warning_text}" if warning_text else ""
        self.status.setText(
            f"已更新 #{source_index}：实际空间 {result.actual_start_m:.3f}~{result.actual_end_m:.3f} m，"
            f"{result.valid_channel_count}/{result.selected_channel_count} 通道，{result.window_count} 个 Welch 窗{suffix}"
        )

    def ordered_records(self) -> list[SpatialPsdCurveRecord]:
        order_index = {key: index for index, key in enumerate(self.source_order)}
        stat_index = {"mean": 0, "median": 1}
        return sorted(
            self.records.values(),
            key=lambda item: (order_index.get(item.source_key, 10_000), stat_index.get(item.statistic, 99)),
        )

    def record_y_values(self, record: SpatialPsdCurveRecord) -> np.ndarray:
        if self.y_mode_combo.currentIndex() == 0:
            return psd_to_db(record.psd_linear)
        return record.psd_linear.astype(np.float32, copy=False)

    def rebuild_plot(self) -> None:
        self.plot.clear()
        if self.legend is not None:
            try:
                self.legend.scene().removeItem(self.legend)
            except Exception:
                pass
            self.legend = None
        self.legend = self.plot.addLegend(offset=(10, 10))
        self.plot.setLabel("bottom", "Frequency", units="Hz")
        if self.y_mode_combo.currentIndex() == 0:
            self.plot.setLabel("left", "PSD", units="dB re 1 数据值²/Hz")
        else:
            self.plot.setLabel("left", "PSD", units="数据值²/Hz")
        for record in self.ordered_records():
            style = QtCore.Qt.PenStyle.SolidLine if record.statistic == "mean" else QtCore.Qt.PenStyle.DashLine
            curve = self.plot.plot(
                record.freqs_hz,
                self.record_y_values(record),
                pen=pg.mkPen(record.color, width=2, style=style),
                name=record.short_label,
            )
            curve.setToolTip(self.tooltip_for_record(record))
        self.rebuild_table()

    def tooltip_for_record(self, record: SpatialPsdCurveRecord) -> str:
        meta = record.metadata
        warnings = "；".join(meta.get("warnings", ()) or ())
        parts = [
            f"数据源: {record.source_label}",
            f"统计: {record.statistic_label}",
            f"协议: {meta.get('protocol')}",
            f"采样率: {meta.get('sample_rate_hz')} Hz",
            f"空间: {float(meta.get('actual_start_m', 0.0)):.3f}~{float(meta.get('actual_end_m', 0.0)):.3f} m",
            f"通道: {meta.get('valid_channel_count')}/{meta.get('selected_channel_count')}",
            f"Welch: {meta.get('window_seconds')} s, {meta.get('overlap_percent')}%, {meta.get('window_name')}, {meta.get('window_count')}窗",
            f"数值模式: {meta.get('value_mode')}",
        ]
        if warnings:
            parts.append(f"提示: {warnings}")
        return "\n".join(parts)

    def rebuild_table(self) -> None:
        records = self.ordered_records()
        self.table.setRowCount(len(records))
        for row, record in enumerate(records):
            meta = record.metadata
            source_no = self.source_order.index(record.source_key) + 1 if record.source_key in self.source_order else row + 1
            values = [
                f"#{source_no}",
                record.source_label,
                record.statistic_label,
                f"{float(meta.get('actual_start_m', 0.0)):.3f}~{float(meta.get('actual_end_m', 0.0)):.3f} m",
                f"{meta.get('valid_channel_count')}/{meta.get('selected_channel_count')}",
                f"{meta.get('window_seconds')}s/{meta.get('overlap_percent')}%/{meta.get('window_name')}",
                str(meta.get("value_mode", "")),
            ]
            for col, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                item.setData(QtCore.Qt.ItemDataRole.UserRole, (record.source_key, record.statistic))
                self.table.setItem(row, col, item)
        self.table.resizeColumnsToContents()
        self.status.setText(f"历史曲线: {len(records)}")

    def delete_selected(self) -> None:
        keys: set[tuple[tuple[str, ...], str]] = set()
        for index in self.table.selectionModel().selectedRows():
            item = self.table.item(index.row(), 0)
            if item is not None:
                data = item.data(QtCore.Qt.ItemDataRole.UserRole)
                if isinstance(data, tuple) and len(data) == 2:
                    keys.add(data)
        if not keys:
            self.status.setText("请先在历史表中选择要删除的曲线")
            return
        for key in keys:
            self.records.pop(key, None)
        self.rebuild_plot()
        self.update_compare_warning()
        self.status.setText(f"已删除 {len(keys)} 条历史曲线")

    def update_compare_warning(self) -> None:
        records = list(self.records.values())
        messages: list[str] = []
        if records:
            protocols = {record.metadata.get("protocol") for record in records}
            value_modes = {record.metadata.get("value_mode") for record in records}
            scales = {
                record.metadata.get("silixa_normalization_scale")
                for record in records
                if record.metadata.get("protocol") == "Silixa TDMS"
            }
            if len(protocols) > 1:
                messages.append("当前叠加了不同协议的数据，绝对幅值不一定可直接比较")
            if len(value_modes) > 1:
                messages.append("当前叠加了不同原始/归一化数值模式，绝对幅值不一定可直接比较")
            finite_scales = {float(value) for value in scales if isinstance(value, (int, float)) and np.isfinite(value)}
            if len(finite_scales) > 1:
                messages.append("当前叠加了不同 Silixa 归一化尺度，绝对能量不宜直接比较")
        self.warning_label.setText("；".join(messages))


class PointAnalysisWindow(QtWidgets.QMainWindow):
    def __init__(self, matrix: np.ndarray, header: DatHeader, point_index: int) -> None:
        super().__init__()
        self.resize(1000, 720)
        self.matrix = matrix.astype(np.float32, copy=False)
        self.header = header
        self.point_count = int(self.matrix.shape[0]) if self.matrix.ndim >= 2 else 0
        self.point_index = max(0, min(max(0, self.point_count - 1), int(point_index)))
        self.distance_m = self.index_to_distance(self.point_index)
        self.setWindowTitle(f"点位分析 - #{self.point_index} / {self.distance_m:.3f} m")

        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        self.setCentralWidget(widget)

        self.title_label = QtWidgets.QLabel()
        layout.addWidget(self.title_label)
        self.sample_rate_hz = max(1, header.frequency_hz)
        self.phase_trace = self.matrix[self.point_index, :].astype(np.float32, copy=False)
        self.trace = self.phase_trace
        self.initial_gauge_length_m, self.gauge_length_source = gauge_length_from_header(header)
        self.time_axis = np.arange(self.trace.size, dtype=np.float32) / self.sample_rate_hz
        self.full_time_end_s = self.trace.size / self.sample_rate_hz if self.trace.size else 0.0
        self.fft_start_s = 0.0
        self.fft_end_s = self.full_time_end_s
        self.fft_n = 0
        self.freqs = np.empty(0, dtype=np.float32)
        self.magnitude_peak = np.empty(0, dtype=np.float32)
        self.magnitude_rms = np.empty(0, dtype=np.float32)
        self.power = np.empty(0, dtype=np.float32)
        self.magnitude = np.empty(0, dtype=np.float32)
        self.psd = np.empty(0, dtype=np.float32)
        self.asd = np.empty(0, dtype=np.float32)

        point_controls = QtWidgets.QHBoxLayout()
        layout.addLayout(point_controls)
        self.position_spin = QtWidgets.QDoubleSpinBox()
        x_min = min(header.position_start_m, header.position_end_m)
        x_max = max(header.position_start_m, header.position_end_m)
        if x_max <= x_min:
            x_max = x_min + max(abs(header.point_interval_m), 1.0)
        self.position_spin.setRange(x_min, x_max)
        self.position_spin.setDecimals(6)
        self.position_spin.setSingleStep(max(abs(header.point_interval_m), 0.001))
        self.position_spin.setKeyboardTracking(False)
        apply_position_btn = QtWidgets.QPushButton("应用位置")
        self.position_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider_max = max(0, self.point_count - 1) if header.point_interval_m > 0 else 0
        self.position_slider.setRange(0, slider_max)
        self.position_slider.setSingleStep(1)
        self.position_slider.setPageStep(max(1, slider_max // 20))
        self.position_slider.setEnabled(slider_max > 0)
        self.position_info = QtWidgets.QLabel()
        self.position_range_label = QtWidgets.QLabel(f"{x_min:.3f} m ~ {x_max:.3f} m")
        point_controls.addWidget(QtWidgets.QLabel("位置(m):"))
        point_controls.addWidget(self.position_spin)
        point_controls.addWidget(apply_position_btn)
        point_controls.addWidget(self.position_slider, stretch=1)
        point_controls.addWidget(self.position_info)
        point_controls.addWidget(self.position_range_label)

        controls = QtWidgets.QHBoxLayout()
        layout.addLayout(controls)
        self.fft_mode_combo = QtWidgets.QComboBox()
        for label, key in SPECTRUM_MODE_OPTIONS:
            self.fft_mode_combo.addItem(label, key)
        controls.addWidget(QtWidgets.QLabel("FFT谱类型"))
        controls.addWidget(self.fft_mode_combo)
        self.fft_window_combo = QtWidgets.QComboBox()
        self.fft_window_combo.addItems(list(FFT_WINDOW_OPTIONS))
        self.fft_window_combo.setCurrentText("Hann")
        controls.addWidget(QtWidgets.QLabel("窗函数"))
        controls.addWidget(self.fft_window_combo)
        self.spectrum_scale_combo = QtWidgets.QComboBox()
        self.spectrum_scale_combo.addItems(["对数(dB)", "线性值"])
        controls.addWidget(QtWidgets.QLabel("纵轴刻度"))
        controls.addWidget(self.spectrum_scale_combo)
        controls.addStretch(1)

        conversion_controls = QtWidgets.QHBoxLayout()
        layout.addLayout(conversion_controls)
        self.signal_unit_combo = QtWidgets.QComboBox()
        self.signal_unit_combo.addItem("相位 (rad)", "rad")
        self.signal_unit_combo.addItem("应变 (pε)", "picostrain")
        self.signal_unit_combo.addItem("应变 (με)", "microstrain")
        self.wavelength_nm_spin = QtWidgets.QDoubleSpinBox()
        self.refractive_index_spin = QtWidgets.QDoubleSpinBox()
        self.photoelastic_spin = QtWidgets.QDoubleSpinBox()
        self.gauge_length_spin = QtWidgets.QDoubleSpinBox()
        for spin, minimum, maximum, decimals, step, value in (
            (self.wavelength_nm_spin, 1.0, 10000.0, 3, 1.0, DEFAULT_WAVELENGTH_NM),
            (self.refractive_index_spin, 1.0, 3.0, 6, 0.0001, DEFAULT_REFRACTIVE_INDEX),
            (self.photoelastic_spin, 0.001, 2.0, 6, 0.01, DEFAULT_PHOTOELASTIC_COEFFICIENT),
            (self.gauge_length_spin, 0.001, 100000.0, 6, 0.1, self.initial_gauge_length_m),
        ):
            spin.setRange(minimum, maximum)
            spin.setDecimals(decimals)
            spin.setSingleStep(step)
            spin.setKeyboardTracking(False)
            spin.setValue(value)
        self.gauge_length_source_label = QtWidgets.QLabel(f"标距来源：{self.gauge_length_source}")
        self.conversion_factor_label = QtWidgets.QLabel()
        conversion_controls.addWidget(QtWidgets.QLabel("数据单位"))
        conversion_controls.addWidget(self.signal_unit_combo)
        conversion_controls.addSpacing(8)
        conversion_controls.addWidget(QtWidgets.QLabel("工作波长 λ(nm)"))
        conversion_controls.addWidget(self.wavelength_nm_spin)
        conversion_controls.addWidget(QtWidgets.QLabel("折射率 n"))
        conversion_controls.addWidget(self.refractive_index_spin)
        conversion_controls.addWidget(QtWidgets.QLabel("光弹性系数 ξ"))
        conversion_controls.addWidget(self.photoelastic_spin)
        conversion_controls.addWidget(QtWidgets.QLabel("标距 G(m)"))
        conversion_controls.addWidget(self.gauge_length_spin)
        conversion_controls.addStretch(1)
        conversion_info_controls = QtWidgets.QHBoxLayout()
        layout.addLayout(conversion_info_controls)
        conversion_info_controls.addWidget(self.conversion_factor_label)
        conversion_info_controls.addSpacing(12)
        conversion_info_controls.addWidget(self.gauge_length_source_label)
        conversion_info_controls.addStretch(1)

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

        fft_time_controls = QtWidgets.QHBoxLayout()
        layout.addLayout(fft_time_controls)
        self.fft_time_min = QtWidgets.QDoubleSpinBox()
        self.fft_time_max = QtWidgets.QDoubleSpinBox()
        for spin in (self.fft_time_min, self.fft_time_max):
            spin.setRange(-1_000_000_000.0, 1_000_000_000.0)
            spin.setDecimals(6)
            spin.setSingleStep(0.1)
            spin.setKeyboardTracking(False)
        self.fft_time_min.setValue(0.0)
        self.fft_time_max.setValue(float(self.full_time_end_s))
        fft_apply_btn = QtWidgets.QPushButton("应用FFT时间")
        fft_full_btn = QtWidgets.QPushButton("FFT使用全部")
        fft_time_controls.addWidget(QtWidgets.QLabel("FFT时间(s):"))
        fft_time_controls.addWidget(self.fft_time_min)
        fft_time_controls.addWidget(QtWidgets.QLabel("到"))
        fft_time_controls.addWidget(self.fft_time_max)
        fft_time_controls.addWidget(fft_apply_btn)
        fft_time_controls.addWidget(fft_full_btn)
        fft_time_controls.addStretch(1)

        self.time_plot = pg.PlotWidget(title="相位数据")
        self.time_plot.setLabel("bottom", "Time", units="s")
        self.time_plot.setLabel("left", "Value")
        self.time_curve = self.time_plot.plot(self.time_axis, self.trace, pen=pg.mkPen("#1769aa", width=1))
        self.time_plot.showGrid(x=True, y=True, alpha=0.25)
        layout.addWidget(self.time_plot, stretch=1)
        apply_position_btn.clicked.connect(self.apply_position_input)
        self.position_spin.editingFinished.connect(self.apply_position_input)
        self.position_slider.valueChanged.connect(self.on_position_slider_changed)
        phase_apply_btn.clicked.connect(self.apply_phase_axis_range)
        phase_full_btn.clicked.connect(self.show_full_phase_range)
        fft_apply_btn.clicked.connect(self.apply_fft_time_range)
        fft_full_btn.clicked.connect(self.show_full_fft_time_range)
        for spin in (self.phase_x_min, self.phase_x_max, self.phase_y_min, self.phase_y_max):
            spin.editingFinished.connect(self.apply_phase_axis_range)
        self.time_plot.getViewBox().sigRangeChanged.connect(self.on_phase_view_range_changed)

        self.freq_plot = pg.PlotWidget(title="FFT频率谱")
        self.freq_plot.setLabel("bottom", "Frequency", units="Hz")
        self.freq_curve = self.freq_plot.plot(pen=pg.mkPen("#b21f35", width=1))
        self.freq_plot.showGrid(x=True, y=True, alpha=0.25)
        layout.addWidget(self.freq_plot, stretch=1)
        self.snr_label = QtWidgets.QLabel("峰值谱线SNR: --")
        layout.addWidget(self.snr_label)
        self.fft_mode_combo.currentIndexChanged.connect(self.update_fft_plot)
        self.fft_window_combo.currentIndexChanged.connect(self.on_fft_window_changed)
        self.spectrum_scale_combo.currentIndexChanged.connect(self.update_fft_plot)
        self.signal_unit_combo.currentIndexChanged.connect(self.on_unit_settings_changed)
        for spin in (self.wavelength_nm_spin, self.refractive_index_spin, self.photoelastic_spin):
            spin.valueChanged.connect(self.on_unit_settings_changed)
        self.gauge_length_spin.valueChanged.connect(self.on_gauge_length_changed)
        self.snr_mode_combo.currentIndexChanged.connect(self.update_fft_plot)
        for spin in (
            self.snr_target_freq,
            self.snr_search_half_width,
            self.snr_noise_window,
            self.snr_guard_band,
        ):
            spin.valueChanged.connect(self.update_fft_plot)
        self.update_unit_labels()
        self.sync_point_controls()
        self.show_full_phase_range()
        self.recompute_fft_from_time_range(0.0, self.full_time_end_s, update_controls=True)

    def index_to_distance(self, point_index: int) -> float:
        return self.header.position_start_m + int(point_index) * self.header.point_interval_m

    def distance_to_index(self, distance_m: float) -> int:
        if self.point_count <= 1 or self.header.point_interval_m <= 0:
            return 0
        index = int(round((float(distance_m) - self.header.position_start_m) / self.header.point_interval_m))
        return max(0, min(self.point_count - 1, index))

    def signal_unit(self) -> str:
        return str(self.signal_unit_combo.currentData() or "rad")

    def phase_to_display_factor(self) -> float:
        if self.signal_unit() == "rad":
            return 1.0
        microstrain_per_rad = phase_to_microstrain_per_rad(
            wavelength_nm=float(self.wavelength_nm_spin.value()),
            refractive_index=float(self.refractive_index_spin.value()),
            photoelastic_coefficient=float(self.photoelastic_spin.value()),
            gauge_length_m=float(self.gauge_length_spin.value()),
        )
        return microstrain_per_rad * (1e6 if self.signal_unit() == "picostrain" else 1.0)

    def update_unit_labels(self) -> None:
        try:
            microstrain_per_rad = phase_to_microstrain_per_rad(
                wavelength_nm=float(self.wavelength_nm_spin.value()),
                refractive_index=float(self.refractive_index_spin.value()),
                photoelastic_coefficient=float(self.photoelastic_spin.value()),
                gauge_length_m=float(self.gauge_length_spin.value()),
            )
            self.conversion_factor_label.setText(
                f"K={microstrain_per_rad * 1e6:.3f} pε/rad = {microstrain_per_rad:.8g} με/rad（输入相位 rad）"
            )
        except ValueError as exc:
            self.conversion_factor_label.setText(f"参数无效：{exc}")

        if self.signal_unit() == "picostrain":
            self.time_plot.setTitle("应变数据")
            self.time_plot.setLabel("left", "Strain", units="pε")
        elif self.signal_unit() == "microstrain":
            self.time_plot.setTitle("应变数据")
            self.time_plot.setLabel("left", "Strain", units="με")
        else:
            self.time_plot.setTitle("相位数据")
            self.time_plot.setLabel("left", "Phase", units="rad")

    def refresh_trace_for_unit(self, recompute_fft: bool = True) -> None:
        try:
            factor = self.phase_to_display_factor()
        except ValueError:
            return
        self.trace = (self.phase_trace * factor).astype(np.float32, copy=False)
        self.time_curve.setData(self.time_axis, self.trace)
        self.update_unit_labels()
        if recompute_fft:
            self.recompute_fft_from_time_range(self.fft_start_s, self.fft_end_s, update_controls=False)

    def on_unit_settings_changed(self) -> None:
        self.refresh_trace_for_unit()
        self.show_full_phase_range()

    def on_gauge_length_changed(self) -> None:
        self.gauge_length_source_label.setText("标距来源：手动设置")
        self.on_unit_settings_changed()

    def sync_point_controls(self) -> None:
        self.distance_m = self.index_to_distance(self.point_index)
        self.setWindowTitle(f"点位分析 - #{self.point_index} / {self.distance_m:.3f} m")
        self.title_label.setText(
            f"点位: {self.point_index}    位置: {self.distance_m:.3f} m    "
            f"采样率: {self.header.frequency_hz} Hz    采样数: {self.header.sample_count}"
        )
        self.position_info.setText(f"#{self.point_index} / {self.distance_m:.3f} m")
        self.position_spin.blockSignals(True)
        self.position_spin.setValue(float(self.distance_m))
        self.position_spin.blockSignals(False)
        self.position_slider.blockSignals(True)
        self.position_slider.setValue(int(self.point_index))
        self.position_slider.blockSignals(False)

    def apply_position_input(self) -> None:
        self.set_point_index(self.distance_to_index(float(self.position_spin.value())))

    def on_position_slider_changed(self, point_index: int) -> None:
        self.set_point_index(int(point_index))

    def set_point_index(self, point_index: int) -> None:
        if self.point_count <= 0:
            return
        next_index = max(0, min(self.point_count - 1, int(point_index)))
        if self.header.point_interval_m <= 0:
            next_index = 0
        if next_index == self.point_index:
            self.sync_point_controls()
            return

        self.point_index = next_index
        self.phase_trace = self.matrix[self.point_index, :].astype(np.float32, copy=False)
        self.refresh_trace_for_unit(recompute_fft=False)
        self.sync_point_controls()
        self.show_full_phase_range()
        self.recompute_fft_from_time_range(self.fft_start_s, self.fft_end_s, update_controls=False)

    def set_fft_time_range_controls(self, start_s: float, end_s: float) -> None:
        for spin, value in (
            (self.fft_time_min, start_s),
            (self.fft_time_max, end_s),
        ):
            spin.blockSignals(True)
            spin.setValue(float(value))
            spin.blockSignals(False)

    def apply_fft_time_range(self) -> None:
        self.recompute_fft_from_time_range(
            float(self.fft_time_min.value()),
            float(self.fft_time_max.value()),
            update_controls=True,
        )

    def show_full_fft_time_range(self) -> None:
        self.recompute_fft_from_time_range(0.0, self.full_time_end_s, update_controls=True)

    def on_fft_window_changed(self) -> None:
        """Recompute only when the spectral window changes."""
        self.recompute_fft_from_time_range(
            self.fft_start_s,
            self.fft_end_s,
            update_controls=False,
        )

    def clear_fft_spectrum(self) -> None:
        self.freqs = np.empty(0, dtype=np.float32)
        self.magnitude_peak = np.empty(0, dtype=np.float32)
        self.magnitude_rms = np.empty(0, dtype=np.float32)
        self.power = np.empty(0, dtype=np.float32)
        # Keep the legacy attribute as an alias for code that reads it for SNR.
        self.magnitude = self.magnitude_peak
        self.psd = np.empty(0, dtype=np.float32)
        self.asd = np.empty(0, dtype=np.float32)
        self.fft_n = 0

    def recompute_fft_from_time_range(
        self,
        start_s: float,
        end_s: float,
        update_controls: bool = False,
    ) -> bool:
        if not np.isfinite(start_s) or not np.isfinite(end_s) or end_s <= start_s:
            self.snr_label.setText("峰值谱线SNR: FFT时间范围无效")
            return False

        clipped_start_s = max(0.0, float(start_s))
        clipped_end_s = min(float(end_s), float(self.full_time_end_s))
        if clipped_end_s <= clipped_start_s:
            self.snr_label.setText("峰值谱线SNR: FFT时间范围无效")
            return False

        start_idx = int(np.floor(clipped_start_s * self.sample_rate_hz))
        end_idx = int(np.ceil(clipped_end_s * self.sample_rate_hz))
        start_idx = max(0, min(self.trace.size, start_idx))
        end_idx = max(0, min(self.trace.size, end_idx))
        actual_start_s = start_idx / self.sample_rate_hz
        actual_end_s = end_idx / self.sample_rate_hz
        if update_controls:
            self.set_fft_time_range_controls(actual_start_s, actual_end_s)

        segment = self.trace[start_idx:end_idx]
        self.fft_start_s = actual_start_s
        self.fft_end_s = actual_end_s
        if segment.size < 2:
            self.clear_fft_spectrum()
            self.update_fft_plot("FFT时间范围内采样点不足")
            return False

        n = 1 << (segment.size.bit_length() - 1)
        if n < 2:
            self.clear_fft_spectrum()
            self.update_fft_plot("FFT时间范围内采样点不足")
            return False

        result = compute_one_sided_spectrum(
            segment[:n],
            self.sample_rate_hz,
            self.fft_window_combo.currentText(),
        )
        self.freqs = result.freqs_hz
        self.magnitude_peak = result.magnitude_peak
        self.magnitude_rms = result.magnitude_rms
        self.power = result.power
        self.psd = result.psd
        self.asd = result.asd
        self.magnitude = self.magnitude_peak
        self.fft_n = result.sample_count
        self.update_fft_plot()
        return True

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

    def update_fft_plot(self, message: str | None = None) -> None:
        if message is not None and not isinstance(message, str):
            message = None
        fft_info = f"FFT时间 {self.fft_start_s:.3f}~{self.fft_end_s:.3f} s，n={self.fft_n}"
        if self.freqs.size == 0 or self.magnitude_peak.size == 0 or self.fft_n < 2:
            self.freq_curve.setData([], [])
            self.freq_plot.setTitle(f"频谱 - 无有效数据 - {fft_info}")
            self.snr_label.setText(f"峰值谱线SNR: {message or 'FFT时间范围内采样点不足'}    {fft_info}")
            return

        logarithmic = self.spectrum_scale_combo.currentIndex() == 0
        spectrum_mode = self.fft_mode_combo.currentData()
        amplitude_unit = {
            "picostrain": "pε",
            "microstrain": "με",
        }.get(self.signal_unit(), "rad")
        power_unit = f"{amplitude_unit}²"
        specs: dict[str, tuple[np.ndarray, str, str, str, float]] = {
            "magnitude-peak": (self.magnitude_peak, "Magnitude (Peak)", amplitude_unit, f"dB re 1 {amplitude_unit}", 20.0),
            "magnitude-rms": (self.magnitude_rms, "Magnitude (RMS)", amplitude_unit, f"dB re 1 {amplitude_unit}", 20.0),
            "power": (self.power, "Power spectrum", power_unit, f"dB re 1 {power_unit}", 10.0),
            "psd": (self.psd, "Power Spectral Density (PSD)", f"{power_unit}/Hz", f"dB re 1 {power_unit}/Hz", 10.0),
            "asd": (self.asd, "Amplitude Spectral Density (ASD)", f"{amplitude_unit}/√Hz", f"dB re 1 {amplitude_unit}/√Hz", 20.0),
        }
        values, name, linear_unit, db_unit, db_factor = specs.get(
            str(spectrum_mode), specs["magnitude-peak"]
        )
        if logarithmic:
            floor = 1e-24 if db_factor == 10.0 else 1e-12
            y = db_factor * np.log10(np.maximum(values, floor))
            self.freq_plot.setTitle(f"{name} (dB) - {fft_info}")
            self.freq_plot.setLabel("left", name, units=db_unit)
        else:
            y = values
            self.freq_plot.setTitle(f"{name} (线性) - {fft_info}")
            self.freq_plot.setLabel("left", name, units=linear_unit)
        self.freq_curve.setData(self.freqs, y)
        mode = "target" if self.snr_mode_combo.currentIndex() == 1 else "auto"
        snr = calculate_peak_line_snr(
            self.freqs,
            self.magnitude_peak,
            mode=mode,
            target_freq_hz=float(self.snr_target_freq.value()),
            search_half_width_hz=float(self.snr_search_half_width.value()),
            noise_window_hz=float(self.snr_noise_window.value()),
            guard_band_hz=float(self.snr_guard_band.value()),
            noise_mode=getattr(self, "snr_noise_mode", "local"),
            exclude_band_hz=float(getattr(self, "snr_exclude_band_hz", 5.0)),
        )
        if snr is None:
            self.snr_label.setText(f"峰值谱线SNR: 未找到有效特征峰    {fft_info}")
        else:
            self.snr_label.setText(
                "峰值谱线SNR: "
                f"{snr['snr_db']:.4g} dB    "
                f"{fft_info}    "
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
        self.auto_level_combo = QtWidgets.QComboBox()
        self.auto_level_combo.addItems(["普通分位(P1-P99.5)", "稳健自适应(P2-P99.5)"])
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
        controls.addWidget(QtWidgets.QLabel("自动色阶"))
        controls.addWidget(self.auto_level_combo)
        auto_level_btn = QtWidgets.QPushButton("应用自动色阶")
        controls.addWidget(auto_level_btn)
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
        self.auto_level_combo.currentIndexChanged.connect(lambda _index: self.apply_auto_levels())
        auto_level_btn.clicked.connect(self.apply_auto_levels)
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

        self.profile_status = QtWidgets.QLabel(
            f"FFT使用原始未聚合数据: {matrix.shape[0]} 点位 x {matrix.shape[1]} 采样；"
            f"频谱显示矩阵: {self.display_amplitude.shape[0]} 点位 x {self.display_amplitude.shape[1]} 频率点；"
            f"频率范围: {self.freq_axis_min:.3f} ~ {self.freq_axis_max:.3f} Hz；"
            "点击热力图可绘制该频率的空间分布曲线"
        )
        layout.addWidget(self.profile_status)

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
        else:
            self.current_display = np.square(self.display_amplitude)
            self.plot.setTitle("频谱热力图 - 功率谱(线性)")
        low, high = self.compute_auto_levels()
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

    def compute_auto_levels(self) -> tuple[float, float]:
        if self.current_display.size == 0:
            return 0.0, 1.0
        robust = self.auto_level_combo.currentIndex() == 1
        if self.mode_combo.currentIndex() == 0:
            if robust:
                return robust_adaptive_color_levels(
                    self.current_display,
                    low_percentile=2.0,
                    high_percentile=99.5,
                    floor_db=-200.0,
                    default_low=-120.0,
                    default_high=0.0,
                )
            return percentile_color_levels(self.current_display, 1.0, 99.5, -120.0, 0.0)

        if robust:
            finite = self.current_display[np.isfinite(self.current_display)]
            positive = finite[finite > 0.0]
            values = positive if positive.size >= max(16, int(finite.size * 0.01)) else finite
            return percentile_color_levels(values, 2.0, 99.5, 0.0, 1.0)
        return percentile_color_levels(self.current_display, 0.0, 99.5, 0.0, 1.0)

    def apply_auto_levels(self) -> None:
        low, high = self.compute_auto_levels()
        self.level_min.blockSignals(True)
        self.level_max.blockSignals(True)
        self.level_min.setValue(low)
        self.level_max.setValue(high)
        self.level_min.blockSignals(False)
        self.level_max.blockSignals(False)
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


class BandEnergyHeatmapWindow(QtWidgets.QMainWindow):
    def __init__(self, matrix: np.ndarray, header: DatHeader, title: str) -> None:
        super().__init__()
        self.setWindowTitle(title)
        self.resize(1120, 760)
        self.matrix = matrix.astype(np.float32, copy=False)
        self.header = header
        self.energy_db = np.empty((0, 0), dtype=np.float32)
        self.time_centers_s = np.empty(0, dtype=np.float32)
        self.psd_cache = np.empty((0, 0, 0), dtype=np.float32)
        self.freqs = np.empty(0, dtype=np.float32)
        self.last_stft_params: tuple[float, float, str] | None = None
        self.has_display = False
        self.image_rect = QtCore.QRectF()

        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        self.setCentralWidget(widget)

        controls = QtWidgets.QHBoxLayout()
        layout.addLayout(controls)
        self.f_low = QtWidgets.QDoubleSpinBox()
        self.f_high = QtWidgets.QDoubleSpinBox()
        self.window_seconds = QtWidgets.QDoubleSpinBox()
        self.overlap_percent = QtWidgets.QDoubleSpinBox()
        for spin in (self.f_low, self.f_high):
            spin.setRange(0.0, 1_000_000.0)
            spin.setDecimals(3)
            spin.setSingleStep(1.0)
            spin.setKeyboardTracking(False)
        self.f_low.setValue(45.0)
        self.f_high.setValue(55.0)
        self.window_seconds.setRange(0.001, 1_000_000.0)
        self.window_seconds.setDecimals(3)
        self.window_seconds.setSingleStep(0.1)
        self.window_seconds.setValue(1.0)
        self.window_seconds.setKeyboardTracking(False)
        self.overlap_percent.setRange(0.0, 99.9)
        self.overlap_percent.setDecimals(1)
        self.overlap_percent.setSingleStep(5.0)
        self.overlap_percent.setValue(50.0)
        self.overlap_percent.setKeyboardTracking(False)
        self.window_combo = QtWidgets.QComboBox()
        self.window_combo.addItems(["Hann", "Hamming", "Blackman", "Rectangular"])
        recalc_btn = QtWidgets.QPushButton("重新计算")
        controls.addWidget(QtWidgets.QLabel("频段下限(Hz)"))
        controls.addWidget(self.f_low)
        controls.addWidget(QtWidgets.QLabel("频段上限(Hz)"))
        controls.addWidget(self.f_high)
        controls.addSpacing(12)
        controls.addWidget(QtWidgets.QLabel("窗长(s)"))
        controls.addWidget(self.window_seconds)
        controls.addWidget(QtWidgets.QLabel("重叠率(%)"))
        controls.addWidget(self.overlap_percent)
        controls.addWidget(QtWidgets.QLabel("窗函数"))
        controls.addWidget(self.window_combo)
        controls.addWidget(recalc_btn)
        controls.addStretch(1)

        range_controls = QtWidgets.QHBoxLayout()
        layout.addLayout(range_controls)
        self.x_min = QtWidgets.QDoubleSpinBox()
        self.x_max = QtWidgets.QDoubleSpinBox()
        self.y_min = QtWidgets.QDoubleSpinBox()
        self.y_max = QtWidgets.QDoubleSpinBox()
        self.level_min = QtWidgets.QDoubleSpinBox()
        self.level_max = QtWidgets.QDoubleSpinBox()
        self.auto_level_combo = QtWidgets.QComboBox()
        self.auto_level_combo.addItems(["普通分位(P1-P99.5)", "稳健自适应(P2-P99.5)"])
        for spin in (self.x_min, self.x_max, self.y_min, self.y_max):
            spin.setRange(-1_000_000.0, 1_000_000.0)
            spin.setDecimals(3)
            spin.setSingleStep(1.0)
            spin.setKeyboardTracking(False)
        for spin in (self.level_min, self.level_max):
            spin.setRange(-300.0, 300.0)
            spin.setDecimals(3)
            spin.setSingleStep(1.0)
            spin.setKeyboardTracking(False)
        apply_range_btn = QtWidgets.QPushButton("应用坐标范围")
        full_range_btn = QtWidgets.QPushButton("显示全部")
        auto_level_btn = QtWidgets.QPushButton("自动色阶")
        range_controls.addWidget(QtWidgets.QLabel("X时间(s):"))
        range_controls.addWidget(self.x_min)
        range_controls.addWidget(QtWidgets.QLabel("到"))
        range_controls.addWidget(self.x_max)
        range_controls.addSpacing(12)
        range_controls.addWidget(QtWidgets.QLabel("Y位置(m):"))
        range_controls.addWidget(self.y_min)
        range_controls.addWidget(QtWidgets.QLabel("到"))
        range_controls.addWidget(self.y_max)
        range_controls.addWidget(apply_range_btn)
        range_controls.addWidget(full_range_btn)
        range_controls.addSpacing(12)
        range_controls.addWidget(QtWidgets.QLabel("色阶"))
        range_controls.addWidget(self.level_min)
        range_controls.addWidget(QtWidgets.QLabel("到"))
        range_controls.addWidget(self.level_max)
        range_controls.addWidget(QtWidgets.QLabel("自动"))
        range_controls.addWidget(self.auto_level_combo)
        range_controls.addWidget(auto_level_btn)
        range_controls.addStretch(1)

        graphics = pg.GraphicsLayoutWidget()
        layout.addWidget(graphics, stretch=1)
        self.plot = graphics.addPlot(row=0, col=0)
        self.plot.setLabel("bottom", "Time", units="s")
        self.plot.setLabel("left", "Position", units="m")
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.invertY(True)
        self.image_item = pg.ImageItem()
        self.plot.addItem(self.image_item)
        self.image_item.setLookupTable(pg.colormap.get("turbo").getLookupTable(0.0, 1.0, 256))
        self.color_bar = pg.ColorBarItem(values=(-120.0, 0.0), colorMap=pg.colormap.get("turbo"))
        self.color_bar.setImageItem(self.image_item, insert_in=self.plot)
        self.status = QtWidgets.QLabel("Band energy: --")
        layout.addWidget(self.status)

        recalc_btn.clicked.connect(self.recompute)
        apply_range_btn.clicked.connect(self.apply_axis_range)
        full_range_btn.clicked.connect(self.show_full_range)
        self.auto_level_combo.currentIndexChanged.connect(lambda _index: self.auto_levels())
        auto_level_btn.clicked.connect(self.auto_levels)
        self.level_min.valueChanged.connect(self.update_levels)
        self.level_max.valueChanged.connect(self.update_levels)
        for spin in (self.x_min, self.x_max, self.y_min, self.y_max):
            spin.editingFinished.connect(self.apply_axis_range)
        self.plot.vb.sigRangeChanged.connect(self.on_view_range_changed)

        self.recompute()

    @staticmethod
    def _time_step(centers_s: np.ndarray, fallback_s: float) -> float:
        if centers_s.size >= 2:
            diffs = np.diff(centers_s.astype(np.float64))
            positive = diffs[diffs > 0]
            if positive.size:
                return float(np.median(positive))
        return max(float(fallback_s), 1e-6)

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

    def current_stft_params(self) -> tuple[float, float, str]:
        return (
            float(self.window_seconds.value()),
            float(self.overlap_percent.value()),
            self.window_combo.currentText(),
        )

    @staticmethod
    def _clip_axis_range(
        current: tuple[float, float] | list[float] | None,
        axis_min: float,
        axis_max: float,
    ) -> tuple[float, float]:
        if current is None or axis_max <= axis_min:
            return axis_min, axis_max
        low, high = sorted(float(value) for value in current)
        data_span = axis_max - axis_min
        span = high - low
        if span <= 0.0 or span >= data_span:
            return axis_min, axis_max
        low = max(axis_min, min(low, axis_max - span))
        high = low + span
        if high > axis_max:
            high = axis_max
            low = high - span
        return low, high

    def full_axis_ranges(self) -> tuple[float, float, float, float]:
        x0 = float(self.image_rect.left())
        x1 = float(self.image_rect.right())
        y0 = float(self.image_rect.top())
        y1 = float(self.image_rect.bottom())
        return x0, x1, y0, y1

    def recompute(self) -> None:
        previous_view = self.plot.vb.viewRange() if self.has_display else None
        previous_x_range = tuple(previous_view[0]) if previous_view is not None else None
        previous_y_range = tuple(previous_view[1]) if previous_view is not None else None
        stft_params = self.current_stft_params()
        stft_changed = self.last_stft_params != stft_params or self.psd_cache.size == 0
        try:
            if stft_changed:
                self.psd_cache, self.freqs, self.time_centers_s = make_stft_psd_cache(
                    self.matrix,
                    self.header.frequency_hz,
                    stft_params[0],
                    stft_params[1],
                    stft_params[2],
                )
                self.last_stft_params = stft_params
            energy_db = integrate_band_energy(
                self.psd_cache,
                self.freqs,
                float(self.f_low.value()),
                float(self.f_high.value()),
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "频段能量热力图", str(exc))
            self.status.setText(f"计算失败: {exc}")
            return

        self.energy_db = energy_db
        f_low = float(self.f_low.value())
        f_high = float(self.f_high.value())
        title = f"DAS Band Energy Heatmap: {f_low:g}-{f_high:g} Hz"
        self.setWindowTitle(title)
        self.plot.setTitle(title)

        time_step = self._time_step(self.time_centers_s, float(self.window_seconds.value()))
        x_min = float(self.time_centers_s[0] - time_step / 2.0)
        x_max = float(self.time_centers_s[-1] + time_step / 2.0)
        if x_max <= x_min:
            x_max = x_min + time_step
        y_min = self.header.position_start_m - self.header.point_interval_m / 2.0
        y_max = self.header.position_end_m + self.header.point_interval_m / 2.0
        if y_max <= y_min:
            y_max = y_min + max(abs(self.header.point_interval_m), 1.0)
        self.image_rect = QtCore.QRectF(x_min, y_min, x_max - x_min, y_max - y_min)
        self.image_item.setImage(self.energy_db, autoLevels=False)
        self.image_item.setRect(self.image_rect)
        self.auto_levels()
        x0, x1, y0, y1 = self.full_axis_ranges()
        if not self.has_display:
            next_x0, next_x1 = x0, x1
            next_y0, next_y1 = y0, y1
            self.has_display = True
        else:
            if stft_changed:
                next_x0, next_x1 = x0, x1
            else:
                next_x0, next_x1 = self._clip_axis_range(previous_x_range, x0, x1)
            next_y0, next_y1 = self._clip_axis_range(previous_y_range, y0, y1)
        self.set_range_controls(next_x0, next_x1, next_y0, next_y1)
        self.plot.setXRange(next_x0, next_x1, padding=0.0)
        self.plot.setYRange(next_y0, next_y1, padding=0.0)
        self.status.setText(
            f"Band energy (dB re 1 rad^2): {self.energy_db.shape[0]} 点位 x "
            f"{self.energy_db.shape[1]} 时间窗；STFT {float(self.window_seconds.value()):.3g}s, "
            f"{float(self.overlap_percent.value()):.3g}% overlap, {self.window_combo.currentText()}"
        )

    def auto_levels(self) -> None:
        if self.energy_db.size == 0:
            return
        if self.auto_level_combo.currentIndex() == 1:
            low, high = robust_adaptive_color_levels(
                self.energy_db,
                low_percentile=2.0,
                high_percentile=99.5,
                floor_db=-200.0,
                default_low=-120.0,
                default_high=0.0,
            )
        else:
            low, high = percentile_color_levels(self.energy_db, 1.0, 99.5, -120.0, 0.0)
        self.level_min.blockSignals(True)
        self.level_max.blockSignals(True)
        self.level_min.setValue(low)
        self.level_max.setValue(high)
        self.level_min.blockSignals(False)
        self.level_max.blockSignals(False)
        self.update_levels()

    def update_levels(self) -> None:
        low = float(self.level_min.value())
        high = float(self.level_max.value())
        if high <= low:
            return
        self.image_item.setLevels((low, high))
        self.color_bar.setLevels((low, high))

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
        if self.image_rect.isNull():
            return
        x0 = float(self.image_rect.left())
        x1 = float(self.image_rect.right())
        y0 = float(self.image_rect.top())
        y1 = float(self.image_rect.bottom())
        self.set_range_controls(x0, x1, y0, y1)
        self.plot.setXRange(x0, x1, padding=0.0)
        self.plot.setYRange(y0, y1, padding=0.0)


class Spectrum3DWindow(QtWidgets.QMainWindow):
    def __init__(self, matrix: np.ndarray, header: DatHeader, title: str) -> None:
        super().__init__()
        self.setWindowTitle(title)
        self.resize(1100, 760)
        self.header = header
        self.profile_windows: list[QtWidgets.QMainWindow] = []
        self._surface_update_pending = False
        self._surface_retry_count = 0

        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        self.setCentralWidget(widget)

        controls = QtWidgets.QHBoxLayout()
        layout.addLayout(controls)
        range_controls = QtWidgets.QHBoxLayout()
        layout.addLayout(range_controls)

        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["dB谱(幅值)", "功率谱(线性)"])
        auto_level_btn = QtWidgets.QPushButton("自动色阶")
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
        controls.addWidget(auto_level_btn)
        controls.addSpacing(12)
        self.detail_combo = QtWidgets.QComboBox()
        self.detail_combo.addItem("中 160", 160)
        self.detail_combo.addItem("高 240", 240)
        self.detail_combo.addItem("极高 320", 320)
        controls.addWidget(QtWidgets.QLabel("3D精细度"))
        controls.addWidget(self.detail_combo)
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
        apply_range_btn = QtWidgets.QPushButton("绘制选定范围")
        full_range_btn = QtWidgets.QPushButton("显示全部")
        profile_btn = QtWidgets.QPushButton("频率范围中心曲线")
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
        range_controls.addWidget(profile_btn)
        range_controls.addStretch(1)

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
        self.display_x_min = self.x_center_min
        self.display_x_max = self.x_center_max
        self.display_y_min = 0.0
        self.display_y_max = float(self.freqs[-1]) if self.freqs.size else 0.0
        self.positions_full = (
            header.position_start_m
            + np.arange(header.point_count, dtype=np.float32) * header.point_interval_m
        )
        self.display_positions = self.positions_full
        self.surface_item = None

        if gl is None:
            self.view_3d = None
            missing = QtWidgets.QLabel("3D显示需要安装 PyOpenGL；请重新安装 requirements-build.txt 后再运行。")
            missing.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(missing, stretch=1)
        else:
            self.view_3d = gl.GLViewWidget()
            self.view_3d.setBackgroundColor((245, 247, 250, 255))
            self.view_3d.opts["distance"] = 78
            self.view_3d.opts["elevation"] = 34
            self.view_3d.opts["azimuth"] = -45
            view_row = QtWidgets.QHBoxLayout()
            layout.addLayout(view_row, stretch=1)
            view_row.addWidget(self.view_3d, stretch=1)

            legend_panel = QtWidgets.QWidget()
            legend_layout = QtWidgets.QVBoxLayout(legend_panel)
            legend_layout.setContentsMargins(8, 0, 0, 0)
            self.legend_high = QtWidgets.QLabel("")
            self.legend_image = QtWidgets.QLabel()
            self.legend_image.setFixedSize(24, 220)
            self.legend_low = QtWidgets.QLabel("")
            legend_layout.addWidget(self.legend_high)
            legend_layout.addWidget(self.legend_image)
            legend_layout.addWidget(self.legend_low)
            legend_layout.addStretch(1)
            view_row.addWidget(legend_panel)

            self.axis_value_labels: list[object] = []
            grid = gl.GLGridItem()
            grid.setSize(40, 40, 1)
            grid.setSpacing(4, 4, 1)
            grid.translate(0, 0, -10)
            self.view_3d.addItem(grid)
        self.refresh_timer = QtCore.QTimer(self)
        self.refresh_timer.setSingleShot(True)
        self.refresh_timer.timeout.connect(self.refresh_spectrum_for_view)

        self.mode_combo.currentIndexChanged.connect(lambda _index: self.update_spectrum_mode(reset_levels=True))
        self.level_min.valueChanged.connect(self.update_levels)
        self.level_max.valueChanged.connect(self.update_levels)
        auto_level_btn.clicked.connect(lambda: self.update_spectrum_mode(reset_levels=True))
        self.detail_combo.currentIndexChanged.connect(lambda _index: self.refresh_timer.start(0))
        apply_range_btn.clicked.connect(self.apply_axis_range)
        full_range_btn.clicked.connect(self.show_full_range)
        profile_btn.clicked.connect(self.on_spectrum_clicked)
        for spin in (self.x_min, self.x_max, self.y_min, self.y_max):
            spin.editingFinished.connect(self.apply_axis_range)

        self.profile_status = QtWidgets.QLabel(
            f"FFT source: {matrix.shape[0]} positions x {matrix.shape[1]} samples; "
            f"spectrum matrix: {self.display_amplitude.shape[0]} positions x {self.display_amplitude.shape[1]} frequency bins; "
            f"frequency range: {self.freq_axis_min:.3f} ~ {self.freq_axis_max:.3f} Hz; "
            "3D surface: X=position, Y=frequency, Z=spectrum value"
        )
        layout.addWidget(self.profile_status)

        self.update_spectrum_mode()
        self.show_full_range()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        if self._surface_update_pending or self.surface_item is None:
            self._surface_update_pending = False
            self.refresh_timer.start(0)

    def request_surface_refresh(self) -> None:
        self._surface_update_pending = True
        if self.isVisible():
            self._surface_update_pending = False
            self.refresh_timer.start(50)

    def set_spectrum_data(self, matrix: np.ndarray, header: DatHeader, title: str) -> None:
        self.setWindowTitle(title)
        self.header = header
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
        self.display_x_min = self.x_center_min
        self.display_x_max = self.x_center_max
        self.display_y_min = 0.0
        self.display_y_max = float(self.freqs[-1]) if self.freqs.size else 0.0
        self.positions_full = (
            header.position_start_m
            + np.arange(header.point_count, dtype=np.float32) * header.point_interval_m
        )
        self.display_positions = self.positions_full
        self.profile_status.setText(
            f"FFT source: {matrix.shape[0]} positions x {matrix.shape[1]} samples; "
            f"spectrum matrix: {self.display_amplitude.shape[0]} positions x {self.display_amplitude.shape[1]} frequency bins; "
            f"frequency range: {self.freq_axis_min:.3f} ~ {self.freq_axis_max:.3f} Hz; "
            "3D surface: X=position, Y=frequency, Z=spectrum value"
        )
        self.update_spectrum_mode(reset_levels=True)
        self.show_full_range()
        self.request_surface_refresh()

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

    def apply_axis_range(self) -> None:
        x0 = float(self.x_min.value())
        x1 = float(self.x_max.value())
        y0 = float(self.y_min.value())
        y1 = float(self.y_max.value())
        if x1 <= x0 or y1 <= y0:
            return
        self.refresh_timer.start(0)

    def show_full_range(self) -> None:
        x0 = self.x_center_min
        x1 = self.x_center_max
        y0 = 0.0
        y1 = float(self.freqs[-1]) if self.freqs.size else 0.0
        self.set_range_controls(x0, x1, y0, y1)
        self.refresh_timer.start(0)

    def make_visible_peak_amplitude(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.freqs.size == 0:
            return self.amplitude_full, self.freqs, self.positions_full
        x0, x1 = sorted((float(self.x_min.value()), float(self.x_max.value())))
        y0, y1 = sorted((float(self.y_min.value()), float(self.y_max.value())))
        x0 = max(float(self.positions_full[0]), x0)
        x1 = min(float(self.positions_full[-1]), x1)
        y0 = max(0.0, y0)
        y1 = min(float(self.freqs[-1]), y1)
        if y1 <= y0:
            y0, y1 = 0.0, float(self.freqs[-1])
        if x1 <= x0:
            x0, x1 = self.x_center_min, self.x_center_max
        self.display_x_min = x0
        self.display_x_max = x1
        self.display_y_min = y0
        self.display_y_max = y1

        visible_freqs = np.where((self.freqs >= y0) & (self.freqs <= y1))[0]
        if visible_freqs.size == 0:
            nearest = int(np.argmin(np.abs(self.freqs - (y0 + y1) / 2.0)))
            visible_freqs = np.array([nearest], dtype=np.int64)

        visible_positions = np.where((self.positions_full >= x0) & (self.positions_full <= x1))[0]
        if visible_positions.size == 0:
            nearest = int(np.argmin(np.abs(self.positions_full - (x0 + x1) / 2.0)))
            visible_positions = np.array([nearest], dtype=np.int64)

        x_start = int(visible_positions[0])
        x_stop = int(visible_positions[-1]) + 1
        y_start = int(visible_freqs[0])
        y_stop = int(visible_freqs[-1]) + 1
        amplitude = self.amplitude_full[x_start:x_stop, y_start:y_stop]
        freqs = self.freqs[y_start:y_stop]
        positions = self.positions_full[x_start:x_stop]
        detail_bins = int(self.detail_combo.currentData() or 160)
        if amplitude.shape[0] > detail_bins:
            step = int(np.ceil(amplitude.shape[0] / detail_bins))
            amplitude = amplitude[::step, :]
            positions = positions[::step]
        if amplitude.shape[1] > detail_bins:
            amplitude, freqs = aggregate_positive_matrix_max(amplitude, freqs, detail_bins)
        return amplitude, freqs, positions

    def refresh_spectrum_for_view(self) -> None:
        self.display_amplitude, self.display_freqs, self.display_positions = self.make_visible_peak_amplitude()
        self.freq_step = self._estimate_freq_step(self.display_freqs)
        self.freq_axis_min = float(self.display_freqs[0] - self.freq_step / 2.0) if self.display_freqs.size else 0.0
        self.freq_axis_max = float(self.display_freqs[-1] + self.freq_step / 2.0) if self.display_freqs.size else 0.0
        self.update_spectrum_mode(reset_levels=False)

    def update_spectrum_mode(self, reset_levels: bool = True) -> None:
        if self.mode_combo.currentIndex() == 0:
            self.current_display = 20.0 * np.log10(np.maximum(self.display_amplitude, 1e-12))
            low = -85.0
            high = 0.0
        else:
            self.current_display = np.square(self.display_amplitude)
            low = float(np.nanpercentile(self.current_display, 5.0))
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
        self.update_levels()

    def update_color_legend(self, low: float, high: float) -> None:
        if not hasattr(self, "legend_image"):
            return
        lut = pg.colormap.get("turbo").getLookupTable(0.0, 1.0, 220, alpha=True)
        image = np.repeat(lut[::-1][:, np.newaxis, :], 24, axis=1).astype(np.uint8)
        qimage = QtGui.QImage(
            image.data,
            image.shape[1],
            image.shape[0],
            image.strides[0],
            QtGui.QImage.Format.Format_RGBA8888,
        ).copy()
        self.legend_image.setPixmap(QtGui.QPixmap.fromImage(qimage))
        self.legend_high.setText(f"{high:.3g}")
        self.legend_low.setText(f"{low:.3g}")

    def update_levels(self) -> None:
        low = float(self.level_min.value())
        high = float(self.level_max.value())
        if high <= low:
            return
        self.update_color_legend(low, high)
        if not self.isVisible():
            self._surface_update_pending = True
            return
        self.update_surface(low, high)

    def update_axis_value_labels(self, positions: np.ndarray, freqs: np.ndarray, low: float, high: float) -> None:
        if gl is None or self.view_3d is None or not hasattr(self, "axis_value_labels"):
            return
        for item in self.axis_value_labels:
            self.view_3d.removeItem(item)
        self.axis_value_labels.clear()

        def add_label(text: str, pos: tuple[float, float, float], color: tuple[int, int, int, int]) -> None:
            label = gl.GLTextItem(pos=pos, text=text, color=color)
            self.axis_value_labels.append(label)
            self.view_3d.addItem(label)

        def add_tick_line(points: list[tuple[float, float, float]], color: tuple[float, float, float, float]) -> None:
            line = gl.GLLinePlotItem(pos=np.asarray(points, dtype=np.float32), color=color, width=1.5, antialias=True)
            self.axis_value_labels.append(line)
            self.view_3d.addItem(line)

        x_min = float(getattr(self, "display_x_min", float(positions[0])))
        x_max = float(getattr(self, "display_x_max", float(positions[-1])))
        y_min = float(getattr(self, "display_y_min", float(freqs[0])))
        y_max = float(getattr(self, "display_y_max", float(freqs[-1])))
        x_ticks = np.linspace(x_min, x_max, 6)
        y_ticks = np.linspace(y_min, y_max, 6)
        x_span = max(x_max - x_min, 1e-6)
        y_span = max(y_max - y_min, 1e-6)
        axis_z = -8.5

        for value in x_ticks:
            x_pos = ((value - x_min) / x_span - 0.5) * 40.0
            add_tick_line([(float(x_pos), -20.0, axis_z), (float(x_pos), -21.4, axis_z)], (0.16, 0.35, 0.86, 1.0))
            add_label(f"{value:.1f} m", (float(x_pos) - 1.6, -24.0, axis_z), (40, 90, 220, 255))
        for value in y_ticks:
            y_pos = ((value - y_min) / y_span - 0.5) * 40.0
            add_tick_line([(-20.0, float(y_pos), axis_z), (-21.4, float(y_pos), axis_z)], (0.75, 0.48, 0.0, 1.0))
            add_label(f"{value:.0f} Hz", (-25.5, float(y_pos) - 0.8, axis_z), (190, 130, 0, 255))

    def update_surface(self, low: float, high: float) -> None:
        if gl is None or self.view_3d is None or self.current_display.size == 0:
            return
        if (
            hasattr(self.view_3d, "isValid")
            and not self.view_3d.isValid()
            and getattr(self, "_surface_retry_count", 0) < 1
        ):
            self._surface_retry_count = getattr(self, "_surface_retry_count", 0) + 1
            self._surface_update_pending = True
            QtCore.QTimer.singleShot(100, self.request_surface_refresh)
            return
        self._surface_retry_count = 0
        positions = getattr(self, "display_positions", self.positions_full)
        freqs = self.display_freqs
        if positions.size == 0 or freqs.size == 0:
            return

        values = np.asarray(self.current_display, dtype=np.float32)
        values = np.nan_to_num(values, nan=low, posinf=high, neginf=low)
        values = np.clip(values, low, high)
        z_scale = high - low if high > low else 1.0
        normalized = np.clip((values - low) / z_scale, 0.0, 1.0)
        z = (normalized - 0.5) * 14.0

        x_min = float(getattr(self, "display_x_min", float(positions[0])))
        x_max = float(getattr(self, "display_x_max", float(positions[-1])))
        y_min = float(getattr(self, "display_y_min", float(freqs[0])))
        y_max = float(getattr(self, "display_y_max", float(freqs[-1])))
        x_span = max(x_max - x_min, 1e-6)
        y_span = max(y_max - y_min, 1e-6)
        x = ((positions.astype(np.float32) - x_min) / x_span - 0.5) * 40.0
        y = ((freqs.astype(np.float32) - y_min) / y_span - 0.5) * 40.0
        if x.size < 2:
            x = np.array([-0.5, 0.5], dtype=np.float32)
            z = np.repeat(z, 2, axis=0)
            normalized = np.repeat(normalized, 2, axis=0)
        if y.size < 2:
            y = np.array([-0.5, 0.5], dtype=np.float32)
            z = np.repeat(z, 2, axis=1)
            normalized = np.repeat(normalized, 2, axis=1)

        lut = pg.colormap.get("turbo").getLookupTable(0.0, 1.0, 256, alpha=True)
        color_indices = np.clip((normalized * 255).astype(np.int16), 0, 255)
        colors = (lut[color_indices].astype(np.float32) / 255.0)
        colors[..., 3] = 1.0
        vertexes = np.empty((x.size, y.size, 3), dtype=np.float32)
        vertexes[:, :, 0] = x.reshape(-1, 1)
        vertexes[:, :, 1] = y.reshape(1, -1)
        vertexes[:, :, 2] = z
        grid_indices = np.arange(x.size * y.size, dtype=np.uint32).reshape(x.size, y.size)
        top_left = grid_indices[:-1, :-1].ravel()
        top_right = grid_indices[1:, :-1].ravel()
        bottom_left = grid_indices[:-1, 1:].ravel()
        bottom_right = grid_indices[1:, 1:].ravel()
        faces = np.empty((top_left.size * 2, 3), dtype=np.uint32)
        faces[0::2] = np.column_stack((top_left, top_right, bottom_left))
        faces[1::2] = np.column_stack((top_right, bottom_right, bottom_left))
        mesh_data = gl.MeshData(
            vertexes=vertexes.reshape(-1, 3),
            faces=faces,
            vertexColors=colors.reshape(-1, 4),
        )

        try:
            if self.surface_item is None:
                self.surface_item = gl.GLMeshItem(
                    meshdata=mesh_data,
                    shader=None,
                    smooth=False,
                    computeNormals=False,
                )
                self.surface_item.setGLOptions("opaque")
                self.view_3d.addItem(self.surface_item)
            else:
                self.surface_item.setMeshData(meshdata=mesh_data)
            self.surface_item.update()
        except Exception as exc:
            self.profile_status.setText(f"3D曲面绘制失败: {exc}")
            return
        self.update_axis_value_labels(positions, freqs, low, high)
        self.view_3d.update()
        self.profile_status.setText(
            f"当前绘制范围: 点位位置 {x_min:.3f}~{x_max:.3f} m; "
            f"频率 {y_min:.3f}~{y_max:.3f} Hz; "
            f"显示网格 {positions.size} x {freqs.size}; X=位置, Y=频率, Z=谱值"
        )

    def on_spectrum_clicked(self, event: object) -> None:
        if self.freqs.size == 0 or self.amplitude_full.size == 0:
            return
        clicked_freq_hz = (float(self.y_min.value()) + float(self.y_max.value())) / 2.0
        if clicked_freq_hz < 0.0 or clicked_freq_hz > float(self.freqs[-1]):
            return

        freq_index = int(np.argmin(np.abs(self.freqs - clicked_freq_hz)))
        selected_freq_hz = float(self.freqs[freq_index])
        profile = self.amplitude_full[:, freq_index].astype(np.float32, copy=True)
        positions = self.positions_full
        self.profile_status.setText(
            f"已选择频率: {selected_freq_hz:.3f} Hz "
            f"(当前范围中心 {clicked_freq_hz:.3f} Hz, 最近FFT频点, 空间点数 {positions.size})"
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


class ChannelAxisItem(pg.AxisItem):
    """Top axis that translates distance coordinates into zero-based channels."""

    def __init__(self) -> None:
        super().__init__(orientation="top")
        self.header: DatHeader | None = None

    def set_header(self, header: DatHeader) -> None:
        self.header = header
        self.picture = None
        self.update()

    def tickStrings(self, values: list[float], scale: float, spacing: float) -> list[str]:  # type: ignore[override]
        if self.header is None or self.header.point_interval_m <= 0.0:
            return ["" for _ in values]
        return [str(int(round((value - self.header.position_start_m) / self.header.point_interval_m))) for value in values]


class ClickableImageItem(pg.ImageItem):
    sig_clicked = QtCore.pyqtSignal(object)

    def mouseClickEvent(self, event: object) -> None:
        if hasattr(event, "button") and event.button() != QtCore.Qt.MouseButton.LeftButton:
            if hasattr(event, "ignore"):
                event.ignore()
            return
        self.sig_clicked.emit(event.scenePos() if hasattr(event, "scenePos") else event)
        if hasattr(event, "accept"):
            event.accept()


class PhaseHeatmapPane(QtWidgets.QWidget):
    """Small phase heatmap with channel numbers above and distance below."""

    sig_data_clicked = QtCore.pyqtSignal(object)

    def __init__(self, title: str) -> None:
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.channel_axis = ChannelAxisItem()
        self.plot = pg.PlotWidget(axisItems={"top": self.channel_axis})
        self.plot.setTitle(title)
        self.plot.showAxis("top")
        self.plot.setLabel("top", "Channel (0-based)")
        self.plot.setLabel("bottom", "Distance", units="m")
        self.plot.setLabel("left", "Time", units="s")
        self.plot.invertY(True)
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.image_item = ClickableImageItem()
        self.image_item.setAcceptedMouseButtons(QtCore.Qt.MouseButton.LeftButton)
        self.plot.addItem(self.image_item)
        self.image_item.sig_clicked.connect(self._on_image_clicked)
        self.image_item.setLookupTable(pg.colormap.get("turbo").getLookupTable(0.0, 1.0, 256))
        layout.addWidget(self.plot)

    def _on_image_clicked(self, scene_position: object) -> None:
        if not isinstance(scene_position, QtCore.QPointF):
            return
        self.sig_data_clicked.emit(self.plot.getViewBox().mapSceneToView(scene_position))

    def set_data(self, matrix: np.ndarray, header: DatHeader, levels: tuple[float, float]) -> None:
        display = make_display_matrix_max_abs(np.asarray(matrix, dtype=np.float32))
        self.image_item.setImage(display.T, autoLevels=False)
        self.image_item.setLevels(levels)
        width = max(header.distance_span_m, header.point_interval_m)
        self.image_item.setRect(QtCore.QRectF(header.position_start_m, 0.0, width, header.time_span_s))
        self.channel_axis.set_header(header)
        self.plot.setXRange(header.position_start_m, header.position_start_m + width, padding=0.02)
        self.plot.setYRange(0.0, header.time_span_s, padding=0.02)


@dataclass
class ProcessingTarget:
    path: Path
    header: DatHeader
    original: np.ndarray | None
    working: np.ndarray | None
    detected_frequencies_hz: dict[int, np.ndarray] = field(default_factory=dict)


class ProcessingWorkspaceWindow(QtWidgets.QMainWindow):
    """Separate, composable target filtering and source replacement workspace."""

    def __init__(self, targets: list[tuple[Path, DatHeader, np.ndarray]] | None = None) -> None:
        super().__init__()
        if not targets:
            raise ValueError("至少需要一个目标文件")
        self.setWindowTitle("TDAS 数据处理：滤波与通道替换")
        self.resize(1400, 940)
        self.targets = [ProcessingTarget(path, header, np.asarray(matrix, dtype=np.float32).copy(), np.asarray(matrix, dtype=np.float32).copy()) for path, header, matrix in targets]
        self.active_target_index = 0
        self.source_data: dict[Path, tuple[np.ndarray, DatHeader]] = {}
        self.mapping_sets: dict[Path, list[ChannelReplacement]] = {item.path: [] for item in self.targets}
        self.analysis_windows: list[QtWidgets.QMainWindow] = []
        self._building_mappings = False
        self._build_ui()
        self.load_target(0)

    @property
    def target(self) -> ProcessingTarget:
        return self.targets[self.active_target_index]

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        controls = QtWidgets.QHBoxLayout()
        root.addLayout(controls)
        self.target_combo = QtWidgets.QComboBox()
        for item in self.targets:
            self.target_combo.addItem(item.path.name, item.path)
        self.filter_start = QtWidgets.QSpinBox()
        self.filter_end = QtWidgets.QSpinBox()
        self.fft_channel = QtWidgets.QSpinBox()
        self.min_frequency = QtWidgets.QDoubleSpinBox()
        self.threshold = QtWidgets.QDoubleSpinBox()
        self.q_factor = QtWidgets.QDoubleSpinBox()
        self.min_frequency.setSuffix(" Hz")
        self.threshold.setSuffix(" dB")
        self.q_factor.setPrefix("Q=")
        self.threshold.setRange(0.0, 100.0)
        self.threshold.setValue(8.0)
        self.q_factor.setRange(1.0, 1000.0)
        self.q_factor.setValue(50.0)
        source_btn = QtWidgets.QPushButton("添加替换来源")
        filter_btn = QtWidgets.QPushButton("仅执行频谱平坦化")
        replace_btn = QtWidgets.QPushButton("仅执行通道替换")
        reset_btn = QtWidgets.QPushButton("恢复当前目标原始数据")
        export_btn = QtWidgets.QPushButton("导出全部目标")
        for label, widget in (("目标文件", self.target_combo), ("滤波通道", self.filter_start), ("至", self.filter_end), ("最低频率", self.min_frequency), ("峰值阈值", self.threshold), ("陷波器", self.q_factor)):
            controls.addWidget(QtWidgets.QLabel(label))
            controls.addWidget(widget)
        controls.addWidget(source_btn)
        controls.addWidget(filter_btn)
        controls.addWidget(replace_btn)
        controls.addWidget(reset_btn)
        controls.addWidget(export_btn)

        mapping_controls = QtWidgets.QHBoxLayout()
        root.addLayout(mapping_controls)
        mapping_controls.addWidget(QtWidgets.QLabel("当前目标的替换映射（0 基闭区间，来源文件时长必须与当前目标一致）"))
        add_mapping_btn = QtWidgets.QPushButton("添加映射")
        remove_mapping_btn = QtWidgets.QPushButton("删除选中映射")
        mapping_controls.addWidget(add_mapping_btn)
        mapping_controls.addWidget(remove_mapping_btn)
        mapping_controls.addStretch(1)
        self.mapping_table = QtWidgets.QTableWidget(0, 5)
        self.mapping_table.setHorizontalHeaderLabels(["来源文件", "目标起始", "目标结束", "来源起始", "来源结束"])
        self.mapping_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        for column in range(1, 5):
            self.mapping_table.horizontalHeader().setSectionResizeMode(column, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.mapping_table.setMaximumHeight(155)
        root.addWidget(self.mapping_table)

        heatmaps = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.target_pane = PhaseHeatmapPane("目标相位（处理后直接更新此图）")
        self.source_pane = PhaseHeatmapPane("当前替换来源（按目标时间轴重采样）")
        self.source_pane.plot.setYLink(self.target_pane.plot)
        heatmaps.addWidget(self.target_pane)
        heatmaps.addWidget(self.source_pane)
        heatmaps.setSizes([700, 700])
        root.addWidget(heatmaps, stretch=1)

        fft_layout = QtWidgets.QHBoxLayout()
        root.addLayout(fft_layout)
        fft_layout.addWidget(QtWidgets.QLabel("FFT/点位分析通道"))
        fft_layout.addWidget(self.fft_channel)
        self.fft_plot = pg.PlotWidget(title="当前目标：原始与当前处理结果 FFT")
        self.fft_plot.setLabel("bottom", "Frequency", units="Hz")
        self.fft_plot.setLabel("left", "Amplitude", units="dB")
        self.fft_plot.addLegend()
        self.fft_before = self.fft_plot.plot(pen=pg.mkPen("#777777", width=1), name="原始")
        self.fft_after = self.fft_plot.plot(pen=pg.mkPen("#d62728", width=1), name="当前结果")
        fft_layout.addWidget(self.fft_plot, stretch=1)
        self.peak_list = QtWidgets.QPlainTextEdit()
        self.peak_list.setReadOnly(True)
        self.peak_list.setMinimumWidth(260)
        fft_layout.addWidget(self.peak_list)
        self.status = QtWidgets.QLabel("滤波与替换可分别、重复执行；左图始终显示当前目标处理结果。点击左图任一点打开点位分析。")
        root.addWidget(self.status)

        self.target_combo.currentIndexChanged.connect(self.on_target_changed)
        source_btn.clicked.connect(self.add_sources)
        filter_btn.clicked.connect(self.apply_filter)
        replace_btn.clicked.connect(self.apply_replacements)
        reset_btn.clicked.connect(self.reset_target)
        export_btn.clicked.connect(self.export_all)
        add_mapping_btn.clicked.connect(self.add_mapping)
        remove_mapping_btn.clicked.connect(self.remove_selected_mapping)
        self.mapping_table.itemSelectionChanged.connect(self.update_source_preview)
        self.fft_channel.valueChanged.connect(self.update_fft)
        self.target_pane.plot.scene().sigMouseClicked.connect(self.on_target_heatmap_clicked)

    def settings(self) -> SpectralFlatteningSettings:
        return SpectralFlatteningSettings(float(self.min_frequency.value()), float(self.threshold.value()), float(self.q_factor.value()))

    @staticmethod
    def _levels(*arrays: np.ndarray) -> tuple[float, float]:
        return percentile_color_levels(np.concatenate([np.asarray(value, dtype=np.float32).reshape(-1) for value in arrays]), 1.0, 99.0, -1.0, 1.0)

    def configure_target_controls(self) -> None:
        maximum = self.target.header.point_count - 1
        for spin in (self.filter_start, self.filter_end, self.fft_channel):
            spin.blockSignals(True)
            spin.setRange(0, maximum)
            spin.blockSignals(False)
        self.filter_start.setValue(min(self.filter_start.value(), maximum))
        self.filter_end.setValue(max(self.filter_start.value(), min(self.filter_end.value(), maximum)))
        self.fft_channel.setValue(min(self.fft_channel.value(), maximum))
        self.min_frequency.setRange(0.0, max(0.0, self.target.header.frequency_hz / 2.0 - 1e-6))
        self.min_frequency.setValue(min(max(200.0, self.min_frequency.value()), self.min_frequency.maximum()))

    def refresh_target_view(self) -> None:
        levels = self._levels(self.target.original, self.target.working)
        self.target_pane.set_data(self.target.working, self.target.header, levels)
        self.update_fft()

    def on_target_changed(self, index: int) -> None:
        if index < 0 or index == self.active_target_index:
            return
        self.save_current_mappings()
        self.load_target(index)

    def load_target(self, index: int) -> None:
        if index < 0:
            return
        self.active_target_index = index
        self.target_combo.blockSignals(True)
        self.target_combo.setCurrentIndex(index)
        self.target_combo.blockSignals(False)
        self.configure_target_controls()
        self.rebuild_mapping_table()
        self.refresh_target_view()
        self.update_source_preview()

    def add_sources(self) -> None:
        names, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "选择替换来源文件", str(self.target.path.parent), "Data files (*.dat *.tdms);;All files (*.*)")
        errors: list[str] = []
        added = 0
        for name in names:
            path = Path(name)
            try:
                header = read_header(path)
                if header.protocol != self.target.header.protocol:
                    raise ValueError(f"协议不同: {header.protocol}")
                self.source_data[path] = (read_signal_matrix(path, header), header)
                added += 1
            except Exception as exc:
                errors.append(f"{path.name}: {exc}")
        self.refresh_source_combos()
        self.status.setText(f"已加载 {added} 个来源文件" + ("；" + " | ".join(errors) if errors else ""))

    @staticmethod
    def _range_spin(maximum: int, value: int) -> QtWidgets.QSpinBox:
        spin = QtWidgets.QSpinBox()
        spin.setRange(0, max(0, maximum))
        spin.setValue(max(0, min(value, maximum)))
        return spin

    def refresh_source_combos(self) -> None:
        for row in range(self.mapping_table.rowCount()):
            combo = self.mapping_table.cellWidget(row, 0)
            if isinstance(combo, QtWidgets.QComboBox):
                selected = combo.currentData()
                combo.blockSignals(True)
                combo.clear()
                for path in self.source_data:
                    combo.addItem(path.name, path)
                current_index = combo.findData(selected)
                if current_index >= 0:
                    combo.setCurrentIndex(current_index)
                combo.blockSignals(False)

    def add_mapping(self, mapping: ChannelReplacement | None = None) -> None:
        if not self.source_data:
            QtWidgets.QMessageBox.information(self, "数据处理", "请先添加替换来源文件")
            return
        row = self.mapping_table.rowCount()
        self.mapping_table.insertRow(row)
        combo = QtWidgets.QComboBox()
        for path in self.source_data:
            combo.addItem(path.name, path)
        if mapping is not None:
            combo.setCurrentIndex(max(0, combo.findData(mapping.source_path)))
        combo.currentIndexChanged.connect(lambda _value, r=row: self.update_mapping_limits(r))
        self.mapping_table.setCellWidget(row, 0, combo)
        values = (mapping.target_start, mapping.target_end, mapping.source_start, mapping.source_end) if mapping else (0, self.target.header.point_count - 1, 0, self.target.header.point_count - 1)
        for column, value in zip(range(1, 5), values):
            self.mapping_table.setCellWidget(row, column, self._range_spin(self.target.header.point_count - 1, value))
        self.update_mapping_limits(row)

    def update_mapping_limits(self, row: int) -> None:
        combo = self.mapping_table.cellWidget(row, 0)
        if not isinstance(combo, QtWidgets.QComboBox) or not isinstance(combo.currentData(), Path):
            return
        source_header = self.source_data[combo.currentData()][1]
        for column in (3, 4):
            spin = self.mapping_table.cellWidget(row, column)
            if isinstance(spin, QtWidgets.QSpinBox):
                spin.setMaximum(source_header.point_count - 1)

    def mappings(self) -> list[ChannelReplacement]:
        mappings: list[ChannelReplacement] = []
        for row in range(self.mapping_table.rowCount()):
            combo = self.mapping_table.cellWidget(row, 0)
            spins = [self.mapping_table.cellWidget(row, column) for column in range(1, 5)]
            if not isinstance(combo, QtWidgets.QComboBox) or not isinstance(combo.currentData(), Path) or not all(isinstance(spin, QtWidgets.QSpinBox) for spin in spins):
                raise ValueError(f"第 {row + 1} 条替换映射不完整")
            mappings.append(ChannelReplacement(combo.currentData(), *(int(spin.value()) for spin in spins if isinstance(spin, QtWidgets.QSpinBox))))
        return mappings

    def save_current_mappings(self) -> None:
        if self._building_mappings:
            return
        try:
            self.mapping_sets[self.target.path] = self.mappings()
        except ValueError:
            pass

    def rebuild_mapping_table(self) -> None:
        self._building_mappings = True
        self.mapping_table.setRowCount(0)
        for mapping in self.mapping_sets[self.target.path]:
            if mapping.source_path in self.source_data:
                self.add_mapping(mapping)
        self._building_mappings = False

    def remove_selected_mapping(self) -> None:
        for row in sorted({item.row() for item in self.mapping_table.selectedItems()}, reverse=True):
            self.mapping_table.removeRow(row)
        self.save_current_mappings()

    def apply_filter(self) -> None:
        try:
            working, detected = flatten_spectral_channels(self.target.working, self.target.header.frequency_hz, int(self.filter_start.value()), int(self.filter_end.value()), self.settings())
            self.target.working = working
            self.target.detected_frequencies_hz.update(detected)
            self.refresh_target_view()
            self.status.setText("频谱平坦化已应用到目标；可继续替换或重复滤波。")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "滤波失败", str(exc))

    def apply_replacements(self) -> None:
        try:
            mappings = self.mappings()
            for mapping in mappings:
                source_matrix, source_header = self.source_data[mapping.source_path]
                validate_replacement_headers(self.target.header, source_header)
                target_count = mapping.target_end - mapping.target_start + 1
                source_count = mapping.source_end - mapping.source_start + 1
                if mapping.target_start < 0 or mapping.target_end >= self.target.header.point_count or mapping.source_start < 0 or mapping.source_end >= source_header.point_count or target_count != source_count:
                    raise ValueError("替换通道范围无效，且目标与来源通道数必须一致")
                block = resample_matrix_to_target(source_matrix[mapping.source_start : mapping.source_end + 1], replace(source_header, point_count=source_count), replace(self.target.header, point_count=source_count))
                self.target.working[mapping.target_start : mapping.target_end + 1] = block
            self.save_current_mappings()
            self.refresh_target_view()
            self.update_source_preview()
            self.status.setText("通道替换已应用到目标；未执行任何滤波。")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "替换失败", str(exc))

    def reset_target(self) -> None:
        self.target.working = self.target.original.copy()
        self.target.detected_frequencies_hz.clear()
        self.refresh_target_view()
        self.status.setText("当前目标已恢复原始数据。")

    def update_source_preview(self) -> None:
        row = self.mapping_table.currentRow()
        if row < 0:
            return
        try:
            mapping = self.mappings()[row]
            source_matrix, source_header = self.source_data[mapping.source_path]
            validate_replacement_headers(self.target.header, source_header)
            block = resample_matrix_to_target(source_matrix[mapping.source_start : mapping.source_end + 1], replace(source_header, point_count=mapping.source_end - mapping.source_start + 1), replace(self.target.header, point_count=mapping.source_end - mapping.source_start + 1))
            display_header = replace(source_header, point_count=block.shape[0], sample_count=self.target.header.sample_count, frequency_hz=self.target.header.frequency_hz, acquisition_time_s=self.target.header.time_span_s, position_start_m=source_header.position_start_m + mapping.source_start * source_header.point_interval_m)
            levels = self._levels(self.target.original, self.target.working, block)
            self.target_pane.image_item.setLevels(levels)
            self.source_pane.set_data(block, display_header, levels)
        except Exception as exc:
            self.status.setText(f"来源预览不可用: {exc}")

    def update_fft(self) -> None:
        channel = int(self.fft_channel.value())
        before = self.target.original[channel] - np.mean(self.target.original[channel])
        after = self.target.working[channel] - np.mean(self.target.working[channel])
        frequencies = np.fft.rfftfreq(before.size, 1.0 / self.target.header.frequency_hz)
        self.fft_before.setData(frequencies[1:], 20.0 * np.log10(np.maximum(np.abs(np.fft.rfft(before))[1:], 1e-12)))
        self.fft_after.setData(frequencies[1:], 20.0 * np.log10(np.maximum(np.abs(np.fft.rfft(after))[1:], 1e-12)))
        peaks = self.target.detected_frequencies_hz.get(channel, np.array([], dtype=float))
        self.peak_list.setPlainText("未对该通道执行自动陷波" if peaks.size == 0 else "该通道检测到的陷波频率 (Hz):\n" + ", ".join(f"{value:.4g}" for value in peaks))

    def on_target_heatmap_clicked(self, event: object) -> None:
        if not hasattr(event, "scenePos"):
            return
        point = self.target_pane.plot.getViewBox().mapSceneToView(event.scenePos())
        header = self.target.header
        if point.y() < 0.0 or point.y() > header.time_span_s:
            return
        index = int(round((point.x() - header.position_start_m) / max(header.point_interval_m, 1e-12)))
        if not 0 <= index < header.point_count:
            return
        window = PointAnalysisWindow(self.target.working, header, index)
        self.analysis_windows.append(window)
        window.show()

    def export_all(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "选择处理结果导出目录", str(self.target.path.parent))
        if not directory:
            return
        try:
            output_dir = Path(directory)
            used_names: set[str] = set()
            for index, item in enumerate(self.targets, start=1):
                name = f"{item.path.stem}_processed{item.path.suffix}"
                if name.casefold() in used_names:
                    name = f"{item.path.stem}_{index}_processed{item.path.suffix}"
                used_names.add(name.casefold())
                output = output_dir / name
                export_processed_signal(item.path, item.header, item.working, output)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "导出失败", str(exc))
            return
        self.status.setText(f"已导出并校验 {len(self.targets)} 个目标文件: {directory}")


class SimpleProcessingWorkspaceWindow(QtWidgets.QMainWindow):
    """File-group processor: filter and replacement are deliberately separate."""

    def __init__(self, targets: list[tuple[Path, DatHeader, np.ndarray]] | None = None) -> None:
        super().__init__()
        self.setWindowTitle("TDAS 数据处理：滤波 / 通道替换")
        self.resize(1420, 860)
        self.records = [ProcessingTarget(p, h, np.asarray(m, np.float32).copy(), np.asarray(m, np.float32).copy()) for p, h, m in (targets or [])]
        self.sources: dict[Path, tuple[np.ndarray, DatHeader]] = {}
        self.point_windows: list[PointAnalysisWindow] = []
        self._build()
        if self.records:
            self.target_list.selectAll()
            self.refresh_target()

    def _build(self) -> None:
        root = QtWidgets.QVBoxLayout()
        central = QtWidgets.QWidget(); central.setLayout(root); self.setCentralWidget(central)
        files = QtWidgets.QHBoxLayout(); root.addLayout(files)
        self.target_list = QtWidgets.QListWidget(); self.target_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        for index, item in enumerate(self.records):
            row = QtWidgets.QListWidgetItem(item.path.name); row.setData(QtCore.Qt.ItemDataRole.UserRole, index); self.target_list.addItem(row)
        self.source_list = QtWidgets.QListWidget(); self.source_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        choose_target_folder = QtWidgets.QPushButton("选择目标文件夹")
        add_source = QtWidgets.QPushButton("选择替换文件")
        choose_source_folder = QtWidgets.QPushButton("选择替换文件夹")
        source_box = QtWidgets.QVBoxLayout(); source_box.addWidget(QtWidgets.QLabel("替换来源（可多选，按时间拼接）")); source_box.addWidget(self.source_list); source_box.addWidget(add_source); source_box.addWidget(choose_source_folder)
        target_box = QtWidgets.QVBoxLayout(); target_box.addWidget(QtWidgets.QLabel("目标文件（可多选，按时间拼接）")); target_box.addWidget(self.target_list); target_box.addWidget(choose_target_folder)
        files.addLayout(target_box, 1); files.addLayout(source_box, 1)
        controls = QtWidgets.QHBoxLayout(); root.addLayout(controls)
        self.start = QtWidgets.QSpinBox(); self.end = QtWidgets.QSpinBox(); self.reference = QtWidgets.QSpinBox(); self.minimum = QtWidgets.QDoubleSpinBox(); self.threshold = QtWidgets.QDoubleSpinBox(); self.q = QtWidgets.QDoubleSpinBox()
        self.color_limit = QtWidgets.QDoubleSpinBox()
        self.minimum.setSuffix(" Hz"); self.threshold.setSuffix(" dB"); self.q.setPrefix("Q="); self.threshold.setRange(0, 100); self.threshold.setValue(8); self.q.setRange(1, 1000); self.q.setValue(50)
        self.color_limit.setRange(0.01, 1_000_000.0); self.color_limit.setDecimals(3); self.color_limit.setSingleStep(0.1); self.color_limit.setKeyboardTracking(False); self.color_limit.setValue(1.58)
        filter_button = QtWidgets.QPushButton("仅滤波目标")
        replace_button = QtWidgets.QPushButton("仅替换通道")
        reset_button = QtWidgets.QPushButton("恢复目标原始数据")
        export_button = QtWidgets.QPushButton("导出所选目标")
        for label, widget in (("滤波通道", self.start), ("至", self.end), ("参考通道", self.reference), ("最低频率", self.minimum), ("阈值", self.threshold), ("陷波", self.q)):
            controls.addWidget(QtWidgets.QLabel(label)); controls.addWidget(widget)
        controls.addWidget(filter_button); controls.addWidget(replace_button); controls.addWidget(reset_button); controls.addWidget(export_button)
        batch_controls = QtWidgets.QHBoxLayout(); root.addLayout(batch_controls)
        batch_filter = QtWidgets.QPushButton("批量滤波所有目标并导出")
        batch_replace = QtWidgets.QPushButton("批量替换所有目标并导出")
        batch_controls.addWidget(batch_filter)
        batch_controls.addWidget(batch_replace)
        self.batch_progress = QtWidgets.QProgressBar(); self.batch_progress.setRange(0, 1); self.batch_progress.setValue(0); self.batch_progress.setFormat("等待批量滤波")
        batch_controls.addWidget(self.batch_progress, 1)
        color_controls = QtWidgets.QHBoxLayout(); root.addLayout(color_controls)
        color_controls.addWidget(QtWidgets.QLabel("两幅相位图共用色条范围 ±："))
        color_controls.addWidget(self.color_limit)
        color_controls.addWidget(QtWidgets.QLabel("（直接输入数值，实际显示范围为 -值 ～ +值）"))
        color_controls.addStretch(1)
        maps = QtWidgets.QHBoxLayout(); root.addLayout(maps)
        maps.addWidget(QtWidgets.QLabel("通道映射（0 基、闭区间；仅映射目标和替换来源）"))
        add_map = QtWidgets.QPushButton("添加映射"); del_map = QtWidgets.QPushButton("删除映射"); maps.addWidget(add_map); maps.addWidget(del_map); maps.addStretch()
        self.table = QtWidgets.QTableWidget(0, 4); self.table.setHorizontalHeaderLabels(["目标起始", "目标结束", "来源起始", "来源结束"]); self.table.setMaximumHeight(130)
        root.addWidget(self.table)
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.target_pane = PhaseHeatmapPane("目标相位（处理后直接更新）")
        self.source_pane = PhaseHeatmapPane("替换来源相位（按时间拼接/重采样）")
        self.shared_color_bar = pg.ColorBarItem(values=self.shared_levels(), colorMap=pg.colormap.get("turbo"))
        self.shared_color_bar.setImageItem(
            self.target_pane.image_item,
            insert_in=self.target_pane.plot.getPlotItem(),
        )
        split.addWidget(self.target_pane); split.addWidget(self.source_pane); root.addWidget(split, 1)
        self.status = QtWidgets.QLabel("左图点击任意点打开点位分析。来源比目标短时，只替换目标开头对应时间。")
        root.addWidget(self.status)
        add_source.clicked.connect(self.add_sources); choose_target_folder.clicked.connect(self.choose_target_folder); choose_source_folder.clicked.connect(self.choose_source_folder); add_map.clicked.connect(self.add_mapping); del_map.clicked.connect(self.delete_mapping)
        filter_button.clicked.connect(self.apply_filter); replace_button.clicked.connect(self.apply_replacement); reset_button.clicked.connect(self.reset); export_button.clicked.connect(self.export)
        batch_filter.clicked.connect(self.batch_filter_all_targets)
        batch_replace.clicked.connect(self.batch_replace_all_targets)
        self.target_list.itemSelectionChanged.connect(self.refresh_target); self.source_list.itemSelectionChanged.connect(self.refresh_source)
        self.color_limit.valueChanged.connect(self.apply_shared_color_levels)
        self.target_pane.plot.scene().sigMouseClicked.connect(self.open_point_analysis)

    def selected_records(self) -> list[ProcessingTarget]:
        return [self.records[item.data(QtCore.Qt.ItemDataRole.UserRole)] for item in self.target_list.selectedItems()]

    def choose_target_folder(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "选择目标数据文件夹", "")
        if not folder:
            return
        self.records.clear(); self.target_list.clear()
        self.status.setText("正在扫描目标文件头...")
        QtWidgets.QApplication.processEvents()
        errors: list[str] = []
        paths = find_dat_files(Path(folder))
        for index, path in enumerate(paths, start=1):
            try:
                header = read_header(path)
                # Do not read the potentially multi-gigabyte payload here.
                # It is loaded only when the user selects this target.
                self.records.append(ProcessingTarget(path, header, None, None))
                row = QtWidgets.QListWidgetItem(path.name); row.setData(QtCore.Qt.ItemDataRole.UserRole, len(self.records) - 1); self.target_list.addItem(row)
            except Exception as exc: errors.append(f"{path.name}: {exc}")
            if index % 25 == 0:
                self.status.setText(f"正在扫描目标文件头: {index}/{len(paths)}")
                QtWidgets.QApplication.processEvents()
        self.status.setText(f"已扫描目标文件夹：{len(self.records)} 个文件；请选择一个或多个文件后显示/处理" + (f"；跳过 {len(errors)} 个无效文件" if errors else ""))

    def choose_source_folder(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "选择替换来源文件夹", "")
        if not folder:
            return
        self.add_source_paths(find_dat_files(Path(folder)))

    def target_data(self) -> tuple[np.ndarray, DatHeader, list[ProcessingTarget]]:
        items = self.selected_records()
        if not items: raise ValueError("请选择至少一个目标文件")
        base = items[0].header
        if any(item.header.protocol != base.protocol or item.header.point_count != base.point_count or item.header.frequency_hz != base.frequency_hz for item in items):
            raise ValueError("选中的目标文件必须具有相同协议、通道数和采样率")
        matrices: list[np.ndarray] = []
        for item in items:
            if item.original is None or item.working is None:
                self.status.setText(f"正在读取目标数据: {item.path.name}")
                QtWidgets.QApplication.processEvents()
                matrix = read_signal_matrix(item.path, item.header).astype(np.float32, copy=False)
                item.original = matrix.copy()
                item.working = matrix.copy()
            matrices.append(item.working)
        matrix = np.concatenate(matrices, axis=1)
        return matrix, replace(base, sample_count=matrix.shape[1], acquisition_time_s=matrix.shape[1] / base.frequency_hz), items

    def source_data(self) -> tuple[np.ndarray, DatHeader]:
        paths = [item.data(QtCore.Qt.ItemDataRole.UserRole) for item in self.source_list.selectedItems()]
        if not paths: raise ValueError("请选择替换来源文件")
        data = [self.sources[path] for path in paths]
        target_rate = self.target_data()[1].frequency_hz
        base = data[0][1]
        target_header = self.target_data()[1]
        if base.protocol != target_header.protocol: raise ValueError("替换来源协议必须与目标一致")
        if any(header.protocol != base.protocol or header.point_count != base.point_count for _, header in data): raise ValueError("替换来源的协议和通道数必须一致")
        matrix = np.concatenate([resample_matrix_to_rate(values, header.frequency_hz, target_rate) for values, header in data], axis=1)
        return matrix, replace(base, frequency_hz=target_rate, sample_count=matrix.shape[1], acquisition_time_s=matrix.shape[1] / target_rate)

    def refresh_target(self) -> None:
        try:
            matrix, header, _ = self.target_data(); maximum = header.point_count - 1
            for spin in (self.start, self.end, self.reference): spin.setRange(0, maximum)
            self.end.setValue(max(self.start.value(), min(self.end.value(), maximum))); self.minimum.setRange(0, header.frequency_hz / 2 - 1e-6); self.minimum.setValue(min(max(200, self.minimum.value()), self.minimum.maximum()))
            self.target_pane.set_data(matrix, header, self.shared_levels())
        except ValueError: pass

    def refresh_source(self) -> None:
        try:
            matrix, header = self.source_data(); self.source_pane.set_data(matrix, header, self.shared_levels()); self.apply_shared_color_levels()
        except ValueError: pass

    def shared_levels(self) -> tuple[float, float]:
        limit = float(self.color_limit.value())
        return -limit, limit

    def apply_shared_color_levels(self, _value: object = None) -> None:
        levels = self.shared_levels()
        self.target_pane.image_item.setLevels(levels)
        self.source_pane.image_item.setLevels(levels)
        self.shared_color_bar.setLevels(levels)

    def add_sources(self) -> None:
        names, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "选择替换文件", "", "Data files (*.dat *.tdms);;All files (*.*)")
        self.add_source_paths([Path(name) for name in names])

    def add_source_paths(self, paths: list[Path]) -> None:
        target_protocol = self.selected_records()[0].header.protocol if self.selected_records() else None
        for path in paths:
            try:
                header = read_header(path)
                if target_protocol is not None and header.protocol != target_protocol: raise ValueError("协议必须与目标一致")
                self.sources[path] = (read_signal_matrix(path, header), header)
                item = QtWidgets.QListWidgetItem(path.name); item.setData(QtCore.Qt.ItemDataRole.UserRole, path); self.source_list.addItem(item)
            except Exception as exc: self.status.setText(f"无法添加 {path.name}: {exc}")

    def add_mapping(self) -> None:
        try: maximum = self.target_data()[1].point_count - 1
        except ValueError: return
        row = self.table.rowCount(); self.table.insertRow(row)
        for column, value in enumerate((0, maximum, 0, maximum)):
            spin = QtWidgets.QSpinBox(); spin.setRange(0, maximum); spin.setValue(value); self.table.setCellWidget(row, column, spin)

    def mappings(self) -> list[tuple[int, int, int, int]]:
        values = []
        for row in range(self.table.rowCount()): values.append(tuple(int(self.table.cellWidget(row, col).value()) for col in range(4)))
        return values

    def delete_mapping(self) -> None:
        for row in sorted({item.row() for item in self.table.selectedItems()}, reverse=True): self.table.removeRow(row)

    def apply_filter(self) -> None:
        try:
            matrix, header, records = self.target_data(); result, detected = flatten_spectral_channels(
                matrix,
                header.frequency_hz,
                self.start.value(),
                self.end.value(),
                SpectralFlatteningSettings(self.minimum.value(), self.threshold.value(), self.q.value()),
                reference_channel=self.reference.value(),
            )
            offset = 0
            for record in records:
                count = record.header.sample_count; record.working = result[:, offset:offset + count]; offset += count
            common_count = next(iter(detected.values()), np.array([], dtype=float)).size
            self.refresh_target(); self.status.setText(f"滤波已完成：以参考通道 {self.reference.value()} 检测到 {common_count} 个频率，并复用于全部选中通道。")
        except Exception as exc: QtWidgets.QMessageBox.critical(self, "滤波失败", str(exc))

    def apply_replacement(self) -> None:
        try:
            target, header, records = self.target_data(); source, _ = self.source_data()
            if source.shape[1] > target.shape[1]: raise ValueError("替换来源总时长不能超过目标；请减少来源文件")
            for ts, te, ss, se in self.mappings():
                if not (0 <= ts <= te < target.shape[0] and 0 <= ss <= se < source.shape[0] and te-ts == se-ss): raise ValueError("通道映射无效或两侧通道数不同")
                target[ts:te+1, :source.shape[1]] = source[ss:se+1]
            offset = 0
            for record in records:
                count = record.header.sample_count; record.working = target[:, offset:offset + count]; offset += count
            self.refresh_target(); self.status.setText(f"替换已完成：覆盖目标前 {source.shape[1] / header.frequency_hz:.3f} s。")
        except Exception as exc: QtWidgets.QMessageBox.critical(self, "替换失败", str(exc))

    def batch_filter_all_targets(self) -> None:
        """Detect once from the current reference trace, then stream files one by one."""
        if not self.records:
            QtWidgets.QMessageBox.information(self, "批量滤波", "请先选择目标文件夹")
            return
        try:
            current, current_header, _ = self.target_data()
            settings = SpectralFlatteningSettings(self.minimum.value(), self.threshold.value(), self.q.value())
            peaks = detect_interference_frequencies(current[self.reference.value()], current_header.frequency_hz, settings)
            if peaks.size == 0:
                QtWidgets.QMessageBox.information(self, "批量滤波", "参考通道未检测到满足阈值的干扰峰，未开始导出。")
                return
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "批量滤波", f"无法生成公共陷波器: {exc}")
            return
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "选择批量滤波导出目录", "")
        if not directory:
            return

        output_dir = Path(directory)
        total = len(self.records)
        completed = 0
        failures: list[str] = []
        self.batch_progress.setRange(0, total)
        self.batch_progress.setValue(0)
        try:
            for index, record in enumerate(self.records, start=1):
                self.status.setText(f"正在批量滤波 {index}/{total}: {record.path.name}")
                self.batch_progress.setFormat(f"正在滤波 {index}/{total}: {record.path.name}")
                QtWidgets.QApplication.processEvents()
                matrix: np.ndarray | None = None
                try:
                    if record.header.point_count <= self.end.value() or record.header.frequency_hz / 2.0 <= float(np.max(peaks)):
                        raise ValueError("通道范围或 Nyquist 频率不兼容")
                    matrix = read_signal_matrix(record.path, record.header).astype(np.float32, copy=False)
                    for channel in range(self.start.value(), self.end.value() + 1):
                        matrix[channel] = apply_notch_frequencies(matrix[channel], record.header.frequency_hz, peaks, settings.q_factor)
                    output = output_dir / f"{record.path.stem}_filtered{record.path.suffix}"
                    export_processed_signal(record.path, record.header, matrix, output)
                    completed += 1
                except Exception as exc:
                    failures.append(f"{record.path.name}: {exc}")
                finally:
                    # Do not retain a completed file's matrix; this is what
                    # keeps a large folder from accumulating RAM usage.
                    del matrix
                    gc.collect()
                self.batch_progress.setValue(index)
                QtWidgets.QApplication.processEvents()
        finally:
            self.batch_progress.setFormat(f"批量滤波完成：成功 {completed}/{total}" + (f"，失败 {len(failures)}" if failures else ""))
        self.status.setText(self.batch_progress.format())
        if failures:
            QtWidgets.QMessageBox.warning(self, "批量滤波完成", "部分文件未处理：\n" + "\n".join(failures[:8]))

    def batch_replace_all_targets(self) -> None:
        """Stream one target/source pair at a time; source paths cycle as needed."""
        if not self.records:
            QtWidgets.QMessageBox.information(self, "批量替换", "请先选择目标文件夹")
            return
        try:
            mappings = self.mappings()
            if not mappings:
                raise ValueError("请先添加至少一条通道映射")
            selected = [item.data(QtCore.Qt.ItemDataRole.UserRole) for item in self.source_list.selectedItems()]
            source_paths = selected or list(self.sources)
            if not source_paths:
                raise ValueError("请先选择至少一个替换来源文件")
        except Exception as exc:
            QtWidgets.QMessageBox.information(self, "批量替换", str(exc))
            return
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "选择批量替换导出目录", "")
        if not directory:
            return

        total = len(self.records)
        completed = 0
        failures: list[str] = []
        self.batch_progress.setRange(0, total)
        self.batch_progress.setValue(0)
        try:
            for index, target_record in enumerate(self.records, start=1):
                source_path = source_paths[(index - 1) % len(source_paths)]
                self.status.setText(f"正在批量替换 {index}/{total}: {target_record.path.name} ← {source_path.name}")
                self.batch_progress.setFormat(f"正在替换 {index}/{total}: {target_record.path.name}")
                QtWidgets.QApplication.processEvents()
                target_matrix: np.ndarray | None = None
                source_matrix: np.ndarray | None = None
                try:
                    source_header = self.sources[source_path][1]
                    if source_header.protocol != target_record.header.protocol:
                        raise ValueError("目标与替换来源协议不一致")
                    target_matrix = read_signal_matrix(target_record.path, target_record.header).astype(np.float32, copy=False)
                    # Read the source fresh for this pair so completed pairs do
                    # not cause memory use to increase with target count.
                    source_matrix = read_signal_matrix(source_path, source_header).astype(np.float32, copy=False)
                    source_matrix = resample_matrix_to_rate(source_matrix, source_header.frequency_hz, target_record.header.frequency_hz)
                    replace_samples = min(target_matrix.shape[1], source_matrix.shape[1])
                    if replace_samples < 1:
                        raise ValueError("替换来源没有可用采样点")
                    for target_start, target_end, source_start, source_end in mappings:
                        if not (
                            0 <= target_start <= target_end < target_matrix.shape[0]
                            and 0 <= source_start <= source_end < source_matrix.shape[0]
                            and target_end - target_start == source_end - source_start
                        ):
                            raise ValueError("通道映射超出目标或来源范围，或两侧通道数不同")
                        target_matrix[target_start : target_end + 1, :replace_samples] = source_matrix[source_start : source_end + 1, :replace_samples]
                    output = Path(directory) / f"{target_record.path.stem}_replaced{target_record.path.suffix}"
                    export_processed_signal(target_record.path, target_record.header, target_matrix, output)
                    completed += 1
                except Exception as exc:
                    failures.append(f"{target_record.path.name}: {exc}")
                finally:
                    del target_matrix
                    del source_matrix
                    gc.collect()
                self.batch_progress.setValue(index)
                QtWidgets.QApplication.processEvents()
        finally:
            self.batch_progress.setFormat(f"批量替换完成：成功 {completed}/{total}" + (f"，失败 {len(failures)}" if failures else ""))
        self.status.setText(self.batch_progress.format())
        if failures:
            QtWidgets.QMessageBox.warning(self, "批量替换完成", "部分文件未处理：\n" + "\n".join(failures[:8]))

    def reset(self) -> None:
        _, _, records = self.target_data()
        for record in records:
            if record.original is not None:
                record.working = record.original.copy()
        self.refresh_target(); self.status.setText("所选目标已恢复原始数据。")

    def open_point_analysis(self, point: object) -> None:
        try:
            if isinstance(point, QtCore.QPointF):
                data_point = point
            elif hasattr(point, "scenePos"):
                data_point = self.target_pane.plot.getViewBox().mapSceneToView(point.scenePos())
            else:
                return
            matrix, header, _ = self.target_data(); index = int(round((data_point.x() - header.position_start_m) / max(header.point_interval_m, 1e-12)))
            if 0 <= index < header.point_count and 0 <= data_point.y() <= header.time_span_s:
                window = PointAnalysisWindow(matrix, header, index); self.point_windows.append(window); window.show()
                self.status.setText(f"已打开点位分析：通道 {index}")
        except Exception as exc:
            self.status.setText(f"无法打开点位分析: {exc}")

    def export(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "选择导出目录", "")
        if not directory: return
        try:
            _, _, records = self.target_data()
            for record in records:
                if record.working is None:
                    raise ValueError(f"目标数据尚未加载: {record.path.name}")
                export_processed_signal(record.path, record.header, record.working, Path(directory) / f"{record.path.stem}_processed{record.path.suffix}")
            self.status.setText("已导出所选目标文件。")
        except Exception as exc: QtWidgets.QMessageBox.critical(self, "导出失败", str(exc))


class DatViewer(QtWidgets.QMainWindow):
    def __init__(self, start_path: Path | None = None) -> None:
        super().__init__()
        self.setWindowTitle("CDAS/TDAS DAT Viewer - PyQtGraph")
        self.resize(1400, 860)

        self.headers: dict[Path, DatHeader] = {}
        self.current_path: Path | None = None
        self.current_paths: list[Path] = []
        self.current_source_header: DatHeader | None = None
        self.current_header: DatHeader | None = None
        self.current_raw_matrix: np.ndarray | None = None
        self.current_matrix: np.ndarray | None = None
        self.current_display_matrix: np.ndarray | None = None
        self.silixa_normalized = True
        self.silixa_normalization_scale: float | None = None
        self.normalization_btn: QtWidgets.QPushButton | None = None
        self.time_difference_enabled = False
        self.time_difference_checkbox: QtWidgets.QCheckBox | None = None
        self.analysis_windows: list[QtWidgets.QMainWindow] = []
        self.spectrum_3d_window: Spectrum3DWindow | None = None
        self.spectrum_3d_compare_windows: list[Spectrum3DWindow] = []
        self.spatial_psd_window: SpatialPsdComparisonWindow | None = None
        self.report_workspace: QtWidgets.QMainWindow | None = None
        self.processing_workspace: ProcessingWorkspaceWindow | None = None

        self._build_ui()
        if start_path is not None:
            self.load_path(start_path)
        else:
            self.status.setText("请选择文件夹或数据文件后开始读取")
            self.info.setPlainText("请选择文件夹或数据文件后开始读取。")

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
        self.path_edit.setPlaceholderText("请选择文件夹或数据文件")
        folder_btn = QtWidgets.QPushButton("选择文件夹")
        file_btn = QtWidgets.QPushButton("选择数据文件")
        reload_btn = QtWidgets.QPushButton("重新读取")
        show_selected_btn = QtWidgets.QPushButton("显示选中文件")
        spectrum_btn = QtWidgets.QPushButton("频谱热力图")
        band_energy_btn = QtWidgets.QPushButton("频段能量热力图")
        spatial_psd_btn = QtWidgets.QPushButton("空间段PSD对比")
        report_btn = QtWidgets.QPushButton("DAS实验报告工作台")
        processing_btn = QtWidgets.QPushButton("数据处理")
        spectrum_3d_btn = QtWidgets.QPushButton("3D频谱图")
        new_spectrum_3d_btn = QtWidgets.QPushButton("新建3D频谱图")
        self.normalization_btn = QtWidgets.QPushButton("显示原始数据")
        self.normalization_btn.setToolTip("Silixa TDMS 默认按最大绝对值归一化到 3.14；点击可切换原始计数/归一化")
        self.normalization_btn.setEnabled(False)
        self.time_difference_checkbox = QtWidgets.QCheckBox("时域一阶差分")
        self.time_difference_checkbox.setToolTip(
            "仅 CDAS 数据可用。对拼接后的每个点位计算 x[t] - x[t-1]，采样数减少 1。"
        )
        self.time_difference_checkbox.setEnabled(False)
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
        toolbar.addWidget(self.normalization_btn)
        toolbar.addWidget(self.time_difference_checkbox)
        toolbar.addWidget(spectrum_btn)
        toolbar.addWidget(band_energy_btn)
        toolbar.addWidget(spatial_psd_btn)
        toolbar.addWidget(report_btn)
        toolbar.addWidget(processing_btn)
        toolbar.addWidget(spectrum_3d_btn)
        toolbar.addWidget(new_spectrum_3d_btn)
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
        self.normalization_btn.clicked.connect(self.toggle_silixa_normalization)
        self.time_difference_checkbox.toggled.connect(self.toggle_time_difference)
        spectrum_btn.clicked.connect(self.show_spectrum_heatmap)
        band_energy_btn.clicked.connect(self.show_band_energy_heatmap)
        spatial_psd_btn.clicked.connect(self.show_spatial_psd_comparison)
        report_btn.clicked.connect(self.show_report_workspace)
        processing_btn.clicked.connect(self.show_processing_workspace)
        spectrum_3d_btn.clicked.connect(self.show_spectrum_3d)
        new_spectrum_3d_btn.clicked.connect(self.show_new_spectrum_3d)
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
        self.file_model = QtGui.QFileSystemModel(self)
        self.file_model.setFilter(
            QtCore.QDir.Filter.AllDirs
            | QtCore.QDir.Filter.Files
            | QtCore.QDir.Filter.NoDotAndDotDot
        )
        self.file_model.setNameFilters(["*.dat", "*.DAT", "*.tdms", "*.TDMS"])
        self.file_model.setNameFilterDisables(False)
        self.file_tree = QtWidgets.QTreeView()
        self.file_tree.setModel(self.file_model)
        self.file_tree.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.file_tree.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.file_tree.setSortingEnabled(True)
        self.file_tree.sortByColumn(0, QtCore.Qt.SortOrder.AscendingOrder)
        for column in range(1, self.file_model.columnCount()):
            self.file_tree.hideColumn(column)
        left_layout.addWidget(QtWidgets.QLabel("数据文件（可展开子文件夹，可多选）"))
        left_layout.addWidget(self.file_tree)
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

        self.file_tree.selectionModel().currentChanged.connect(self.on_file_selected)
        self.plot.scene().sigMouseClicked.connect(self.on_plot_clicked)
        self.plot.vb.sigRangeChanged.connect(self.on_view_range_changed)

    def is_silixa_current(self) -> bool:
        return self.current_header is not None and self.current_header.protocol == "Silixa TDMS"

    def is_cdas_current(self) -> bool:
        return self.current_source_header is not None and self.current_source_header.protocol == "CDAS"

    def current_value_mode_note(self) -> str | None:
        if self.is_cdas_current() and self.time_difference_enabled:
            return "当前值模式: CDAS 时域一阶差分 x[t] - x[t-1]"
        if not self.is_silixa_current():
            return None
        if self.silixa_normalized:
            scale = self.silixa_normalization_scale
            if scale is None or scale <= 0.0:
                return "当前值模式: Silixa归一化，数据全零或无有效最大值"
            return (
                "当前值模式: Silixa归一化，"
                f"最大绝对值 {scale:.6g} raw counts -> {SILIXA_NORMALIZED_MAX:.2f}"
            )
        return "当前值模式: Silixa原始 int16 计数"

    def update_normalization_button(self) -> None:
        if self.normalization_btn is None:
            return
        enabled = self.is_silixa_current() and self.current_raw_matrix is not None
        self.normalization_btn.setEnabled(enabled)
        if not enabled:
            self.normalization_btn.setText("显示原始数据")
            return
        self.normalization_btn.setText("显示原始数据" if self.silixa_normalized else "恢复归一化")

    def update_time_difference_control(self) -> None:
        if self.time_difference_checkbox is None:
            return
        enabled = (
            self.is_cdas_current()
            and self.current_raw_matrix is not None
            and self.current_raw_matrix.shape[1] >= 2
        )
        if not enabled:
            self.time_difference_enabled = False
            self.time_difference_checkbox.blockSignals(True)
            self.time_difference_checkbox.setChecked(False)
            self.time_difference_checkbox.blockSignals(False)
        self.time_difference_checkbox.setEnabled(enabled)

    def apply_current_value_mode(self) -> None:
        if self.current_raw_matrix is None or self.current_source_header is None:
            self.current_matrix = None
            self.current_header = None
            self.silixa_normalization_scale = None
            self.update_normalization_button()
            self.update_time_difference_control()
            return

        self.update_time_difference_control()
        header = self.current_source_header
        matrix = self.current_raw_matrix
        if header.protocol == "CDAS" and self.time_difference_enabled:
            if matrix.shape[1] < 2:
                raise ValueError("时域一阶差分至少需要 2 个采样点")
            matrix = np.diff(matrix, axis=1)
            header = replace(
                header,
                sample_count=int(matrix.shape[1]),
                acquisition_time_s=float(matrix.shape[1]) / max(1, header.frequency_hz),
            )

        self.current_header = header
        if self.is_silixa_current() and self.silixa_normalized:
            self.current_matrix, self.silixa_normalization_scale = normalize_silixa_matrix(matrix)
        else:
            self.current_matrix = matrix
            self.silixa_normalization_scale = None
        self.update_normalization_button()
        self.update_time_difference_control()

    def refresh_current_heatmap(self, reset_view: bool = False) -> None:
        if self.current_header is None or self.current_matrix is None:
            return
        header = self.current_header
        self.current_display_matrix = make_display_matrix_max_abs(self.current_matrix)

        image = self.current_display_matrix.T
        self.image_item.setImage(image, autoLevels=False)
        x_width = max(header.distance_span_m, header.point_interval_m)
        rect = QtCore.QRectF(header.position_start_m, 0.0, x_width, header.time_span_s)
        self.image_item.setRect(rect)
        if reset_view:
            self.set_range_controls(header.position_start_m, header.position_start_m + x_width, 0.0, header.time_span_s)
            self.show_full_range()
            self.marker.hide()
            self.x_selector.hide()
        self.update_levels_only(update_status=False)

    def toggle_silixa_normalization(self) -> None:
        if not self.is_silixa_current() or self.current_raw_matrix is None:
            return
        self.silixa_normalized = not self.silixa_normalized
        self.apply_current_value_mode()
        self.refresh_current_heatmap(reset_view=False)
        self.update_spatial_psd_source()
        mode_text = "归一化" if self.silixa_normalized else "原始计数"
        self.status.setText(f"已切换 Silixa TDMS 显示/分析模式: {mode_text}")

    def toggle_time_difference(self, checked: bool) -> None:
        if not self.is_cdas_current() or self.current_raw_matrix is None:
            return
        self.time_difference_enabled = bool(checked)
        self.apply_current_value_mode()
        self.refresh_current_heatmap(reset_view=True)
        self.update_spatial_psd_source()
        mode_text = "已启用" if self.time_difference_enabled else "已关闭"
        self.status.setText(f"CDAS 时域一阶差分{mode_text}；后续新打开的分析窗口将使用当前数据")

    def choose_folder(self) -> None:
        selected = QtWidgets.QFileDialog.getExistingDirectory(self, "选择数据文件夹", self.dialog_start_path())
        if selected:
            self.load_path(Path(selected))

    def choose_file(self) -> None:
        selected, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择数据文件",
            self.dialog_start_path(),
            "Data files (*.dat *.tdms);;DAT files (*.dat);;TDMS files (*.tdms);;All files (*.*)",
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
            self.status.setText("请先选择文件夹或数据文件")
            return
        self.load_path(Path(text))

    def load_path(self, path: Path) -> None:
        path = path.expanduser()
        if path.is_file():
            root_path = path.parent
            selected_file: Path | None = path
        elif path.is_dir():
            root_path = path
            selected_file = None
        else:
            self.status.setText(f"路径不存在: {path}")
            return

        self.path_edit.setText(str(path))
        self.headers = {}
        root_index = self.file_model.setRootPath(str(root_path))
        self.file_tree.setRootIndex(root_index)
        self.file_tree.expand(root_index)
        self.status.setText(f"已打开文件树：{root_path}；展开子文件夹后选择 .dat 或 .tdms 文件")
        if selected_file is not None:
            QtCore.QTimer.singleShot(0, lambda selected=selected_file: self.select_tree_file(selected))

    def select_tree_file(self, path: Path) -> None:
        index = self.file_model.index(str(path))
        if not index.isValid():
            return
        parent = index.parent()
        while parent.isValid():
            self.file_tree.expand(parent)
            parent = parent.parent()
        self.file_tree.setCurrentIndex(index)
        self.file_tree.scrollTo(index)

    def data_path_from_tree_index(self, index: QtCore.QModelIndex) -> Path | None:
        if not index.isValid() or self.file_model.isDir(index):
            return None
        path = Path(self.file_model.filePath(index))
        if path.suffix.lower() not in {".dat", ".tdms"}:
            return None
        return path

    def selected_tree_data_paths(self) -> list[Path]:
        indexes = self.file_tree.selectionModel().selectedRows(0)
        paths = {path for index in indexes if (path := self.data_path_from_tree_index(index)) is not None}
        if not paths:
            current = self.data_path_from_tree_index(self.file_tree.currentIndex())
            if current is not None:
                paths.add(current)
        return sorted(paths, key=lambda path: str(path).lower())

    def ensure_headers(self, paths: list[Path]) -> None:
        for path in paths:
            if path not in self.headers:
                self.headers[path] = read_header(path)

    def on_file_selected(self, current: QtCore.QModelIndex, _previous: QtCore.QModelIndex) -> None:
        path = self.data_path_from_tree_index(current)
        if path is None:
            return
        self.display_paths([path])

    def show_selected_files(self) -> None:
        paths = self.selected_tree_data_paths()
        if paths:
            self.display_paths(paths)
        else:
            self.status.setText("请选择一个或多个 .dat/.tdms 数据文件")

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
        try:
            self.ensure_headers(paths)
            paths = sorted(paths, key=lambda path: str(path).lower())
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
        self.current_source_header = header
        self.current_raw_matrix = matrix
        self.silixa_normalized = header.protocol == "Silixa TDMS"
        self.apply_current_value_mode()
        self.refresh_current_heatmap(reset_view=True)
        self.info.setPlainText(
            format_display_info(
                paths,
                self.current_header or header,
                self.current_matrix,
                self.color_limit.value(),
                self.current_display_matrix,
                self.current_value_mode_note(),
            )
        )
        self.update_spatial_psd_source()
        if len(paths) == 1:
            self.status.setText(f"已显示 {first_path.name}")
        else:
            display_header = self.current_header or header
            self.status.setText(f"已拼接显示 {len(paths)} 个文件，总时长 {display_header.time_span_s:.3f} s")

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

    def update_levels_only(self, _value: object = None, update_status: bool = True) -> None:
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
                    self.current_value_mode_note(),
                )
            )
            self.status.setText(f"已更新色条范围: {-limit:.2f} ~ {limit:.2f}")

    def show_spectrum_heatmap(self) -> None:
        if self.current_header is None or self.current_matrix is None:
            QtWidgets.QMessageBox.information(self, "提示", "请先显示一个或多个数据文件")
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

    def show_band_energy_heatmap(self) -> None:
        if self.current_header is None or self.current_matrix is None:
            QtWidgets.QMessageBox.information(self, "提示", "请先显示一个或多个数据文件")
            return
        paths = self.current_paths or ([self.current_path] if self.current_path else [])
        if len(paths) <= 1:
            title = "DAS Band Energy Heatmap"
        else:
            title = f"DAS Band Energy Heatmap - {len(paths)} 个文件"
        try:
            self.status.setText("正在打开频段能量热力图...")
            QtWidgets.QApplication.processEvents()
            window = BandEnergyHeatmapWindow(self.current_matrix, self.current_header, title)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "频段能量热力图失败", str(exc))
            self.status.setText("频段能量热力图计算失败")
            return
        self.analysis_windows.append(window)
        window.show()
        self.status.setText("已打开频段能量热力图")

    def update_spatial_psd_source(self) -> None:
        if self.spatial_psd_window is None or self.current_header is None or self.current_matrix is None:
            return
        paths = self.current_paths or ([self.current_path] if self.current_path else [])
        if not paths:
            return
        self.spatial_psd_window.set_source(
            self.current_matrix,
            self.current_header,
            paths,
            self.current_value_mode_note(),
            self.silixa_normalized,
            self.silixa_normalization_scale,
        )

    def show_spatial_psd_comparison(self) -> None:
        if self.current_header is None or self.current_matrix is None:
            QtWidgets.QMessageBox.information(self, "提示", "请先显示一个或多个数据文件")
            return
        if self.spatial_psd_window is None:
            self.spatial_psd_window = SpatialPsdComparisonWindow()
            self.analysis_windows.append(self.spatial_psd_window)
        self.update_spatial_psd_source()
        self.spatial_psd_window.show()
        self.spatial_psd_window.raise_()
        self.spatial_psd_window.activateWindow()
        self.status.setText("已打开空间段 PSD 曲线对比窗口")

    def show_report_workspace(self) -> None:
        """Open the persisted, time-window based batch-report workflow.

        The import is intentionally lazy: ``das_report`` reuses this module's
        pure readers and PSD routine, so importing it during module startup
        would create a circular import.
        """
        if self.report_workspace is None:
            from das_report import ReportWorkspaceWindow

            self.report_workspace = ReportWorkspaceWindow()
            self.analysis_windows.append(self.report_workspace)
            self.report_workspace.destroyed.connect(lambda _obj=None: setattr(self, "report_workspace", None))
        self.report_workspace.show()
        self.report_workspace.raise_()
        self.report_workspace.activateWindow()
        self.status.setText("已打开 DAS 实验报告工作台")

    def show_processing_workspace(self) -> None:
        """Open a non-destructive processor for one or more selected targets."""
        targets: list[tuple[Path, DatHeader, np.ndarray]] = []
        if self.current_paths:
            try:
                targets = [(path, self.headers[path], read_signal_matrix(path, self.headers[path])) for path in self.current_paths]
            except Exception as exc:
                QtWidgets.QMessageBox.critical(self, "数据处理", f"无法读取目标文件: {exc}")
                return
        window = SimpleProcessingWorkspaceWindow(targets)
        window.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.processing_workspace = window
        self.analysis_windows.append(window)
        window.destroyed.connect(lambda _obj=None: setattr(self, "processing_workspace", None))
        window.show()
        window.raise_()
        window.activateWindow()
        self.status.setText("已打开数据处理工作台")

    def show_spectrum_3d(self) -> None:
        if self.current_header is None or self.current_matrix is None:
            QtWidgets.QMessageBox.information(self, "提示", "请先显示一个或多个数据文件")
            return
        paths = self.current_paths or ([self.current_path] if self.current_path else [])
        if len(paths) <= 1:
            title = "3D频谱图"
        else:
            title = f"3D频谱图 - {len(paths)} 个文件"
        try:
            self.status.setText("正在计算3D频谱图...")
            QtWidgets.QApplication.processEvents()
            window = self.spectrum_3d_window
            if window is None:
                window = Spectrum3DWindow(self.current_matrix, self.current_header, title)
                window.destroyed.connect(lambda _obj=None: setattr(self, "spectrum_3d_window", None))
                self.spectrum_3d_window = window
                self.analysis_windows.append(window)
            else:
                window.set_spectrum_data(self.current_matrix, self.current_header, title)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "3D频谱图失败", str(exc))
            self.status.setText("3D频谱图计算失败")
            return
        window.show()
        window.raise_()
        window.activateWindow()
        window.request_surface_refresh()
        QtCore.QTimer.singleShot(200, window.request_surface_refresh)
        self.status.setText("已打开3D频谱图")

    def _remove_spectrum_3d_compare_window(self, window: Spectrum3DWindow) -> None:
        self.spectrum_3d_compare_windows = [
            item for item in self.spectrum_3d_compare_windows if item is not window
        ]
        self.analysis_windows = [item for item in self.analysis_windows if item is not window]

    def show_new_spectrum_3d(self) -> None:
        if self.current_header is None or self.current_matrix is None:
            QtWidgets.QMessageBox.information(self, "提示", "请先显示一个有效数据文件")
            return
        max_compare_windows = 3
        if len(self.spectrum_3d_compare_windows) >= max_compare_windows:
            QtWidgets.QMessageBox.information(
                self,
                "3D频谱图",
                f"最多同时打开 {max_compare_windows} 个3D对比窗口，请先关闭旧窗口。",
            )
            return

        paths = self.current_paths or ([self.current_path] if self.current_path else [])
        compare_index = len(self.spectrum_3d_compare_windows) + 1
        if len(paths) <= 1:
            title = f"3D频谱图 - 对比 {compare_index}"
        else:
            title = f"3D频谱图 - 对比 {compare_index} - {len(paths)} 个文件"
        try:
            self.status.setText("正在加载新的3D频谱图...")
            QtWidgets.QApplication.processEvents()
            window = Spectrum3DWindow(self.current_matrix, self.current_header, title)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "3D频谱图失败", str(exc))
            self.status.setText("3D频谱图生成失败")
            return
        window.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.spectrum_3d_compare_windows.append(window)
        self.analysis_windows.append(window)
        window.destroyed.connect(lambda _obj=None, w=window: self._remove_spectrum_3d_compare_window(w))
        window.show()
        window.raise_()
        window.activateWindow()
        window.request_surface_refresh()
        QtCore.QTimer.singleShot(200, window.request_surface_refresh)
        QtCore.QTimer.singleShot(500, window.request_surface_refresh)
        QtCore.QTimer.singleShot(1000, window.request_surface_refresh)
        self.status.setText(f"已打开3D频谱图对比窗口 {compare_index}")

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

        window = PointAnalysisWindow(self.current_matrix, header, point_index)
        self.analysis_windows.append(window)
        window.show()


def main() -> int:
    QtWidgets.QApplication.setAttribute(
        QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts,
        True,
    )
    app = QtWidgets.QApplication(sys.argv)
    if "--smoke-test" in sys.argv:
        # Exercise the shipped Qt DLLs, platform plugin, and report imports
        # through the actual executable before publishing a portable build.
        from das_report import ReportWorkspaceWindow

        viewer = DatViewer(None)
        report = ReportWorkspaceWindow()
        viewer.show()
        report.show()
        QtCore.QTimer.singleShot(1000, app.quit)
        return app.exec()
    start_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    viewer = DatViewer(start_path)
    viewer.showMaximized()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
