from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
from PyQt6 import QtWidgets

from tdas_dat_viewer import (
    DatHeader,
    SpatialPsdComparisonWindow,
    compute_spatial_segment_psd,
    psd_to_db,
)


def make_header(
    *,
    fs: int = 1024,
    points: int = 3,
    samples: int = 1024,
    dx: float = 1.0,
    start: float = 0.0,
    protocol: str = "CDAS",
) -> DatHeader:
    return DatHeader(
        protocol=protocol,
        system_id="TEST",
        year=2026,
        month=7,
        day=10,
        hour=0,
        minute=0,
        second=0.0,
        length=0,
        global_alarm=0,
        acquisition_status=0,
        channel=1,
        frequency_hz=fs,
        acquisition_time_s=samples / fs,
        point_interval_m=dx,
        point_count=points,
        sample_count=samples,
        position_start_m=start,
        raw_value_note="synthetic",
    )


class SpatialPsdAlgorithmTests(unittest.TestCase):
    def test_bin_aligned_sine_peak_and_variance(self) -> None:
        fs = 1024
        samples = 1024
        freq = 64.0
        amplitude = 2.0
        t = np.arange(samples, dtype=np.float32) / fs
        trace = amplitude * np.sin(2.0 * np.pi * freq * t)
        matrix = np.vstack([trace, trace]).astype(np.float32)
        header = make_header(fs=fs, points=2, samples=samples)

        result = compute_spatial_segment_psd(matrix, header, 0.0, 1.0, 1.0, 0.0, "Rectangular")
        peak_freq = float(result.freqs_hz[int(np.argmax(result.mean_psd))])
        df = float(result.freqs_hz[1] - result.freqs_hz[0])
        integrated = float(np.sum(result.mean_psd) * df)

        self.assertAlmostEqual(peak_freq, freq, places=5)
        self.assertAlmostEqual(integrated, amplitude**2 / 2.0, places=5)

    def test_opposite_phase_channels_do_not_cancel(self) -> None:
        fs = 512
        samples = 512
        t = np.arange(samples, dtype=np.float32) / fs
        trace = np.sin(2.0 * np.pi * 32.0 * t)
        matrix = np.vstack([trace, -trace]).astype(np.float32)
        header = make_header(fs=fs, points=2, samples=samples)

        result = compute_spatial_segment_psd(matrix, header, 0.0, 1.0, 1.0, 0.0, "Rectangular")
        df = float(result.freqs_hz[1] - result.freqs_hz[0])
        self.assertAlmostEqual(float(np.sum(result.mean_psd) * df), 0.5, places=5)

    def test_spatial_mean_and_median_handle_outlier_differently(self) -> None:
        fs = 512
        samples = 512
        t = np.arange(samples, dtype=np.float32) / fs
        base = np.sin(2.0 * np.pi * 16.0 * t)
        matrix = np.vstack([base, base, 10.0 * base]).astype(np.float32)
        header = make_header(fs=fs, points=3, samples=samples)

        result = compute_spatial_segment_psd(matrix, header, 0.0, 2.0, 1.0, 0.0, "Rectangular")
        peak = int(np.argmax(result.mean_psd))
        self.assertGreater(float(result.mean_psd[peak]), float(result.median_psd[peak]) * 10.0)
        self.assertAlmostEqual(float(result.median_psd[peak]), 0.5, places=5)

    def test_db_conversion_uses_10log10(self) -> None:
        value = psd_to_db(np.array([100.0], dtype=np.float32))[0]
        self.assertAlmostEqual(float(value), 20.0, places=5)

    def test_closed_interval_non_aligned_range_and_single_channel(self) -> None:
        matrix = np.ones((5, 128), dtype=np.float32)
        header = make_header(fs=128, points=5, samples=128, dx=2.0, start=0.0)
        result = compute_spatial_segment_psd(matrix, header, 1.0, 5.0, 0.5, 0.0, "Hann")
        self.assertEqual(result.selected_channel_count, 2)
        self.assertAlmostEqual(result.actual_start_m, 2.0)
        self.assertAlmostEqual(result.actual_end_m, 4.0)

        single = np.ones((1, 128), dtype=np.float32)
        single_header = make_header(fs=128, points=1, samples=128, dx=0.0, start=12.0)
        single_result = compute_spatial_segment_psd(single, single_header, 12.0, 12.0, 0.5, 0.0, "Hann")
        self.assertEqual(single_result.selected_channel_count, 1)

    def test_invalid_ranges_windows_overlap_dx_and_nonfinite_channels(self) -> None:
        matrix = np.ones((3, 100), dtype=np.float32)
        header = make_header(fs=100, points=3, samples=100)
        with self.assertRaisesRegex(ValueError, "起始位置"):
            compute_spatial_segment_psd(matrix, header, 2.0, 1.0, 0.5, 0.0, "Hann")
        with self.assertRaisesRegex(ValueError, "没有实际通道"):
            compute_spatial_segment_psd(matrix, header, 10.0, 11.0, 0.5, 0.0, "Hann")
        with self.assertRaisesRegex(ValueError, "至少需要 2"):
            compute_spatial_segment_psd(matrix, header, 0.0, 2.0, 0.001, 0.0, "Hann")
        with self.assertRaisesRegex(ValueError, "不足以容纳"):
            compute_spatial_segment_psd(matrix, header, 0.0, 2.0, 2.0, 0.0, "Hann")
        with self.assertRaisesRegex(ValueError, "重叠率"):
            compute_spatial_segment_psd(matrix, header, 0.0, 2.0, 0.5, 100.0, "Hann")
        bad_header = make_header(fs=100, points=3, samples=100, dx=0.0)
        with self.assertRaisesRegex(ValueError, "点位间隔"):
            compute_spatial_segment_psd(matrix, bad_header, 0.0, 2.0, 0.5, 0.0, "Hann")

        matrix_with_nan = matrix.copy()
        matrix_with_nan[1, 5] = np.nan
        result = compute_spatial_segment_psd(matrix_with_nan, header, 0.0, 2.0, 0.5, 0.0, "Hann")
        self.assertEqual(result.selected_channel_count, 3)
        self.assertEqual(result.valid_channel_count, 2)
        self.assertTrue(result.warnings)


class SpatialPsdWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self) -> None:
        self.window = SpatialPsdComparisonWindow()

    def tearDown(self) -> None:
        self.window.hide()
        self.window.deleteLater()
        self.app.processEvents()

    def payload_for_current(self, *, mean: bool = True, median: bool = True) -> dict[str, object]:
        payload = dict(self.window.current_payload or {})
        payload.update(
            {
                "include_mean": mean,
                "include_median": median,
                "window_seconds": 0.5,
                "overlap_percent": 0.0,
                "window_name": "Rectangular",
            }
        )
        return payload

    def test_source_replacement_append_delete_y_mode_and_hide(self) -> None:
        fs = 128
        samples = 128
        t = np.arange(samples, dtype=np.float32) / fs
        matrix_a = np.vstack(
            [
                np.sin(2.0 * np.pi * 8.0 * t),
                np.sin(2.0 * np.pi * 8.0 * t),
                np.sin(2.0 * np.pi * 8.0 * t),
            ]
        ).astype(np.float32)
        header_a = make_header(fs=fs, points=3, samples=samples, protocol="CDAS")
        path_a = ROOT / "a.dat"

        self.window.set_source(matrix_a, header_a, [path_a], "当前数据值", False, None)
        result_a = compute_spatial_segment_psd(matrix_a, header_a, 0.0, 2.0, 0.5, 0.0, "Rectangular")
        self.window.on_compute_finished(self.payload_for_current(), result_a)
        self.assertEqual(len(self.window.records), 2)

        result_a2 = compute_spatial_segment_psd(matrix_a * 2.0, header_a, 0.0, 2.0, 0.5, 0.0, "Rectangular")
        self.window.on_compute_finished(self.payload_for_current(), result_a2)
        self.assertEqual(len(self.window.records), 2)
        self.assertEqual(self.window.table.rowCount(), 2)

        matrix_b = matrix_a * 0.5
        header_b = make_header(fs=fs, points=3, samples=samples, protocol="Guanggu DAT")
        path_b = ROOT / "b.dat"
        self.window.set_source(matrix_b, header_b, [path_b], "当前数据值", False, None)
        result_b = compute_spatial_segment_psd(matrix_b, header_b, 0.0, 2.0, 0.5, 0.0, "Rectangular")
        self.window.on_compute_finished(self.payload_for_current(), result_b)
        self.assertEqual(len(self.window.records), 4)
        self.assertEqual(self.window.table.rowCount(), 4)
        self.assertIn("不同协议", self.window.warning_label.text())

        self.window.table.selectRow(0)
        self.window.delete_selected()
        self.assertEqual(len(self.window.records), 3)
        self.assertEqual(self.window.table.rowCount(), 3)

        before = [self.window.record_y_values(record).copy() for record in self.window.ordered_records()]
        self.window.y_mode_combo.setCurrentIndex(1)
        after = [self.window.record_y_values(record).copy() for record in self.window.ordered_records()]
        self.assertFalse(np.allclose(before[0], after[0]))

        self.window.show()
        self.window.close()
        self.assertFalse(self.window.isVisible())
        self.assertEqual(len(self.window.records), 3)


if __name__ == "__main__":
    unittest.main()
