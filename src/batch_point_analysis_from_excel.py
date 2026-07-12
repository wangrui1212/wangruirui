#!/usr/bin/env python
"""Batch-export point-analysis images from DAS test workbook rows.

The script reads the 100 ns and 40 ns sheets, derives a middle 10-second file
window from each row's remark time range, then delegates image generation to
point_analysis_export.py.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET


DEFAULT_EXCEL = Path("outputs/png_to_excel/DAS测试记录表格_40ns_100ns_无EDFA_AOM频率.xlsx")
DEFAULT_DATA_ROOT = Path(r"E:\新方案das数据分析\20260623")
DEFAULT_OUTPUT_DIR = Path("outputs/point_analysis_batch")
DEFAULT_SHEETS = ("100ns脉宽", "40ns脉宽")
DEFAULT_SHEET_POINTS = {"40ns脉宽": 1997}
DEFAULT_TEST_DATE = date(2026, 6, 23)
TIME_RE = re.compile(r"(\d{1,2}):(\d{2})(?::(\d{2}))?")
DAT_RE = re.compile(r"(?P<stamp>\d{14})\.\d+\.dat$", re.IGNORECASE)


@dataclass(frozen=True)
class WorkbookRow:
    sheet: str
    row_number: int
    frequency_hz: object
    pzt_mvpp: object
    remark: str


@dataclass(frozen=True)
class ParsedRange:
    start: datetime
    end: datetime
    status: str
    note: str


@dataclass(frozen=True)
class RawRange:
    start: datetime
    end: datetime
    start_had_seconds: bool
    end_had_seconds: bool


@dataclass(frozen=True)
class DatFile:
    path: Path
    timestamp: datetime


def column_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    value = 0
    for ch in letters:
        value = value * 26 + (ord(ch.upper()) - ord("A") + 1)
    return value


def text_from_si(si: ET.Element, ns: dict[str, str]) -> str:
    parts = []
    for text_node in si.findall(".//m:t", ns):
        parts.append(text_node.text or "")
    return "".join(parts)


def read_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    try:
        root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    return [text_from_si(si, ns) for si in root.findall("m:si", ns)]


def workbook_sheet_paths(zf: zipfile.ZipFile) -> dict[str, str]:
    ns = {
        "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    }
    rel_ns = {"rel": "http://schemas.openxmlformats.org/package/2006/relationships"}
    wb_root = ET.fromstring(zf.read("xl/workbook.xml"))
    rel_root = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    rels = {
        rel.attrib["Id"]: rel.attrib["Target"]
        for rel in rel_root.findall("rel:Relationship", rel_ns)
    }

    paths: dict[str, str] = {}
    for sheet in wb_root.findall("m:sheets/m:sheet", ns):
        rel_id = sheet.attrib[f"{{{ns['r']}}}id"]
        target = rels[rel_id]
        if not target.startswith("/"):
            target = "xl/" + target
        else:
            target = target.lstrip("/")
        paths[sheet.attrib["name"]] = target.replace("\\", "/")
    return paths


def parse_cell(cell: ET.Element, shared_strings: list[str], ns: dict[str, str]) -> object:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        inline = cell.find("m:is", ns)
        return text_from_si(inline, ns) if inline is not None else ""

    value_node = cell.find("m:v", ns)
    if value_node is None:
        return None
    raw = value_node.text or ""

    if cell_type == "s":
        return shared_strings[int(raw)]
    if cell_type in {"str", "b"}:
        return raw

    try:
        number = float(raw)
    except ValueError:
        return raw
    return int(number) if number.is_integer() else number


def read_sheet_rows(excel_path: Path, sheet_name: str) -> list[dict[int, object]]:
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(excel_path) as zf:
        shared_strings = read_shared_strings(zf)
        sheet_paths = workbook_sheet_paths(zf)
        if sheet_name not in sheet_paths:
            raise ValueError(f"Sheet not found: {sheet_name}")
        root = ET.fromstring(zf.read(sheet_paths[sheet_name]))

    rows: list[dict[int, object]] = []
    for row in root.findall(".//m:sheetData/m:row", ns):
        values: dict[int, object] = {0: int(row.attrib.get("r", "0"))}
        for cell in row.findall("m:c", ns):
            ref = cell.attrib.get("r", "")
            values[column_index(ref)] = parse_cell(cell, shared_strings, ns)
        rows.append(values)
    return rows


def iter_workbook_rows(excel_path: Path, sheet_names: Iterable[str]) -> list[WorkbookRow]:
    records: list[WorkbookRow] = []
    for sheet_name in sheet_names:
        for row in read_sheet_rows(excel_path, sheet_name):
            row_number = int(row.get(0, 0) or 0)
            frequency = row.get(1)
            pzt = row.get(2)
            remark = str(row.get(9) or "").strip()
            if row_number < 6 or not remark or not is_number_like(frequency) or not is_number_like(pzt):
                continue
            records.append(WorkbookRow(sheet_name, row_number, frequency, pzt, remark))
    return records


def is_number_like(value: object) -> bool:
    try:
        float(str(value).strip())
    except (TypeError, ValueError):
        return False
    return True


def parse_time_match(match: re.Match[str], default_second: int) -> time:
    hour = int(match.group(1))
    minute = int(match.group(2))
    second = int(match.group(3)) if match.group(3) is not None else default_second
    return time(hour, minute, second)


def combine(test_date: date, t: time) -> datetime:
    return datetime.combine(test_date, t)


def raw_remark_range(remark: str, test_date: date) -> RawRange | None:
    matches = list(TIME_RE.finditer(remark))
    if len(matches) < 2:
        return None

    start_had_seconds = matches[0].group(3) is not None
    end_had_seconds = matches[1].group(3) is not None
    start_time = parse_time_match(matches[0], 0)
    end_default_second = 59 if not end_had_seconds else 0
    end_time = parse_time_match(matches[1], end_default_second)
    return RawRange(
        combine(test_date, start_time),
        combine(test_date, end_time),
        start_had_seconds,
        end_had_seconds,
    )


def infer_reversed_range(raw: RawRange, prev_end: datetime | None, next_start: datetime | None) -> tuple[datetime, datetime, str]:
    candidates: list[tuple[float, datetime, datetime, str]] = []

    def add_candidate(start_dt: datetime, end_dt: datetime, note: str) -> None:
        if end_dt <= start_dt:
            return
        duration = (end_dt - start_dt).total_seconds()
        if duration < 10 or duration > 180:
            return
        score = abs(duration - 35)
        if prev_end is not None and start_dt < prev_end:
            score += (prev_end - start_dt).total_seconds() * 10
        if next_start is not None and end_dt > next_start:
            score += (end_dt - next_start).total_seconds() * 10
        candidates.append((score, start_dt, end_dt, note))

    start_second = raw.start.second
    end_second = raw.end.second
    end_minute_base = raw.end.replace(second=0, microsecond=0)
    start_minute_base = raw.start.replace(second=0, microsecond=0)

    for offset in range(-10, 11):
        adjusted_start = end_minute_base + timedelta(minutes=offset, seconds=start_second)
        add_candidate(adjusted_start, raw.end, "start minute inferred from end time and neighboring rows")

        adjusted_end = start_minute_base + timedelta(minutes=offset, seconds=end_second)
        add_candidate(raw.start, adjusted_end, "end minute inferred from start time and neighboring rows")

    swapped_start, swapped_end = raw.end, raw.start
    add_candidate(swapped_start, swapped_end, "start/end swapped because end was earlier than start")

    if candidates:
        _, start_dt, end_dt, note = min(candidates, key=lambda item: item[0])
        return start_dt, end_dt, note
    return swapped_start, swapped_end, "start/end swapped because end was earlier than start"


def parse_remark_range(
    remark: str,
    test_date: date,
    prev_end: datetime | None = None,
    next_start: datetime | None = None,
) -> ParsedRange | None:
    raw = raw_remark_range(remark, test_date)
    if raw is None:
        return None

    start_dt = raw.start
    end_dt = raw.end

    notes: list[str] = []
    status = "ok"
    if not raw.start_had_seconds or not raw.end_had_seconds:
        status = "time_inferred"
        notes.append("seconds completed from minute-level remark")

    if end_dt < start_dt:
        start_dt, end_dt, note = infer_reversed_range(raw, prev_end, next_start)
        status = "time_inferred"
        notes.append(note)

    if end_dt == start_dt:
        return None
    return ParsedRange(start_dt, end_dt, status, "; ".join(notes))


def normal_neighbor_ranges(records: list[WorkbookRow], test_date: date) -> dict[tuple[str, int], RawRange]:
    ranges: dict[tuple[str, int], RawRange] = {}
    for record in records:
        raw = raw_remark_range(record.remark, test_date)
        if raw is not None and raw.end >= raw.start:
            ranges[(record.sheet, record.row_number)] = raw
    return ranges


def neighbor_bounds(
    record: WorkbookRow,
    records: list[WorkbookRow],
    raw_ranges: dict[tuple[str, int], RawRange],
) -> tuple[datetime | None, datetime | None]:
    same_sheet = [item for item in records if item.sheet == record.sheet]
    current_idx = same_sheet.index(record)

    prev_end = None
    for previous in reversed(same_sheet[:current_idx]):
        raw = raw_ranges.get((previous.sheet, previous.row_number))
        if raw is not None:
            prev_end = raw.end
            break

    next_start = None
    for following in same_sheet[current_idx + 1 :]:
        raw = raw_ranges.get((following.sheet, following.row_number))
        if raw is not None:
            next_start = raw.start
            break

    return prev_end, next_start


def parse_dat_timestamp(path: Path) -> datetime | None:
    match = DAT_RE.search(path.name)
    if not match:
        return None
    return datetime.strptime(match.group("stamp"), "%Y%m%d%H%M%S")


def find_candidate_files(data_root: Path, start: datetime, end: datetime) -> list[DatFile]:
    candidates: list[DatFile] = []
    current = start.replace(second=0, microsecond=0)
    end_minute = end.replace(second=0, microsecond=0)
    seen_dirs: set[Path] = set()

    while current <= end_minute:
        minute_dir = data_root / f"{current:%H}" / f"{current:%M}"
        if minute_dir not in seen_dirs and minute_dir.exists():
            for path in minute_dir.glob("*.dat"):
                timestamp = parse_dat_timestamp(path)
                if timestamp is not None and start <= timestamp <= end:
                    candidates.append(DatFile(path, timestamp))
        seen_dirs.add(minute_dir)
        current += timedelta(minutes=1)

    return sorted(candidates, key=lambda item: item.timestamp)


def choose_middle_files(files: list[DatFile], start: datetime, end: datetime, count: int) -> list[DatFile]:
    if len(files) <= count:
        return files

    midpoint = start + (end - start) / 2
    nearest_index = min(range(len(files)), key=lambda idx: abs(files[idx].timestamp - midpoint))
    left = nearest_index - count // 2
    right = left + count
    if left < 0:
        left = 0
        right = count
    if right > len(files):
        right = len(files)
        left = right - count
    return files[left:right]


def format_value(value: object) -> str:
    number = float(str(value).strip())
    if number.is_integer():
        return str(int(number))
    return ("%g" % number).replace(".", "p")


def safe_sheet_name(sheet_name: str) -> str:
    if "100" in sheet_name:
        return "100ns"
    if "40" in sheet_name:
        return "40ns"
    return re.sub(r"[^\w.-]+", "_", sheet_name, flags=re.UNICODE).strip("_")


def output_name(record: WorkbookRow, parsed: ParsedRange, point: int) -> str:
    sheet = safe_sheet_name(record.sheet)
    freq = format_value(record.frequency_hz)
    pzt = format_value(record.pzt_mvpp)
    return (
        f"{sheet}_freq_{freq}Hz_PZT_{pzt}mVpp_point_{point}_"
        f"{parsed.start:%H%M%S}_{parsed.end:%H%M%S}.png"
    )


def parse_sheet_points(items: list[str] | None) -> dict[str, int]:
    sheet_points = dict(DEFAULT_SHEET_POINTS)
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Invalid --sheet-point value, expected SHEET=POINT: {item}")
        sheet, point_text = item.split("=", 1)
        sheet_points[sheet.strip()] = int(point_text.strip())
    return sheet_points


def point_for_record(record: WorkbookRow, default_point: int, sheet_points: dict[str, int]) -> int:
    return sheet_points.get(record.sheet, default_point)


def default_export_python(project_root: Path) -> Path:
    venv_python = project_root / ".venv-build" / "Scripts" / "python.exe"
    return venv_python if venv_python.exists() else Path(sys.executable)


def run_export(
    export_python: Path,
    export_script: Path,
    files: list[DatFile],
    point: int,
    output: Path,
    width: int,
    height: int,
) -> subprocess.CompletedProcess[str]:
    command = [
        str(export_python),
        str(export_script),
        *[str(item.path) for item in files],
        "--point",
        str(point),
        "--output",
        str(output),
        "--width",
        str(width),
        "--height",
        str(height),
    ]
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    return subprocess.run(command, text=True, capture_output=True, env=env)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch export DAS point-analysis images from workbook remarks.")
    parser.add_argument("--excel", type=Path, default=DEFAULT_EXCEL, help="Input DAS test workbook.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Root folder containing HH/MM dat files.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for exported images and summary.csv.")
    parser.add_argument("--sheet", action="append", dest="sheets", default=None, help="Sheet to process; repeatable.")
    parser.add_argument("--point", type=int, default=799, help="Default viewer/0-based point index. Use 799 for the 800th point.")
    parser.add_argument(
        "--sheet-point",
        action="append",
        default=None,
        help="Per-sheet viewer/0-based point override, e.g. 40ns脉宽=1997. Repeatable.",
    )
    parser.add_argument("--window-seconds", type=int, default=10, help="Number of middle seconds/files to export.")
    parser.add_argument("--date", default=DEFAULT_TEST_DATE.isoformat(), help="Test date in YYYY-MM-DD.")
    parser.add_argument("--export-python", type=Path, default=None, help="Python executable used to run point_analysis_export.py.")
    parser.add_argument("--export-script", type=Path, default=Path("point_analysis_export.py"), help="Point export script.")
    parser.add_argument("--width", type=int, default=1200, help="Export image width.")
    parser.add_argument("--height", type=int, default=850, help="Export image height.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of rows to export.")
    parser.add_argument("--dry-run", action="store_true", help="Only select files and write summary.csv; do not export images.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    source_dir = Path(__file__).resolve().parent
    project_root = source_dir.parent
    excel_path = args.excel if args.excel.is_absolute() else project_root / args.excel
    data_root = args.data_root
    output_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    export_script = args.export_script if args.export_script.is_absolute() else source_dir / args.export_script
    export_python = args.export_python or default_export_python(project_root)
    test_date = date.fromisoformat(args.date)
    sheets = tuple(args.sheets or DEFAULT_SHEETS)
    sheet_points = parse_sheet_points(args.sheet_point)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    records = iter_workbook_rows(excel_path, sheets)
    if args.limit is not None:
        records = records[: args.limit]
    raw_ranges = normal_neighbor_ranges(records, test_date)

    fieldnames = [
        "sheet",
        "row",
        "frequency_hz",
        "pzt_mvpp",
        "point",
        "remark",
        "parsed_start",
        "parsed_end",
        "selected_files",
        "output",
        "status",
        "note",
    ]

    exported = 0
    skipped = 0
    with summary_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for record in records:
            prev_end, next_start = neighbor_bounds(record, records, raw_ranges)
            parsed = parse_remark_range(record.remark, test_date, prev_end, next_start)
            base_row = {
                "sheet": record.sheet,
                "row": record.row_number,
                "frequency_hz": record.frequency_hz,
                "pzt_mvpp": record.pzt_mvpp,
                "point": point_for_record(record, args.point, sheet_points),
                "remark": record.remark,
                "parsed_start": "",
                "parsed_end": "",
                "selected_files": "",
                "output": "",
                "status": "",
                "note": "",
            }

            if parsed is None:
                skipped += 1
                writer.writerow(base_row | {"status": "skipped", "note": "no usable time range"})
                continue

            files = choose_middle_files(
                find_candidate_files(data_root, parsed.start, parsed.end),
                parsed.start,
                parsed.end,
                args.window_seconds,
            )
            selected = ";".join(str(item.path) for item in files)
            point = point_for_record(record, args.point, sheet_points)
            output = output_dir / safe_sheet_name(record.sheet) / output_name(record, parsed, point)

            row = base_row | {
                "parsed_start": parsed.start.isoformat(sep=" "),
                "parsed_end": parsed.end.isoformat(sep=" "),
                "selected_files": selected,
                "output": str(output),
                "status": parsed.status,
                "note": parsed.note,
            }

            if len(files) < args.window_seconds:
                skipped += 1
                writer.writerow(row | {"status": "skipped", "note": f"only {len(files)} files found; {parsed.note}".strip("; ")})
                continue

            if args.dry_run:
                writer.writerow(row | {"status": f"dry_run_{parsed.status}"})
                exported += 1
                continue

            output.parent.mkdir(parents=True, exist_ok=True)
            result = run_export(export_python, export_script, files, point, output, args.width, args.height)
            if result.returncode != 0:
                skipped += 1
                note = (result.stderr or result.stdout).strip().replace("\r", " ").replace("\n", " | ")
                writer.writerow(row | {"status": "failed", "note": note})
                continue

            exported += 1
            writer.writerow(row | {"status": f"exported_{parsed.status}"})
            print(f"exported: {output}")

    print(f"records: {len(records)}, exported/planned: {exported}, skipped/failed: {skipped}")
    print(f"summary: {summary_path}")
    return 0 if exported > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
