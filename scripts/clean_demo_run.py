#!/usr/bin/env python3
"""
Clean a demo-run telemetry CSV before analysis or plotting.

Rules (from docs/thesis/demo-run-notes.md):
  1. Drop stale reconnect rows: vision_valid=0 OR chromatic_x=0.0, detected
     immediately after a sequence gap OR at run start.
  2. Drop isolated spurious-ripe rows: class=ripe where BOTH the previous
     and next non-dropped rows are class=unripe (single-row RGB565 noise flip).
  3. Drop saturated-X rows: chromatic_x >= 0.99. The RGB565 perception path
     caps a real chromatic index at ~0.66, so a pegged ~1.0 is non-physical
     camera saturation (e.g. the chamber opened during the 2026-06-09 restart
     after the earthquake brownout flooded the lens with ambient light).
  4. Drop temperature-sensor dropout rows: temp <= 0 °C. The chamber never runs
     below its ~12.5 °C setpoint floor, so 0.0 °C is an I2C read failure (returns
     0.0), seen on reconnect (e.g. the 2026-06-11 06:12 block after a blip).

Usage:
    python scripts/clean_demo_run.py outputs/experiments/demo-2026-06-10.csv
    python scripts/clean_demo_run.py outputs/experiments/demo-2026-06-10.csv --out outputs/experiments/demo-2026-06-10_clean.csv
"""

import argparse
import csv
import io
import sys
from pathlib import Path

# Column indices (0-based) matching the cloud /download CSV header.
COL_SEQ         = 0
COL_TEMP        = 3
COL_CHROMATIC_X = 8
COL_CLASS       = 10
COL_VISION_VALID = 15


def load_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    """Return (comment_lines, data_rows). Header row is in data_rows[0]."""
    comments, rows = [], []
    with open(path, newline="") as f:
        for line in f:
            if line.startswith("#"):
                comments.append(line.rstrip("\n"))
            else:
                rows.append(next(csv.reader(io.StringIO(line))))
    return comments, rows


def is_stale(row: list[str]) -> bool:
    try:
        vv = row[COL_VISION_VALID]
        cx = row[COL_CHROMATIC_X]
        return vv == "0" or cx == "0.0"
    except IndexError:
        return False


def is_saturated(row: list[str]) -> bool:
    try:
        return float(row[COL_CHROMATIC_X]) >= 0.99
    except (IndexError, ValueError):
        return False


def is_temp_dropout(row: list[str]) -> bool:
    # The chamber never runs below the ~12.5 °C setpoint floor; a temp <= 0 °C is a
    # sensor dropout (I2C read fails -> 0.0), typically on reconnect.
    try:
        return float(row[COL_TEMP]) <= 0.0
    except (IndexError, ValueError):
        return False


def clean(rows: list[list[str]]) -> tuple[list[list[str]], dict]:
    header = rows[0]
    data   = rows[1:]

    # --- Rule 1: stale reconnect rows ---
    stale_indices = {i for i, r in enumerate(data) if is_stale(r)}

    # --- Rule 3: saturated-X rows (non-physical, X >= 0.99) ---
    saturated = {i for i, r in enumerate(data) if is_saturated(r)}

    # --- Rule 4: temperature-sensor dropout rows (temp <= 0 °C) ---
    temp_dropout = {i for i, r in enumerate(data) if is_temp_dropout(r)}

    # --- Rule 2: isolated spurious-ripe rows ---
    # Build class list, skipping already-dropped (stale/saturated/dropout) rows.
    pre_dropped = stale_indices | saturated | temp_dropout
    active = [(i, r) for i, r in enumerate(data) if i not in pre_dropped]
    spurious = set()
    for pos, (i, r) in enumerate(active):
        if r[COL_CLASS] != "ripe":
            continue
        prev_class = active[pos - 1][1][COL_CLASS] if pos > 0 else None
        next_class = active[pos + 1][1][COL_CLASS] if pos < len(active) - 1 else None
        if prev_class == "unripe" and next_class == "unripe":
            spurious.add(i)

    dropped = stale_indices | saturated | temp_dropout | spurious
    clean_rows = [header] + [r for i, r in enumerate(data) if i not in dropped]

    stats = {
        "total_input":  len(data),
        "stale_dropped": len(stale_indices),
        "saturated_dropped": len(saturated),
        "temp_dropout_dropped": len(temp_dropout),
        "spurious_ripe_dropped": len(spurious),
        "total_dropped": len(dropped),
        "total_output": len(clean_rows) - 1,
    }
    return clean_rows, stats


def write_csv(path: Path, comments: list[str], rows: list[list[str]]) -> None:
    with open(path, "w", newline="") as f:
        for c in comments:
            f.write(c + "\n")
        w = csv.writer(f)
        for r in rows:
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser(description="Clean demo-run telemetry CSV")
    ap.add_argument("input", help="raw CSV from /download endpoint")
    ap.add_argument("--out", help="output path (default: <input>_clean.csv)")
    args = ap.parse_args()

    src = Path(args.input)
    dst = Path(args.out) if args.out else src.with_stem(src.stem + "_clean")

    comments, rows = load_csv(src)
    clean_rows, stats = clean(rows)
    write_csv(dst, comments, clean_rows)

    print(f"Input rows  : {stats['total_input']}")
    print(f"  stale dropped          : {stats['stale_dropped']}")
    print(f"  saturated (X>=0.99)    : {stats['saturated_dropped']}")
    print(f"  temp dropout (T<=0)    : {stats['temp_dropout_dropped']}")
    print(f"  spurious ripe dropped  : {stats['spurious_ripe_dropped']}")
    print(f"  ─────────────────────")
    print(f"  total dropped          : {stats['total_dropped']}")
    print(f"Output rows : {stats['total_output']}")
    print(f"Written to  : {dst}")


if __name__ == "__main__":
    main()
