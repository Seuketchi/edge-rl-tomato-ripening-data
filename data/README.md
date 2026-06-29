# Hardware-Run Telemetry

Telemetry from the two unattended proof-of-concept hardware runs, logged once per
minute by the ESP32-S3 firmware over WiFi to the cloud ingest service and retained
on-device as a redundant copy. Column meanings are documented in the manuscript
(`../Manuscript.pdf`, telemetry schema table in Chapter 3).

- **Run 1** — five-day run (14:13 PHT 2026-06-07 → 23:21 PHT 2026-06-13), RGB565 capture path.
- **Run 2** — continuous re-run (00:00 PHT 2026-06-22 → 21:27 PHT 2026-06-26), corrected RGB888 capture path.

## Files

| File | Rows | Description |
|------|------|-------------|
| `demo-2026-06-10.csv` | 7,418 | **Run 1** cleaned series (run `demo-2026-06-10`), used for the Run 1 figures and statistics. |
| `demo-2026-06-22.csv` | 7,045 | **Run 2** continuous series (run `demo-2026-06-22`, RGB888 path). Uninterrupted, so cleaning drops no rows — used directly for the Run 2 figure and statistics. |

The two Run 1 recording gaps (a ~28 h grid brownout on 06-08→06-09 and a ~1 h
power-cycle for the Q1-MOSFET replacement on 06-11) are *not* errors; they are
auditable outage windows, left visible and correctly ordered by the firmware's
dual-timestamp design. Run 2 has no such gaps.

## Cleaning

`demo-2026-06-10.csv` is the cleaned Run 1 series, produced by
`../scripts/clean_demo_run.py`, which drops 51 rows from the original cloud
capture (7,469 → 7,418) under four rules:

1. **Stale reconnect** — duplicate/stale rows emitted on link recovery.
2. **Isolated spurious-ripe** — a lone `harvest`/ripe flip surrounded by unripe rows.
3. **Saturated chroma** — `chromatic_x >= 0.99` (camera saturation during a manual chamber inspection).
4. **Temperature dropout** — `temp <= 0` (transient I²C sensor dropout).

Run 2 was uninterrupted and contains none of these, so the script removes nothing.

## Citable record

These CSV files, version-controlled in this repository
(<https://github.com/Seuketchi/edge-rl-tomato-ripening-data>) under `data/`, are
the canonical, citable record of the run telemetry. They are self-contained and
do not depend on any external service. A cloud dashboard was used for live
monitoring during the runs but is a convenience mirror only, not guaranteed to
remain online.
