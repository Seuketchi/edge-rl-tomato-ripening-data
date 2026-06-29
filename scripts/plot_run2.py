#!/usr/bin/env python3
"""Plot the Run 2 hardware demo run (June 22-26 2026) for Figure (fig:run2_panels).

Recorded telemetry from the continuous, uninterrupted second closed-loop run on
the corrected RGB888 capture path. Same generator and layout as the Run 1 figure
(scripts/plot_demo_run.py); only the input CSV, output path, and Run-2-specific
labels differ. Run 2 has no outage and no MOSFET swap, so the break/intra-gap
annotations from the Run 1 plotter auto-skip (single continuous segment).

Three stacked panels on a shared, plain-language time axis:
  (top)    Tomato ripeness — camera colour index X (flat: RGB888 still compressed)
  (mid)    Chamber temperature vs. the controller's target setpoint
  (bottom) What the controller decided — a colour-coded timeline strip

Readability choices:
  * Plain-language titles; explanatory detail lives in the figure caption
    (printed to stdout by this script), not crammed onto the plot.
  * The decision panel is a Gantt-style colour strip (blue=cool, grey=maintain,
    red=heat) rather than an abstract three-level scatter.
"""

from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
CSV  = ROOT / "data/demo-2026-06-22.csv"
OUT  = ROOT / "figures/run2_perceived_panels.png"

ACOLOR = {"cool": "#2c7fb8", "maintain": "#95a5a6", "heat": "#c0392b"}
ALABEL = {"cool": "Cool", "maintain": "Maintain", "heat": "Heat"}

SEG_GAP_HOURS = 6.0    # a recording break longer than this splits the axis
BREAK_WIDTH   = 0.18   # plot-days of blank inserted at each break
HARVEST_X     = 0.15


def load(path: Path):
    rows, header = [], None
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if header is None and parts[0] == "seq":
                header = parts
                continue
            rows.append(parts)
    idx = {name: i for i, name in enumerate(header)}
    return rows, idx


def col(rows, idx, name, cast=float):
    out = []
    for r in rows:
        try:
            out.append(cast(r[idx[name]]))
        except (ValueError, IndexError, KeyError):
            out.append(None)
    return out


def main():
    rows, idx = load(CSV)

    times_raw = [
        datetime.strptime(r[idx["datetime_pht"]], "%Y-%m-%d %H:%M:%S")
        if r[idx["datetime_pht"]] else None
        for r in rows
    ]
    keep  = [k for k, t in enumerate(times_raw) if t is not None]
    times = [times_raw[k] for k in keep]
    sub   = [rows[k] for k in keep]

    temp   = col(sub, idx, "temp")
    setp   = col(sub, idx, "setpoint")
    esp    = col(sub, idx, "esp_temp")
    x_raw  = col(sub, idx, "chromatic_x")
    vvalid = col(sub, idx, "vision_valid", int)
    x      = [xi if (vi == 1 and xi is not None) else None
              for xi, vi in zip(x_raw, vvalid)]
    action = [r[idx["action"]] for r in sub]

    # --- Compress recording breaks: build a piecewise time->plot-x map. ---
    # Each contiguous run of data is laid end-to-end with a small fixed gap
    # between runs, so the 28 h outage shrinks to one labelled break.
    segs, s = [], 0
    for k in range(1, len(times)):
        if (times[k] - times[k - 1]).total_seconds() / 3600.0 > SEG_GAP_HOURS:
            segs.append((s, k - 1))
            s = k
    segs.append((s, len(times) - 1))

    seg_off, off = [], 0.0
    for a, b in segs:
        seg_off.append(off)
        off += (times[b] - times[a]).total_seconds() / 86400.0 + BREAK_WIDTH

    px = [0.0] * len(times)
    for (a, b), o in zip(segs, seg_off):
        for i in range(a, b + 1):
            px[i] = o + (times[i] - times[a]).total_seconds() / 86400.0

    # Intra-segment downtime (0.5-6 h): not big enough to split the axis, but worth
    # marking — the ~1 h Jun-11 power-cycle is the Q1 heater-MOSFET replacement
    # (esp_temp dropped 55->37 C across it = a deliberate shutdown).
    intra_gaps = []
    for k in range(1, len(times)):
        dh = (times[k] - times[k - 1]).total_seconds() / 3600.0
        if 0.5 < dh <= SEG_GAP_HOURS:
            intra_gaps.append((px[k - 1] + px[k]) / 2.0)

    def map_time(t):
        for (a, b), o in zip(segs, seg_off):
            if times[a] <= t <= times[b]:
                return o + (t - times[a]).total_seconds() / 86400.0
        return None

    # Rolling median of X (per segment) for an honest flat trend line.
    import statistics

    def seg_rolling_median(a, b, win=25):
        vals = x[a:b + 1]
        half = win // 2
        out = []
        for i in range(len(vals)):
            w = [v for v in vals[max(0, i - half):i + half + 1] if v is not None]
            out.append(statistics.median(w) if w else None)
        return out

    # --- figure ---
    # Panels: ripeness, chamber temp, ESP die-temp (compact), decision strip.
    fig, (ax0, ax1, axe, axa) = plt.subplots(
        4, 1, figsize=(9.2, 8.8), sharex=True,
        gridspec_kw={"hspace": 0.34, "height_ratios": [1.0, 1.4, 0.55, 0.5]},
    )
    panels = (ax0, ax1, axe, axa)

    # Night shading (18:00-06:00) on the temp panel only, mapped to plot-x.
    for a, b in segs:
        start = None
        for i in range(a, b + 1):
            night = times[i].hour >= 18 or times[i].hour < 6
            if night and start is None:
                start = px[i]
            if (not night or i == b) and start is not None:
                ax1.axvspan(start, px[i], color="#34495e", alpha=0.06, zorder=0)
                start = None

    # Panel 1 — Ripeness
    ax0.scatter(px, x, s=2, color="#d08a3e", alpha=0.30, linewidths=0)
    for (a, b) in segs:
        med = seg_rolling_median(a, b)
        ax0.plot(px[a:b + 1], med, color="#9c5511", lw=1.6,
                 label="ripeness (smoothed)" if (a, b) == segs[0] else None)
    ax0.axhline(HARVEST_X, ls="--", lw=1.0, color="#777",
                label=f"harvest threshold ($X$={HARVEST_X})")
    ax0.set_ylim(0.0, 1.0)
    ax0.set_ylabel("Color index $X$", fontsize=9.5)
    ax0.set_title("Tomato ripeness (camera, RGB888) — still compressed: never reaches harvest threshold",
                  fontsize=10.5, loc="left", pad=6)
    ax0.legend(loc="upper right", fontsize=8, frameon=False, ncol=2)

    # Panel 2 — Temperature vs target
    for (a, b) in segs:
        first = (a, b) == segs[0]
        ax1.plot(px[a:b + 1], temp[a:b + 1], color="#c0392b", lw=1.3,
                 label="chamber temperature" if first else None)
        ax1.plot(px[a:b + 1], setp[a:b + 1], color="#2c7fb8", lw=1.2, ls="--",
                 label="target setpoint" if first else None)
    ax1.set_ylabel("Temperature (°C)", fontsize=9.5)
    ax1.set_title("Chamber temperature vs. controller target — TEC can't reach the cold target",
                  fontsize=10.5, loc="left", pad=6)
    ax1.legend(loc="upper right", fontsize=8, frameon=False, ncol=2)

    # Panel 3 — ESP32 die temperature (self-heating; compact)
    for (a, b) in segs:
        axe.plot(px[a:b + 1], esp[a:b + 1], color="#8e44ad", lw=1.1)
    axe.set_ylabel("ESP die (°C)", fontsize=9.5)
    axe.set_title("ESP32 on-die temperature — sustained self-heating",
                  fontsize=10.5, loc="left", pad=6)

    # Panel 4 — Decision timeline strip (contiguous same-action runs per segment)
    for (a, b) in segs:
        i = a
        while i <= b:
            j = i
            while j + 1 <= b and action[j + 1] == action[i]:
                j += 1
            c = ACOLOR.get(action[i], "#ccc")
            width = max(px[j] - px[i], 0.004)
            axa.broken_barh([(px[i], width)], (0.0, 1.0), facecolors=c, edgecolor="none")
            i = j + 1
    axa.set_ylim(0.0, 1.0)
    axa.set_yticks([])
    axa.set_title("What the controller decided each step", fontsize=10.5, loc="left", pad=6)
    patches = [mpatches.Patch(color=ACOLOR[k], label=ALABEL[k])
               for k in ("cool", "maintain", "heat")]
    axa.legend(handles=patches, loc="upper center", bbox_to_anchor=(0.5, -0.45),
               fontsize=8.5, frameon=False, ncol=3)

    # --- break markers + outage label across all panels ---
    for k in range(1, len(segs)):
        bx = seg_off[k] - BREAK_WIDTH / 2.0
        gap_h = (times[segs[k][0]] - times[segs[k - 1][1]]).total_seconds() / 3600.0
        for ax in panels:
            ax.axvspan(bx - BREAK_WIDTH / 2, bx + BREAK_WIDTH / 2,
                       color="#ffffff", zorder=2)
            for off_x in (-BREAK_WIDTH / 2, BREAK_WIDTH / 2):
                ax.plot([bx + off_x - 0.02, bx + off_x + 0.02],
                        [ax.get_ylim()[0], ax.get_ylim()[0]], color="k", lw=0)
            ax.text(bx, ax.get_ylim()[1], "∥", ha="center", va="top",
                    fontsize=11, color="#555", zorder=3)
        if gap_h > 12:
            ax1.text(bx, sum(ax1.get_ylim()) / 2,
                     f"≈{gap_h:.0f} h\npower\noutage", ha="center", va="center",
                     fontsize=7.5, color="#6c3483", zorder=3,
                     bbox=dict(boxstyle="round,pad=0.25", fc="#f4ecf7", ec="#b59bc9", lw=0.6))

    # Mark intra-segment downtime (the Q1 heater-MOSFET replacement, Jun-11).
    for gi, gx in enumerate(intra_gaps):
        for ax in panels:
            ax.axvline(gx, color="#2c3e50", lw=0.9, ls=(0, (4, 2)), alpha=0.8, zorder=2)
        ax1.text(gx, ax1.get_ylim()[1], " Q1 MOSFET replaced", rotation=90,
                 ha="right", va="top", fontsize=6.8, color="#2c3e50", zorder=3)

    # --- date ticks at local midnights, mapped to plot-x ---
    ticks, labels = [], []
    day = times[0].replace(hour=0, minute=0, second=0) + timedelta(days=1)
    while day <= times[-1]:
        mx = map_time(day)
        if mx is not None:
            ticks.append(mx)
            labels.append(day.strftime("%b %d"))
        day += timedelta(days=1)
    axa.set_xticks(ticks)
    axa.set_xticklabels(labels, fontsize=9)
    axa.set_xlabel("Date (PHT, 2026) — midnight separators (continuous run, no breaks)",
                   fontsize=9.5)
    for ax in panels:
        ax.set_xlim(-0.1, off - BREAK_WIDTH + 0.1)
        # Day separators at each midnight tick (clearer than a faint grid).
        for tx in ticks:
            ax.axvline(tx, color="#bbb", lw=0.7, alpha=0.55, zorder=0)
    for ax in (ax0, ax1, axe):
        ax.grid(True, axis="y", alpha=0.18, lw=0.5)

    fig.align_ylabels(panels)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")

    # --- suggested caption (paste into the manuscript) ---
    ac = Counter(a for a in action if a in ACOLOR)
    tot = sum(ac.values()) or 1
    xv = [v for v in x if v is not None]
    print(f"wrote {OUT}")
    print(f"rows : {len(sub)}  span: {times[0]} → {times[-1]}  segments: {len(segs)}")
    esp_v = sorted(v for v in esp if v is not None)
    # Operating band (ignore the reboot cold-start transient at the power-cycle).
    esp_lo = esp_v[int(0.25 * len(esp_v))] if esp_v else 0
    esp_hi = esp_v[int(0.99 * (len(esp_v) - 1))] if esp_v else 0
    print("\n--- suggested caption ---")
    print(
        f"Figure. Recorded Run 2 telemetry — the continuous, uninterrupted second "
        f"closed-loop demonstration (PHT). Panel 1 (ripeness): the camera index X stays "
        f"compressed at {min(xv):.2f}-{max(xv):.2f} and never reaches the {HARVEST_X} "
        f"harvest threshold — the RGB888 capture path removed the quantisation compressor "
        f"but the uncorrected white balance leaves X pinned near 0.5 (a perception gap; "
        f"harvest is decided on the model ODE state, not this signal). Panel 2 (chamber "
        f"temperature): follows the diurnal cycle while the target setpoint sits near the "
        f"12.5 °C floor — the undersized TEC cannot close the gap against tropical ambient "
        f"(an actuator-sizing limit, not a policy error). Panel 3 (ESP32 die temperature): "
        f"sustained {esp_lo:.0f}-{esp_hi:.0f} °C self-heating with continuous camera "
        f"capture, motivating the thermal-hardening recommendation. Panel 4 (decision): "
        f"per-step controller action ({100*ac['cool']/tot:.1f}% cool / "
        f"{100*ac['heat']/tot:.1f}% heat / {100*ac['maintain']/tot:.1f}% maintain across "
        f"{tot:,} steps). The run is unbroken: no power outage and no MOSFET replacement, "
        f"unlike Run 1."
    )


if __name__ == "__main__":
    main()
