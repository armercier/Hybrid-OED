"""One-page sheet: how to time a multi-source LDV scan (excitation script + Polytec macro) safely.

Generic timeline with an illustrative example (scan time 6 h +/- 30 min, three sources shown, the real
scans 25 min shorter / longer than expected) built from src/ldv_pmma/scan_schedule.py.

Usage:  python reports/2026-11_specimen/make_schedule_sheet.py
Output: reports/2026-11_specimen/ldv_scan_schedule_sheet.{pdf,png}
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src" / "ldv_pmma"))
from scan_schedule import check, schedule  # noqa: E402

COL = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
D, DELTA = 6 * 3600.0, 1800.0
REAL = [D, D - 1500, D + 1500]          # illustrative real scan durations (within tolerance)


def bracket(ax, x0, x1, y, text, color="k"):
    ax.annotate("", (x0, y), (x1, y), arrowprops=dict(arrowstyle="<->", color=color, lw=0.9, shrinkA=0, shrinkB=0))
    ax.text((x0 + x1) / 2, y + 0.07, text, ha="center", va="bottom", fontsize=8, color=color)


def timeline(ax, s):
    M, G, W = s["safety_margin"], s["delay_before_next_trigger"], s["ldv_wait"]
    h = lambda x: x / 3600
    ys, yl = 1.0, 0.0
    t = 0.0
    starts = []
    for k in range(3):
        starts.append(t)
        ax.add_patch(Rectangle((h(t), ys - 0.22), h(D + M), 0.44, fc=COL[k], ec="k", lw=0.5))
        ax.text(h(t + (D + M) / 2), ys, f"S{k + 1} firing (trigger on)", ha="center", va="center", color="w",
                fontsize=8, fontweight="bold")
        t += D + M
        if k < 2:
            ax.add_patch(Rectangle((h(t), ys - 0.22), h(G), 0.44, fc="#eee", ec="k", lw=0.5, hatch="//"))
            ax.text(h(t + G / 2), ys + 0.3, "trigger off", ha="center", fontsize=7)
            t += G
    # LDV lane: scan k starts with source k; real duration REAL[k]; then Wait W; then waits for trigger
    for k in range(3):
        a = starts[k]
        b = a + REAL[k]
        ax.add_patch(Rectangle((h(a), yl - 0.22), h(REAL[k]), 0.44, fc=COL[k], ec="k", lw=0.5, alpha=0.85))
        ax.text(h((a + b) / 2), yl, f"LDV scan {k + 1}" + ("" if k == 0 else f"  ({(REAL[k] - D) / 60:+.0f} min)"),
                ha="center", va="center", fontsize=8, color="w", fontweight="bold")
        if k < 2:
            ax.add_patch(Rectangle((h(b), yl - 0.22), h(W), 0.44, fc="#f4f4f4", ec="k", lw=0.5))
            ax.text(h(b + W / 2), yl, "Wait W", ha="center", va="center", fontsize=7.5)
            nxt = starts[k + 1]
            ax.add_patch(Rectangle((h(b + W), yl - 0.22), h(nxt - b - W), 0.44, fc="w", ec="k", lw=0.5, ls=":"))
            ax.text(h((b + W + nxt) / 2), yl - 0.36, "waits for\ntrigger", ha="center", va="top", fontsize=6.5)
            ax.plot([h(nxt), h(nxt)], [yl - 0.25, ys + 0.25], color="k", lw=0.7, ls="--")
    # dimensions on source 1
    bracket(ax, 0, h(D), -0.62, "D = expected scan time")
    bracket(ax, h(D), h(D + M), -0.62, "M")
    bracket(ax, h(D + M), h(D + M + G), 1.55, "G = trigger-off gap")
    bracket(ax, h(REAL[0]), h(REAL[0] + W), -0.95, "W = M + G/2")
    ax.set_yticks([yl, ys], ["LDV macro", "Excitation\nscript"], fontsize=9)
    ax.set_ylim(-1.15, 1.85)
    ax.set_xlim(0, h(t) * 1.005)
    ax.set_xlabel("time (h), illustrative example: D = 6 h +/- 30 min", fontsize=8.5)
    ax.tick_params(labelsize=8)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)


def box(fig, x, y, w, h, title, body, fs=8.6, mono=False):
    fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006", transform=fig.transFigure,
                                      fc="#f7f7f7", ec="#bbbbbb", lw=0.8))
    fig.text(x + 0.012, y + h - 0.012, title, fontsize=10, fontweight="bold", va="top")
    fig.text(x + 0.012, y + h - 0.038, body, fontsize=fs, va="top", family="monospace" if mono else None,
             linespacing=1.45)


def main():
    s = schedule(D, DELTA, n_sources=3)
    assert all(check(s, r) for r in REAL)
    fig = plt.figure(figsize=(8.27, 11.69))          # A4 portrait
    fig.text(0.06, 0.965, "Running a multi-source LDV scan without mixing up sources", fontsize=15, fontweight="bold")
    fig.text(0.06, 0.94, "PMMA plate setup, Nov 2026: PlaySamplesArnaud.py (excitation) + Auto_Arnaud.wwb (Polytec macro)",
             fontsize=8.5, color="#555")
    fig.text(0.06, 0.905,
             "The two programs run on separate clocks and only share the trigger. The LDV records a point only when a\n"
             "trigger arrives. So: keep each source on a little longer than the scan, switch the trigger off for a while,\n"
             "and make the LDV restart in the middle of that off-time. It then waits and starts exactly with the next source.",
             fontsize=9, va="top", linespacing=1.45)
    ax = fig.add_axes([0.13, 0.60, 0.83, 0.23])
    timeline(ax, s)
    fig.text(0.06, 0.555, "Scan 2 runs 25 min short and scan 3 runs 25 min long: both still start with their own source "
                          "(dashed lines),\nbecause the LDV waits for the trigger at every switch. Errors do not add up "
                          "from scan to scan.", fontsize=8.3, va="top", style="italic", linespacing=1.4)
    box(fig, 0.06, 0.305, 0.42, 0.205, "Rules",
        "D  expected scan time (points x s/point)\n"
        "dD  uncertainty on D (+/-)\n"
        "M  safety_margin     = f * dD\n"
        "G  delay_before_next_trigger = 2 f * dD\n"
        "W  LDV Wait          = M + G/2\n\n"
        "Safe if the real scan lasts between\n"
        "D - G/2 and D + min(M, G/2).\n"
        "f = 2 by default; G >= ~10 min so the\n"
        "LDV can save the file and re-arm.", fs=8.3, mono=True)
    box(fig, 0.52, 0.305, 0.42, 0.205, "Compute the settings",
        "python src/ldv_pmma/scan_schedule.py \\\n"
        "  --points 6418 --sec-per-point 4.0 \\\n"
        "  --uncertainty-pct 5\n\n"
        "or  --scan-time 21600 --uncertainty 1800\n\n"
        "It prints the three Python values,\n"
        "the macro Wait, the tolerance and the\n"
        "total duration.", fs=8.3, mono=True)
    box(fig, 0.06, 0.185, 0.88, 0.095, f"Example (this page): D = 6 h, dD = 30 min, f = 2",
        f"Python (PlaySamplesArnaud.py):   scan_time = {s['scan_time']:.0f}    safety_margin = {s['safety_margin']:.0f}    "
        f"delay_before_next_trigger = {s['delay_before_next_trigger']:.0f}\n"
        f"Macro (Auto_Arnaud.wwb, line 45):   Wait {s['ldv_wait']:.0f}\n"
        f"Tolerance per scan:  -{s['tolerance_minus'] / 60:.0f} / +{s['tolerance_plus'] / 60:.0f} min        "
        f"Total for 4 sources: {(4 * (D + s['safety_margin']) + 3 * s['delay_before_next_trigger']) / 3600:.1f} h", fs=8.3)
    box(fig, 0.06, 0.03, 0.88, 0.13, "Before launching",
        "1. Measure the time per point on a short test scan (same grid spacing and averages).\n"
        "2. Compute the settings and set the three Python values and the macro Wait.\n"
        "3. Start the LDV macro FIRST (it waits for the trigger), then the Python script.\n"
        "4. If possible, check the first switch: the next source must be firing when the LDV restarts.\n"
        "5. Zeroing loop: range(0, N_channels) gives single-source files; range(1, ...) keeps S1 on in every file.", fs=8.3)
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"ldv_scan_schedule_sheet.{ext}", dpi=120)
    print(HERE / "ldv_scan_schedule_sheet.pdf")


if __name__ == "__main__":
    main()
