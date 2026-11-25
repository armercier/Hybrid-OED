"""Graphical summary of the two pristine-scan acquisition bugs (companion of pristine_scan_bug_report.md).

A: timeline of the excitation script and the LDV macro during the pristine night (independent clocks
   drift apart: the LDV starts each scan while the previous source still fires), and the fixed schedule
   used for the defect scan.
B: the channel-zeroing bug (channel 0 = S1 never cleared).
C: where the wrong-source points are on the scan grid (ldv_work/prepared/pristine.npz).

Usage:  python reports/2026-11_specimen/make_bug_figure.py
Output: reports/2026-11_specimen/pristine_scan_bug_figure.{png,pdf}
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
COL = {1: "#4C72B0", 2: "#DD8452", 3: "#55A868", 4: "#C44E52"}
WRONG = "#d62728"


def timeline(ax, src_on, gap, scan, wait, n=4, title=""):
    """Simulate the two clocks. The LDV only records while the trigger is on."""
    y_src, y_ldv = 1.0, 0.0
    t = 0.0
    sources = []
    for k in range(1, n + 1):
        sources.append((k, t, t + src_on))
        ax.add_patch(Rectangle((t / 3600, y_src - 0.3), src_on / 3600, 0.6, fc=COL[k], ec="k", lw=0.5))
        ax.text((t + src_on / 2) / 3600, y_src, f"S{k} firing", ha="center", va="center", fontsize=8, color="w",
                fontweight="bold")
        t += src_on
        if k < n:
            ax.add_patch(Rectangle((t / 3600, y_src - 0.3), gap / 3600, 0.6, fc="#eee", ec="k", lw=0.5, hatch="//"))
            t += gap
    on = lambda x: next(((k, a, b) for k, a, b in sources if a <= x < b), None)
    t = 0.0
    wrong_total = []
    for k in range(1, n + 1):
        # the scan needs `scan` seconds of trigger-on time; it waits while the trigger is off
        start, need, x, segs = t, scan, t, []
        while need > 1e-6:
            s = on(x)
            if s is None:                                   # trigger off: wait for the next source
                nxt = min(a for _, a, _ in sources if a > x)
                if nxt - x > 600:
                    ax.add_patch(Rectangle((x / 3600, y_ldv - 0.3), (nxt - x) / 3600, 0.6, fc="w", ec="k", lw=0.5, ls=":"))
                    ax.text((x + nxt) / 2 / 3600, y_ldv, "waits for\ntrigger", ha="center", va="center", fontsize=6.5)
                x = nxt
                continue
            dur = min(need, s[2] - x)
            segs.append((x, x + dur, s[0]))
            need -= dur
            x += dur
        for a, b, kk in segs:
            ok = kk == k
            ax.add_patch(Rectangle((a / 3600, y_ldv - 0.3), (b - a) / 3600, 0.6, fc=COL[k] if ok else WRONG,
                                   ec="k", lw=0.5, alpha=0.9))
            if not ok:
                wrong_total.append((k, kk, b - a))
                ax.annotate(f"file {k} recorded\nwith S{kk}: {(b - a) / 60:.0f} min", ((a + b) / 2 / 3600, y_ldv - 0.3),
                            xytext=((a + b) / 2 / 3600, y_ldv - 0.95), ha="center", fontsize=7.5, color=WRONG,
                            arrowprops=dict(arrowstyle="->", color=WRONG, lw=0.8))
        ax.text((segs[0][0] + segs[-1][1]) / 2 / 3600, y_ldv + 0.42, f"LDV scan {k} -> excitation_{k}.svd", ha="center",
                fontsize=7.5)
        t = x
        if k < n:
            ax.add_patch(Rectangle((t / 3600, y_ldv - 0.3), wait / 3600, 0.6, fc="#f6f6f6", ec="k", lw=0.5))
            ax.text((t + wait / 2) / 3600, y_ldv, "wait", ha="center", va="center", fontsize=7)
            t += wait
    ax.set_yticks([y_ldv, y_src], ["LDV macro\n(Auto_Arnaud.wwb)", "Excitation script\n(PlaySamplesArnaud.py)"])
    ax.set_ylim(-1.35, 1.5)
    ax.set_xlim(0, t / 3600 * 1.01)
    ax.set_xlabel("hours from start")
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return wrong_total


def code_panel(ax):
    ax.axis("off")
    ax.set_title("B. Why S1 fires in every file (script, lines 49-51)", loc="left", fontsize=10.5,
                 fontweight="bold")
    ax.text(0.0, 0.93, "for ch in range(1, N_channels):   # 1, 2, 3: channel 0 skipped\n"
                       "    signalGenerator.setChannelData(ch, Zeros)",
            family="monospace", fontsize=8, va="top", transform=ax.transAxes,
            bbox=dict(fc="#f7f7f7", ec="#bbb"))
    # channel state per file
    x0, y0, w, h = 0.08, 0.08, 0.2, 0.11
    ax.text(x0 - 0.02, y0 + 4.6 * h, "channel loaded with the wavelet while each file is recorded:", fontsize=8.5,
            transform=ax.transAxes)
    for j, k in enumerate((1, 2, 3, 4)):
        ax.text(x0 + (j + 0.5) * w, y0 + 4 * h, f"file {k}", ha="center", fontsize=8.5, transform=ax.transAxes)
    for c in range(4):
        ax.text(x0 - 0.01, y0 + (3 - c + 0.5) * h, f"ch {c} (S{c + 1})", ha="right", va="center", fontsize=8.5,
                transform=ax.transAxes)
        for j, k in enumerate((1, 2, 3, 4)):
            active = (c == k - 1) or (c == 0)
            stuck = c == 0 and k > 1
            ax.add_patch(Rectangle((x0 + j * w + 0.01, y0 + (3 - c) * h + 0.01), w - 0.02, h - 0.02,
                                   transform=ax.transAxes, fc=(WRONG if stuck else COL[c + 1]) if active else "#f2f2f2",
                                   ec="k", lw=0.4))
            if stuck:
                ax.text(x0 + (j + 0.5) * w, y0 + (3 - c + 0.5) * h, "never\ncleared", ha="center", va="center",
                        fontsize=7, color="w", transform=ax.transAxes)
    ax.text(x0, y0 - 0.06, "Result: file k = S1 + Sk. Kept on purpose for the defect scan\n(same content as the pristine). Fix for future scans:\nrange(0, N_channels).", fontsize=8, transform=ax.transAxes, va="top")


def map_panel(fig, gs):
    d = np.load(REPO / "ldv_work" / "prepared" / "pristine.npz")
    R = d["R"]
    O = 0.5 * (R.min(0) + R.max(0))
    U, V = R[:, 0] - O[0], R[:, 1] - O[1]
    for j, k in enumerate((2, 3, 4)):
        ax = fig.add_subplot(gs[j])
        bad = d[f"wrong{k}"]
        ax.scatter(U[~bad], V[~bad], s=7, marker="s", c="#d9d9d9")
        ax.scatter(U[bad], V[bad], s=7, marker="s", c=WRONG)
        ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(f"file {k}: {bad.sum()} points hold S{k - 1}'s shot", fontsize=8.5)
        if j == 0:
            ax.text(0, 1.22, "C. Where the wrong-source points are (pristine scan grid, 300 x 200 mm)", fontsize=10.5,
                    fontweight="bold", transform=ax.transAxes)
            ax.set_ylabel("detected (file k = file k-1 within noise)", fontsize=7)


def main():
    fig = plt.figure(figsize=(16, 13))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.9], width_ratios=[1.6, 1.0], hspace=0.62, wspace=0.12,
                          left=0.13, right=0.98, top=0.93, bottom=0.04)
    ax = fig.add_subplot(gs[0, :])
    w = timeline(ax, src_on=7443 + 300, gap=32, scan=6630, wait=600,
                 title="A1. Pristine night: the two programs run on separate clocks and drift apart")
    ax.text(0.0, -0.24, "Script: each source on for scan_time + safety_margin = 7443 + 300 = 7743 s.   "
                        "LDV: real scan 6630 s, then Wait 600 s, then the next scan starts on its own.\n"
                        "The scan-time estimate was 813 s too long, so each LDV scan starts ~510 s more before the "
                        "source switch than the previous one: the overlap grows 510 -> 1030 -> 1540 s.",
            transform=ax.transAxes, fontsize=8.5, va="top", linespacing=1.4)
    ax = fig.add_subplot(gs[1, :])
    timeline(ax, src_on=25700 + 7200, gap=7200, scan=24700, wait=10800,
             title="A2. Defect scan (fixed): the LDV restarts inside the trigger-off gap and waits for the next source")
    ax.text(0.0, -0.24, "Script: source on 25700 + 7200 s, then trigger off 7200 s.   LDV: Wait 10800 s = margin + "
                        "gap/2 -> restarts mid-gap and waits for the trigger.\nNo drift accumulates: the scan time may be "
                        "off by +/- 1 h per scan without any wrong-source point (shown with the measured ~24700 s scans).",
            transform=ax.transAxes, fontsize=8.5, va="top", linespacing=1.4)
    sub = gs[2, 0].subgridspec(1, 3, wspace=0.05)
    map_panel(fig, sub)
    code_panel(fig.add_subplot(gs[2, 1]))
    handles = [Patch(fc=COL[k], ec="k", label=f"S{k} / file {k}") for k in (1, 2, 3, 4)] + \
              [Patch(fc=WRONG, ec="k", label="wrong source recorded"), Patch(fc="#eee", ec="k", hatch="//", label="trigger off")]
    fig.legend(handles=handles, loc="upper right", ncol=6, fontsize=8.5, bbox_to_anchor=(0.98, 0.985))
    fig.suptitle("Why the pristine LDV scan mixed up sources", x=0.13, ha="left", fontsize=14, fontweight="bold")
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"pristine_scan_bug_figure.{ext}", dpi=110)
    print(HERE / "pristine_scan_bug_figure.png")


if __name__ == "__main__":
    main()
