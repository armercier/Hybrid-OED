"""Paper figure F10 (lab result at small budgets, exploratory): three cases from relaxed to strict.

F10a, forest plot: gain in features found (of 5) of a designed variant over its baseline, per comparison:
  (i)   equal inversion cost   moving array, gradient moves (final 30 points, ~145 measured) vs static 30-point layouts;
                               moves planned beforehand on the background model (blue); random moves (grey control)
  (ii)  equal scan cost        adaptive points (3 batches of 10) re-inverted in one pass vs static 30; 30-point design
                               made beforehand (blue)
  (iii) equal scan + inversion adaptive as run vs the same protocol without design (progressive static, 33/33/34)
  (iv)  ~145 points            all positions visited by gradient (random: grey) moves, one pass, vs a jittered static
                               layout with the same number of points (4 and 2 shots only)
  The second criterion (log) is in the appendix.
Gain = mean(designed) - mean(baseline) per shot setting (small markers 4 / 2 / 1 shots); pooled = average of the per-shot
gains weighted by the number of designed runs (diamond) with a 95 % bootstrap interval (runs resampled within each
group and shot setting, 5000 draws). Left panel: fixed false-indication rate 1 % (threshold = 99th percentile of the
background anomaly, exploratory); right panel: v3 (|h - 10| > 3 mm, pre-registered). Static 30 = STATIC30 of run_overview.

F10b, paired slope plots: each line one start layout, from its baseline to its designed variant (1 % rule); colour and
marker = shot setting; small vertical offsets separate equal values. (i) shows random moves to the left of the static start
and gradient moves to the right.

Usage:  python src/ldv_pmma/plot_funnel.py
Output: reports/2026-11_mssp_paper/figures/F10a_funnel.{pdf,png}, F10b_slopes.{pdf,png}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paper_style import CAT, GRID, INK, MUTED, apply
from plot_fixed_rate import found
from run_overview import STATIC30
from score_inversions import INV, PREP

OUT = Path(__file__).resolve().parents[2] / "reports" / "2026-11_mssp_paper" / "figures"
S = json.loads((INV / "scores_v3.json").read_text())
SHOTS = (4, 2, 1)
SH_COL = {4: CAT[4], 2: CAT[2], 1: "#8c8c8c"}
SH_MK = {4: "o", 2: "s", 1: "^"}
J = ("j30a", "j30b", "j30c", "j30d")
Q1 = ("rc30", "j30a", "j30b", "j30c")
MV = {4: ("rc30", "j30a", "x48c"), 2: ("rc30", "j30a", "j30b", "x48a", "x48c"), 1: ("rc30", "j30a", "j30b", "x48a", "x48c")}
MPF = {4: ("ad", "rn", ""), 2: ("a2", "r2", "s2"), 1: ("a1", "r1", "k1")}
AD = {4: ["a30"] + [f"q4a_{s}" for s in J], 2: ["s2a30"] + [f"q2a_{s}" for s in J], 1: [f"q1a_{s}" for s in Q1]}
LG = {4: [f"q4l_{s}" for s in J], 2: [f"q2l_{s}" for s in J], 1: [f"q1l_{s}" for s in Q1]}
PR = {4: ["n30"] + [f"q4n_{s}" for s in J], 2: ["s2n30"] + [f"q2n_{s}" for s in J], 1: [f"q1n_{s}" for s in Q1]}
PRL = {4: [f"q4n_{s}" for s in J], 2: [f"q2n_{s}" for s in J], 1: [f"q1n_{s}" for s in Q1]}
# same-start static runs (for the slope plots)
SAME_RI = {4: {"a30": "rc30", "q4a_j30a": "j30a", "q4a_j30b": "j30b"},
           2: {"s2a30": "s2rc30", "q2a_j30a": "s2j30a", "q2a_j30b": "s2j30b"},
           1: {f"q1a_{s}": f"q1s_{s}" for s in Q1}}

OD = {4: "od", 2: "o2", 1: "o1"}
SIM = {4: ["d30a", "d30b", "d30c"], 2: ["s2d30", "s2d30b", "s2d30c"], 1: ["q1d_rc30", "q1d_j30a", "q1d_j30b"]}
EV = {4: ("ad", "rn"), 2: ("a2", "r2")}                       # all visited positions exist for 4 and 2 shots
ST30 = {k: STATIC30[k] for k in SHOTS}
PLAN = CAT[2]                                                  # designed beforehand (best-case model)
# (label, case annotation, designed, baseline, colour); log-criterion rows moved to the appendix (24 Nov)
COMP = [("(i) gradient moves\nvs static 30", "equal inversion cost",
         {k: [f"{MPF[k][0]}_{s}" for s in MV[k]] for k in SHOTS}, ST30, INK),
        ("(i) moves planned beforehand\nvs static 30", "",
         {k: [f"{OD[k]}_{s}" for s in MV[k]] for k in SHOTS}, ST30, PLAN),
        ("(i) control: random moves\nvs static 30", "",
         {k: [f"{MPF[k][1]}_{s}" for s in MV[k]] for k in SHOTS}, ST30, MUTED),
        ("(ii) adaptive, re-inverted\nvs static 30", "equal scan cost",
         {k: [f"ri_{t}" for t in AD[k]] for k in SHOTS}, ST30, INK),
        ("(ii) designed beforehand\nvs static 30", "", SIM, ST30, PLAN),
        ("(iii) adaptive as run\nvs progressive static", "equal scan +\ninversion cost", AD, PR, INK),
        ("(iv) all positions visited by\ngradient moves vs static, same n", "~145 points,\nequal scan +\ninversion cost",
         {k: [f"ev_{EV[k][0]}_{s}" for s in MV[k]] for k in EV}, {k: [f"mj{k}_{s}" for s in MV[k]] for k in EV}, INK),
        ("(iv) control: random moves,\nall visited vs static, same n", "",
         {k: [f"ev_{EV[k][1]}_{s}" for s in MV[k]] for k in EV}, {k: [f"mj{k}_{s}" for s in MV[k]] for k in EV}, MUTED)]


def scores(tags, cache, geo, O):
    out = []
    for t in tags:
        if t not in cache:
            cache[t] = (found(t, geo, O)[0], S[t]["found"])
        out.append(cache[t])
    return np.array(out, float)


def gain(des, base, r, rng=None):
    """Per-shot mean differences and the pooled (designed-n weighted) gain; rng: bootstrap resample within shot."""
    g, w = [], []
    for k in SHOTS:
        if not len(des.get(k, [])):
            continue
        d, b = des[k][:, r], base[k][:, r]
        if rng is not None:
            d, b = rng.choice(d, len(d)), rng.choice(b, len(b))
        g.append(d.mean() - b.mean()); w.append(len(d))
    return np.array(g), float(np.average(g, weights=w))


def shots_of(des):
    return [k for k in SHOTS if len(des.get(k, []))]


def forest(cache, geo, O):
    fig, axs = plt.subplots(1, 2, figsize=(8.0, 5.6), sharey=True, sharex=True, layout="constrained")
    rng = np.random.default_rng(0)
    rows = []
    for lab, case, D, B, col in COMP:
        rows.append((lab, case, col, {k: scores(D[k], cache, geo, O) for k in D if D[k]},
                     {k: scores(B[k], cache, geo, O) for k in B if B[k]}))
    ys = -np.arange(len(rows), dtype=float)
    off = {4: 0.22, 2: 0.0, 1: -0.22}                                    # per-shot markers on their own sub-rows
    for r, (ax, title) in enumerate(zip(axs, ("false-indication rate 1 % (exploratory)", "v3: |h - 10| > 3 mm (pre-registered)"))):
        for y, (lab, case, col, des, base) in zip(ys, rows):
            per, pooled = gain(des, base, r)
            boot = np.array([gain(des, base, r, rng)[1] for _ in range(5000)])
            lo, hi = np.percentile(boot, [2.5, 97.5])
            ax.plot([lo, hi], [y] * 2, color=col, lw=1.6, zorder=2, solid_capstyle="butt")
            ax.plot(pooled, y, "D", ms=7.5, color=col, mec="white", mew=0.8, zorder=4)
            for k, v in zip(shots_of(des), per):
                ax.plot(v, y + off[k], SH_MK[k], ms=4.5, mfc="white", mec=SH_COL[k], mew=1.1, zorder=3)
        ax.axvline(0, color=INK, lw=0.8)
        for y, v in zip(ys[1:], rows[1:]):                           # separator before each new case
            if v[1]:
                ax.axhline(y + 0.5, color=GRID, lw=0.8)
        ax.set_title(title, fontsize=8.5, loc="left")
        ax.set_xlabel("gain in features found (of 5)")
        ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    axs[0].set_yticks(ys, [v[0] for v in rows], fontsize=7.5)
    for t, v in zip(axs[0].get_yticklabels(), rows):
        t.set_color(v[2])
    for y, v in zip(ys, rows):
        if v[1]:
            axs[1].text(1.02, y, v[1], transform=axs[1].get_yaxis_transform(), fontsize=7, color=MUTED, va="center")
    axs[0].set_ylim(ys[-1] - 0.5, 0.5)
    h = [plt.Line2D([], [], color=INK, lw=1.6, marker="D", ms=7, mec="white", label="pooled gain, 95 % bootstrap interval")]
    h += [plt.Line2D([], [], ls="", marker=SH_MK[k], ms=5, mfc="white", mec=SH_COL[k], mew=1.1,
                     label=f"gain at {k} shot{'s' if k > 1 else ''}") for k in SHOTS]
    fig.legend(handles=h, loc="outside lower center", ncol=4, frameon=False, fontsize=7.5)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"F10a_funnel.{ext}", dpi=200)
    print(OUT / "F10a_funnel.png")


def slopes(cache, geo, O):
    fig, axs = plt.subplots(1, 3, figsize=(8.6, 3.4), sharey=True, layout="constrained")
    panels = [("(i) equal inversion cost", ["random\nmoves", "static\n(same start)", "gradient\nmoves"],
               [(k, (f"{MPF[k][1]}_{s}", f"{MPF[k][2]}{s}", f"{MPF[k][0]}_{s}")) for k in SHOTS for s in MV[k]]),
              ("(ii) equal scan cost", ["static\n(same start)", "adaptive,\nre-inverted"],
               [(k, (st, f"ri_{a}")) for k in SHOTS for a, st in SAME_RI[k].items()]),
              ("(iii) equal scan + inversion cost", ["progressive\nstatic", "adaptive\nas run"],
               [(k, (p, a)) for k in SHOTS for a, p in zip(AD[k], PR[k])])]
    for ax, (title, xl, rows) in zip(axs, panels):
        vals = [(k, [scores([t], cache, geo, O)[0, 0] for t in tags]) for k, tags in rows]
        # small offsets so equal paths do not hide each other
        seen = {}
        for k, v in vals:
            key = tuple(v); n = seen.get(key, 0); seen[key] = n + 1
            off = 0.07 * ((n + 1) // 2) * (1 if n % 2 else -1)
            x = np.arange(len(v))
            ax.plot(x, np.array(v) + off, "-", color=SH_COL[k], lw=1.0, alpha=0.8, zorder=2)
            ax.plot(x, np.array(v) + off, SH_MK[k], color=SH_COL[k], ms=4, mec="white", mew=0.4, zorder=3)
        up = sum(v[-1] > v[-2] for _, v in vals); eq = sum(v[-1] == v[-2] for _, v in vals); dn = sum(v[-1] < v[-2] for _, v in vals)
        ax.text(0.98, 0.03, f"higher {up}, same {eq}, lower {dn}", transform=ax.transAxes, ha="right", fontsize=7, color=INK)
        if len(xl) == 3:
            upr = sum(v[0] > v[1] for _, v in vals); eqr = sum(v[0] == v[1] for _, v in vals); dnr = sum(v[0] < v[1] for _, v in vals)
            ax.text(0.02, 0.97, f"random vs static:\nhigher {upr}, same {eqr}, lower {dnr}", transform=ax.transAxes,
                    fontsize=7, color=MUTED, va="top")
        ax.set_xticks(range(len(xl)), xl, fontsize=7.5)
        ax.set_xlim(-0.3, len(xl) - 0.7)
        ax.set_title(title, fontsize=9, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    axs[0].set_ylabel("features found (of 5)\nfalse-indication rate 1 %")
    axs[0].set_ylim(0.4, 6.0); axs[0].set_yticks(range(1, 6))
    h = [plt.Line2D([], [], color=SH_COL[k], marker=SH_MK[k], ms=4, label=f"{k} shot{'s' if k > 1 else ''}") for k in SHOTS]
    fig.legend(handles=h, loc="outside lower center", ncol=3, frameon=False, fontsize=8)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"F10b_slopes.{ext}", dpi=200)
    print(OUT / "F10b_slopes.png")


def main():
    apply()
    geo = json.loads((PREP / "geometry.json").read_text())["scatter"]["defects"]
    R = np.load(PREP / "scatter.npz")["R"]
    O = 0.5 * (R.min(0) + R.max(0))
    cache = {}
    forest(cache, geo, O)
    slopes(cache, geo, O)


if __name__ == "__main__":
    main()
