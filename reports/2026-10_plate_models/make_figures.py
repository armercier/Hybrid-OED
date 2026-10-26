"""Figures for the plate-model note: dispersion, mode shapes, phase-error growth, grid dispersion.

All curves for the 10 mm PMMA plate of the Salvus benchmark (vp 2730, vs 1340 m/s, rho 1180).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "elastic"))
from mindlin import exact_a0_phase_velocity, mindlin_phase_velocity  # noqa: E402
from plane_stress import s0_phase_velocity  # noqa: E402

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq

FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)
VP, VS, RHO, H = 2730.0, 1340.0, 1180.0, 0.010
INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#d9d9d9"
CS0, CSH, CA0 = "#2a6fdb", "#1a9850", "#c2571a"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})

c_plate = 2 * VS * np.sqrt(1 - VS**2 / VP**2)
kappa_std = np.pi**2 / 12
kappa_fit = brentq(lambda k: mindlin_phase_velocity(25e3, H, VP, VS, RHO, k) - exact_a0_phase_velocity(25e3, H, VP, VS), 0.5, 1.2)
f = np.linspace(2e3, 80e3, 140)
s0 = np.array([s0_phase_velocity(x, H, VP, VS) for x in f])
a0 = np.array([exact_a0_phase_velocity(x, H, VP, VS) for x in f])
m_std = np.array([mindlin_phase_velocity(x, H, VP, VS, RHO, kappa_std) for x in f])
m_fit = np.array([mindlin_phase_velocity(x, H, VP, VS, RHO, kappa_fit) for x in f])
# S0-corrected plane stress: plate velocity matched to exact S0 at 25 kHz (constant in f)
c_ps_corr = s0_phase_velocity(25e3, H, VP, VS)


def group(fr, c):
    k = 2 * np.pi * fr / c
    return 2 * np.pi * np.gradient(fr) / np.gradient(k)


# ---------------------------------------------------------------- 1. dispersion
fig, axs = plt.subplots(2, 2, figsize=(10, 6.6), layout="constrained")
ax = axs[0, 0]
ax.plot(f / 1e3, s0, color=CS0, lw=2, label="S0 exact (Rayleigh-Lamb)")
ax.axhline(c_plate, color=CS0, ls="--", lw=1.2, label=f"plane stress S0 = {c_plate:.0f} m/s")
ax.axhline(c_ps_corr, color=CS0, ls=":", lw=1.2, label="plane stress S0, corrected at 25 kHz")
ax.axhline(VS, color=CSH, lw=2, label="SH0 exact = plane stress SH0 = vs")
ax.plot(f / 1e3, a0, color=CA0, lw=2, label="A0 exact")
ax.plot(f / 1e3, m_std, color=CA0, ls="--", lw=1.2, label=r"Mindlin A0, $\kappa=\pi^2/12$")
ax.plot(f / 1e3, m_fit, color=CA0, ls=":", lw=1.5, label=fr"Mindlin A0, $\kappa={kappa_fit:.3f}$")
ax.axvline(VS / (2 * H) / 1e3, color=MUTED, lw=0.8)
ax.text(VS / (2 * H) / 1e3 + 0.5, 1150, "A1 / SH1\ncut-on\n67 kHz", fontsize=7, color=MUTED)
ax.axvspan(10, 50, color="#f1f1f1", zorder=-1)
ax.text(11, 2480, "data band", fontsize=7, color=MUTED)
ax.set_ylim(300, 2600), ax.set_xlabel("frequency (kHz)"), ax.set_ylabel("phase velocity (m/s)")
ax.set_title("phase velocity, 10 mm PMMA", fontsize=9, loc="left")
ax.legend(fontsize=6.2, frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.62), ncol=2)
ax.grid(color=GRID, lw=0.5)

ax = axs[0, 1]
ax.plot(f / 1e3, group(f, s0), color=CS0, lw=2, label="S0 exact")
ax.axhline(c_plate, color=CS0, ls="--", lw=1.2, label="plane stress S0 (no dispersion)")
ax.axhline(VS, color=CSH, lw=2, label="SH0")
ax.plot(f / 1e3, group(f, a0), color=CA0, lw=2, label="A0 exact")
ax.plot(f / 1e3, group(f, m_std), color=CA0, ls="--", lw=1.2, label=r"Mindlin, $\kappa=\pi^2/12$")
ax.plot(f / 1e3, group(f, m_fit), color=CA0, ls=":", lw=1.5, label=r"Mindlin, $\kappa$ matched")
ax.axvspan(10, 50, color="#f1f1f1", zorder=-1)
ax.set_ylim(300, 2600), ax.set_xlabel("frequency (kHz)"), ax.set_ylabel("group velocity (m/s)")
ax.set_title("group velocity (energy / envelope speed)", fontsize=9, loc="left"), ax.legend(fontsize=6.5, frameon=False)
ax.grid(color=GRID, lw=0.5)

ax = axs[1, 0]
ax.plot(f / 1e3, 100 * (c_plate / s0 - 1), color=CS0, ls="--", label="plane stress vs exact S0")
ax.plot(f / 1e3, 100 * (c_ps_corr / s0 - 1), color=CS0, ls=":", label="plane stress, corrected at 25 kHz")
ax.plot(f / 1e3, 100 * (m_std / a0 - 1), color=CA0, ls="--", label=r"Mindlin $\kappa=\pi^2/12$ vs exact A0")
ax.plot(f / 1e3, 100 * (m_fit / a0 - 1), color=CA0, ls=":", label=r"Mindlin $\kappa$ matched at 25 kHz")
ax.axhline(0, color=INK, lw=0.6)
ax.axvspan(10, 50, color="#f1f1f1", zorder=-1)
ax.set_xlabel("frequency (kHz)"), ax.set_ylabel("phase-velocity error (%)"), ax.set_ylim(-4, 4)
ax.set_title("model error in phase velocity", fontsize=9, loc="left"), ax.legend(fontsize=6.5, frameon=False)
ax.grid(color=GRID, lw=0.5)

# phase error accumulated over distance, in cycles, at three frequencies
ax = axs[1, 1]
L = np.linspace(0, 1.0, 200)
for fr, ls in ((25e3, "-"), (39e3, "--"), (50e3, ":")):
    s0x, a0x = s0_phase_velocity(fr, H, VP, VS), exact_a0_phase_velocity(fr, H, VP, VS)
    mstd = mindlin_phase_velocity(fr, H, VP, VS, RHO, kappa_std)
    ax.plot(L, fr * L * np.abs(1 / c_plate - 1 / s0x), color=CS0, ls=ls, label=f"plane stress S0, {fr/1e3:.0f} kHz")
    ax.plot(L, fr * L * np.abs(1 / mstd - 1 / a0x), color=CA0, ls=ls, label=fr"Mindlin $\pi^2/12$, {fr/1e3:.0f} kHz")
ax.axhline(0.25, color=MUTED, lw=0.8)
ax.text(0.02, 0.27, "quarter cycle: L2 misfit ~ 1", fontsize=7, color=MUTED)
ax.set_xlabel("propagation distance (m)"), ax.set_ylabel("phase error (cycles)"), ax.set_ylim(0, 0.6)
ax.set_title("why small speed errors give large waveform misfits", fontsize=9, loc="left")
ax.legend(fontsize=6, frameon=False, ncol=2, loc="upper left"), ax.grid(color=GRID, lw=0.5)
fig.savefig(FIG / "dispersion.pdf")


# ---------------------------------------------------------------- 2. mode shapes
def lamb_mode_shape(fr, c, symmetric, z):
    """Through-thickness (u_x, u_z) of a Lamb mode (potentials, traction-free faces), real profiles."""
    w, d = 2 * np.pi * fr, H / 2
    k = w / c
    lam, mu = RHO * (VP**2 - 2 * VS**2), RHO * VS**2
    p = np.sqrt(complex((w / VP) ** 2 - k**2))
    q = np.sqrt(complex((w / VS) ** 2 - k**2))
    if symmetric:  # phi = A cos(pz), psi = B sin(qz)
        phi = lambda z, A: A * np.cos(p * z); dphi = lambda z, A: -A * p * np.sin(p * z)
        psi = lambda z, B: B * np.sin(q * z); dpsi = lambda z, B: B * q * np.cos(q * z)
    else:          # phi = A sin(pz), psi = B cos(qz)
        phi = lambda z, A: A * np.sin(p * z); dphi = lambda z, A: A * p * np.cos(p * z)
        psi = lambda z, B: B * np.cos(q * z); dpsi = lambda z, B: -B * q * np.sin(q * z)

    def tractions(A, B):
        szz = lam * (-k**2 - p**2) * phi(d, A) + 2 * mu * (-p**2 * phi(d, A) + 1j * k * dpsi(d, B))
        sxz = mu * (2j * k * dphi(d, A) + (q**2 - k**2) * psi(d, B))
        return np.array([szz, sxz])

    M = np.stack([tractions(1, 0), tractions(0, 1)], 1)
    _, _, vh = np.linalg.svd(M)
    A, B = np.conj(vh[-1])
    ux = 1j * k * phi(z, A) - dpsi(z, B)
    uz = dphi(z, A) + 1j * k * psi(z, B)
    real = lambda u: np.real(u * np.exp(-1j * np.angle(u[np.argmax(np.abs(u))])))
    ux, uz = real(ux), real(uz)
    s = max(np.abs(ux).max(), np.abs(uz).max())
    return ux / s, uz / s


z = np.linspace(-H / 2, H / 2, 101)
fig, axs = plt.subplots(1, 4, figsize=(10, 3.4), layout="constrained", sharey=True)
for ax, (fr, sym, title) in zip(axs, ((25e3, True, "S0, 25 kHz"), (50e3, True, "S0, 50 kHz"),
                                      (25e3, False, "A0, 25 kHz"), (50e3, False, "A0, 50 kHz"))):
    c = s0_phase_velocity(fr, H, VP, VS) if sym else exact_a0_phase_velocity(fr, H, VP, VS)
    ux, uz = lamb_mode_shape(fr, c, sym, z)
    ax.plot(ux, z * 1e3, color=CS0, lw=2, label="in-plane $u_x$ (exact)")
    ax.plot(uz, z * 1e3, color=CA0, lw=2, label="out-of-plane $u_z$ (exact)")
    if sym:
        ax.plot(np.full_like(z, ux[50]), z * 1e3, color=CS0, ls=":", lw=1.3, label="plane stress: $u_x$ uniform")
    else:
        slope = (ux[-1] - ux[0]) / (z[-1] - z[0])
        ax.plot(slope * z, z * 1e3, color=CS0, ls=":", lw=1.3, label=r"Mindlin: $u_x = z\,\varphi$ (linear)")
        ax.plot(np.full_like(z, uz[50]), z * 1e3, color=CA0, ls=":", lw=1.3, label="Mindlin: $w$ uniform")
    ax.axvline(0, color=MUTED, lw=0.5)
    ax.set_title(title, fontsize=9, loc="left"), ax.set_xlabel("normalised displacement")
    ax.grid(color=GRID, lw=0.5)
    ax.legend(fontsize=6, frameon=False, loc="lower left")
axs[0].set_ylabel("z (mm, 0 = mid-plane)")
fig.savefig(FIG / "mode_shapes.pdf")

# ---------------------------------------------------------------- 3. grid dispersion
G = np.linspace(4, 40, 200)  # points per wavelength
fig, ax = plt.subplots(figsize=(6, 3.0), layout="constrained")
for C, ls in ((0.42, "-"), (0.2, "--")):
    kdx = 2 * np.pi / G
    wdt = 2 * np.arcsin(np.clip(C * np.sin(kdx / 2), -1, 1))
    err = 100 * (wdt / (C * kdx) - 1)
    ax.plot(G, err, color=INK, ls=ls, label=f"Courant number {C}")
for lab, lam, col in (("A0, 50 kHz", exact_a0_phase_velocity(50e3, H, VP, VS) / 50e3, CA0),
                      ("SH0, 50 kHz", VS / 50e3, CSH), ("S0, 50 kHz", s0_phase_velocity(50e3, H, VP, VS) / 50e3, CS0)):
    for dx, mk in ((1e-3, "o"), (0.5e-3, "s"), (2e-3, "^")):
        g = lam / dx
        if 4 <= g <= 40:
            kd = 2 * np.pi / g
            e = 100 * (2 * np.arcsin(0.42 * np.sin(kd / 2)) / (0.42 * kd) - 1)
            ax.plot(g, e, mk, color=col, ms=5)
            ax.annotate(f"{lab}, dx={dx*1e3:g} mm", (g, e), fontsize=6, xytext=(3, -8), textcoords="offset points", color=col)
ax.set_xlabel("grid points per wavelength"), ax.set_ylabel("phase-velocity error (%)")
ax.set_title("2nd-order staggered FD: numerical dispersion (1-D estimate)", fontsize=9, loc="left")
ax.grid(color=GRID, lw=0.5), ax.legend(fontsize=7, frameon=False, loc="lower right")
fig.savefig(FIG / "grid_dispersion.pdf")
print("kappa matched:", kappa_fit, "| plate velocity", c_plate, "| S0 at 25 kHz", c_ps_corr)
print("Mindlin thickness-shear cut-on: pi^2/12 ->", np.sqrt(12 * kappa_std) * VS / (2 * np.pi * H),
      " matched ->", np.sqrt(12 * kappa_fit) * VS / (2 * np.pi * H))
