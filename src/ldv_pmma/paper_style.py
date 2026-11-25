"""Shared style for the MSSP paper figures (Crameri scientific colour maps, cmcrameri).

  THICK_CMAP  vik (diverging) for thickness maps, centred on the nominal 10 mm
  SEQ_CMAP    oslo (sequential) for every other colour map
  colours     categorical colours from osloS (Crameri's categorical oslo); categories also differ by marker or line style
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from cmcrameri import cm

THICK_CMAP = cm.vik
SEQ_CMAP = cm.oslo
OSLO = lambda f: SEQ_CMAP(f)                 # 0 = black, 1 = white

CAT = [cm.osloS(i) for i in range(8)]      # osloS order: near-black, white, blue, light, navy, mid, pale, deep blue
ADAPTIVE = CAT[4]           # navy  #163a5d
RANDOM = CAT[2]             # blue  #4e7abb
STATIC = CAT[3]             # light #9fafc9 (open markers)
DATA = CAT[0]               # measured data, near-black
MODEL_A = CAT[7]            # first model curve (Mindlin) #26578c
MODEL_B = CAT[5]            # second model curve (Salvus) #7a98ca
INK, MUTED, GRID = "black", "#4d4d4d", OSLO(0.92)     # axes and text black; notes dark grey


def apply():
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK,
                         "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                         "text.color": INK, "axes.titlecolor": INK})
