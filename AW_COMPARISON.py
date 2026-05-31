"""
plot_iter_vs_fixed_combined.py  —  AE4135 Rotor/Wake Aerodynamics, Assignment 2
Produces two PDF figures comparing iterated vs fixed a_W results:
  1. CT_CP_iter_vs_fixed.pdf        — CT and CP vs TSR (twin y-axes)
  2. CT_CP_pct_diff_iter_vs_fixed.pdf — % difference vs TSR

Authors: Douwe de Jong (5313899), Martijn van Leeuwen (5614422)
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# 0.  PATHS
# =============================================================================

_HERE        = os.path.dirname(os.path.abspath(__file__))
LL_NPZ_ITER  = os.path.join(_HERE, "LLM_results_iter.npz")
LL_NPZ_FIXED = os.path.join(_HERE, "LLM_results.npz")
OUT_DIR      = os.path.join(_HERE, "LLM_plots_standalone")
os.makedirs(OUT_DIR, exist_ok=True)

SHOW_PLOTS = False

# =============================================================================
# 1.  GLOBAL STYLE
# =============================================================================

mpl.rcParams.update({
    "text.usetex"        : False,
    "font.family"        : "serif",
    "font.serif"         : ["CMU Serif", "Computer Modern Roman",
                            "Latin Modern Roman", "DejaVu Serif"],
    "mathtext.fontset"   : "cm",
    "axes.labelsize"     : 12,
    "legend.fontsize"    : 10,
    "xtick.labelsize"    : 10,
    "ytick.labelsize"    : 10,
    "axes.titlesize"     : 12,
    "savefig.bbox"       : "tight",
    "savefig.pad_inches" : 0.02,
    "legend.frameon"     : True,
})

_CB = sns.color_palette("colorblind").as_hex()

# =============================================================================
# 2.  SAVE HELPER
# =============================================================================

def save_fig(name):
    stem  = name.rsplit(".", 1)[0] if "." in name else name
    fpath = os.path.join(OUT_DIR, stem + ".pdf")
    plt.savefig(fpath)
    print(f"  Saved: {stem}.pdf")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

# =============================================================================
# 3.  LOAD DATA
# =============================================================================

it  = np.load(LL_NPZ_ITER,  allow_pickle=True)
fix = np.load(LL_NPZ_FIXED, allow_pickle=True)

tsr    = it["perf_tsrs"]
CT_it  = it["perf_CT"];  CP_it  = it["perf_CP"]
CT_fix = fix["perf_CT"]; CP_fix = fix["perf_CP"]

A_WAKE_fix = float(fix["cfg_A_WAKE"])

# colour: quantity-based, not method-based
c_CT = _CB[0]   # blue  → CT
c_CP = _CB[2]   # green → CP

# =============================================================================
# 4.  FIGURE 1 — CT and CP vs TSR (twin y-axes)
# =============================================================================

fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()

# CT (left axis)
l1, = ax1.plot(tsr, CT_it,  color=c_CT, ls="-",  lw=2.0, marker="o", ms=4,
               label=r"$C_T$ iterated $a_w$")
l2, = ax1.plot(tsr, CT_fix, color=c_CT, ls="--", lw=1.5, marker="o", ms=4,
               alpha=0.5, label=rf"$C_T$ fixed $a_w={A_WAKE_fix}$")

# CP (right axis)
l3, = ax2.plot(tsr, CP_it,  color=c_CP, ls="-",  lw=2.0, marker="s", ms=4,
               label=r"$C_P$ iterated $a_w$")
l4, = ax2.plot(tsr, CP_fix, color=c_CP, ls="--", lw=1.5, marker="s", ms=4,
               alpha=0.5, label=rf"$C_P$ fixed $a_w={A_WAKE_fix}$")

# fill between to show the gap
ax1.fill_between(tsr, CT_it, CT_fix, color=c_CT, alpha=0.10)
ax2.fill_between(tsr, CP_it, CP_fix, color=c_CP, alpha=0.10)

ax1.set_xlabel(r"Tip-Speed Ratio $\lambda$ [-]")
ax1.set_ylabel(r"$C_T$ [-]", color=c_CT)
ax2.set_ylabel(r"$C_P$ [-]", color=c_CP)
ax1.tick_params(axis="y", colors=c_CT)
ax2.tick_params(axis="y", colors=c_CP)
ax1.set_xlim(tsr[0] - 0.2, tsr[-1] + 0.2)
ax1.legend(handles=[l1, l2, l3, l4], loc="upper left")
ax1.grid(True, lw=0.5)
fig.tight_layout()

save_fig("CT_CP_iter_vs_fixed")

# =============================================================================
# 5.  FIGURE 2 — percentage difference (iterated − fixed) / fixed
# =============================================================================

CT_pct = (CT_it - CT_fix) / CT_fix * 100
CP_pct = (CP_it - CP_fix) / CP_fix * 100

fig, ax = plt.subplots(figsize=(8, 5))

ax.axhline(0, color="grey", lw=0.8, ls=":", zorder=1)
ax.axhspan( 0,  5, color="grey", alpha=0.04)
ax.axhspan(-6,  0, color="grey", alpha=0.04)

l1, = ax.plot(tsr, CT_pct, color=c_CT, ls="-", lw=2.0, marker="o", ms=4,
              label=r"$\Delta C_T$")
l2, = ax.plot(tsr, CP_pct, color=c_CP, ls="-", lw=2.0, marker="s", ms=4,
              label=r"$\Delta C_P$")

ax.fill_between(tsr, CT_pct, 0, where=(CT_pct >= 0),
                color=c_CT, alpha=0.15, interpolate=True)
ax.fill_between(tsr, CT_pct, 0, where=(CT_pct <  0),
                color=c_CT, alpha=0.15, interpolate=True)
ax.fill_between(tsr, CP_pct, 0, where=(CP_pct >= 0),
                color=c_CP, alpha=0.15, interpolate=True)
ax.fill_between(tsr, CP_pct, 0, where=(CP_pct <  0),
                color=c_CP, alpha=0.15, interpolate=True)

ax.axvline(8, color="grey", lw=0.8, ls="--", zorder=1)
ax.text(8.1, -4.5, r"$\lambda=8$  ($a_w$ match)",
        fontsize=9, color="grey", va="bottom")

ax.set_xlabel(r"Tip-Speed Ratio $\lambda$ [-]")
ax.set_ylabel(
    r"$(C_{\mathrm{iter}} - C_{\mathrm{fixed}})\,/\,C_{\mathrm{fixed}}\ [\%]$")
ax.set_xlim(tsr[0] - 0.2, tsr[-1] + 0.2)
ax.legend(loc="lower left")
ax.grid(True, lw=0.5)
fig.tight_layout()

save_fig("CT_CP_pct_diff_iter_vs_fixed")