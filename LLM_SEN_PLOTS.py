"""
PLOTTING_LL_FINAL.py  —  AE4135 Rotor/Wake Aerodynamics, Assignment 2
Standalone plot generator: loads LLM_results.npz and reproduces all
Lifting Line figures without re-running the solver.

Authors: Douwe de Jong (5313899), Martijn van Leeuwen (5614422)
================================================================
Optionally also loads bem_results.npz for BEM/LL comparison plots.

Run:  python PLOTTING_LL_FINAL.py
"""

import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# 0.  PATHS  —  edit if your .npz files live elsewhere
# =============================================================================

_HERE       = os.path.dirname(os.path.abspath(__file__))
LL_NPZ      = os.path.join(_HERE, "LLM_results.npz")
BEM_NPZ     = os.path.join(_HERE, "bem_results.npz")   # optional
OUT_DIR     = os.path.join(_HERE, "LLM_plots_standalone")
os.makedirs(OUT_DIR, exist_ok=True)

SHOW_PLOTS = False   # True → plt.show() after each save

# =============================================================================
# 1.  PLOT TOGGLES  (set False to skip a section)
# =============================================================================

PLOT_LL_1          = True   # inflow angle + AoA spanwise
PLOT_LL_2          = True   # induction factors spanwise
PLOT_LL_3          = True   # loading distributions spanwise
PLOT_LL_4          = True   # circulation spanwise
PLOT_LL_5          = True   # CT / CP vs TSR  (wide performance sweep)
PLOT_BEM_COMPARE   = True   # BEM vs LL comparison (requires bem_results.npz)
PLOT_SENS_AW       = True   # sensitivity: wake convection speed
PLOT_SENS_DISC     = True   # sensitivity: blade discretisation
PLOT_SENS_DPSI     = True   # sensitivity: azimuthal step
PLOT_SENS_WAKE     = True   # sensitivity: wake length

# =============================================================================
# 2.  GLOBAL STYLE  (matches LiftingLine_FINAL.py)
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

_CB_PALETTE = sns.color_palette("colorblind").as_hex() + [
    '#8b0000', '#556b2f', '#4b0082', '#ff6347', '#20b2aa',
    '#8b4513', '#2f4f4f', '#9400d3', '#6a5acd', '#b8860b',
]

def _tsr_color(idx, n=None):
    return _CB_PALETTE[idx % len(_CB_PALETTE)]

def _sens_color(idx, n=None):
    return _CB_PALETTE[idx % len(_CB_PALETTE)]

# =============================================================================
# 3.  SAVE HELPER
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

def _skip(name, reason):
    print(f"  [SKIP] {name} — {reason}")
    plt.close("all")

# =============================================================================
# 4.  LOAD DATA
# =============================================================================

print(f"Loading LL results  →  {LL_NPZ}")
if not os.path.exists(LL_NPZ):
    sys.exit(f"ERROR: {LL_NPZ} not found. Run LiftingLine_FINAL.py first.")

L = np.load(LL_NPZ, allow_pickle=False)

# ── Config scalars ─────────────────────────────────────────────────────────────
def _lcfg(key, default):
    return float(L[key]) if key in L.files else default

Radius          = _lcfg("cfg_Radius",          50.0)
NBlades         = int(_lcfg("cfg_NBlades",       3))
U0              = _lcfg("cfg_U0",               10.0)
rho             = _lcfg("cfg_rho",               1.0)
N_PANELS        = int(_lcfg("cfg_N_PANELS",      20))
N_WAKE          = int(_lcfg("cfg_N_WAKE",         5))
DPSI_DEG        = _lcfg("cfg_DPSI_DEG",         10.0)
A_WAKE          = _lcfg("cfg_A_WAKE",            0.25)

norm_val = 0.5 * rho * U0**2 * Radius

# ── TSR sweep arrays ───────────────────────────────────────────────────────────
TSR_SWEEP_SPAN = list(L["sweep_tsrs"].astype(int)) if "sweep_tsrs" in L.files else [6, 8, 10]
TSR_SWEEP_PERF = list(L["perf_tsrs"])              if "perf_tsrs"  in L.files else TSR_SWEEP_SPAN

tsr_CT_span    = L["tsr_CT"]   if "tsr_CT"   in L.files else np.array([])
tsr_CP_span    = L["tsr_CP"]   if "tsr_CP"   in L.files else np.array([])
tsr_CT_perf    = L["perf_CT"]  if "perf_CT"  in L.files else np.array([])
tsr_CP_perf    = L["perf_CP"]  if "perf_CP"  in L.files else np.array([])

sweep_data_span = {}
for tsr in TSR_SWEEP_SPAN:
    key = f"sweep_res_{int(tsr)}"
    if key in L.files:
        sweep_data_span[tsr] = L[key]

# ── Sensitivity arrays ─────────────────────────────────────────────────────────
def _lget(key, default=None):
    return L[key] if key in L.files else (default if default is not None else np.array([]))

SENS_AW_LIST    = list(_lget("sens_aw_vals"))
SENS_DPSI_LIST  = list(_lget("sens_dpsi_vals"))
SENS_NWAKE_LIST = [int(x) for x in _lget("sens_nwake_vals")]
SENS_N_LIST     = [int(x) for x in _lget("sens_N_vals")]

sens_aw_CT      = _lget("sens_aw_CT")
sens_aw_CP      = _lget("sens_aw_CP")
sens_dpsi_CT    = _lget("sens_dpsi_CT")
sens_dpsi_CP    = _lget("sens_dpsi_CP")
sens_nwake_CT   = _lget("sens_nwake_CT")
sens_nwake_CP   = _lget("sens_nwake_CP")
sens_N_CT_cos   = _lget("sens_N_CT_cosine")
sens_N_CP_cos   = _lget("sens_N_CP_cosine")

def _sens_res(prefix, val):
    key = f"{prefix}{val}"
    return L[key] if key in L.files else None

sens_aw_data   = {}
for aw in SENS_AW_LIST:
    r = _sens_res("sens_aw_res_", aw)
    i = SENS_AW_LIST.index(aw)
    if r is not None:
        sens_aw_data[aw] = (r,
                            float(sens_aw_CT[i]) if i < len(sens_aw_CT) else np.nan,
                            float(sens_aw_CP[i]) if i < len(sens_aw_CP) else np.nan)

sens_dpsi_data = {}
for dp in SENS_DPSI_LIST:
    r = _sens_res("sens_dpsi_res_", float(dp))
    i = SENS_DPSI_LIST.index(dp)
    if r is not None:
        sens_dpsi_data[dp] = (r,
                              float(sens_dpsi_CT[i]) if i < len(sens_dpsi_CT) else np.nan,
                              float(sens_dpsi_CP[i]) if i < len(sens_dpsi_CP) else np.nan)

sens_wake_data = {}
for nw in SENS_NWAKE_LIST:
    r = _sens_res("sens_nwake_res_", int(nw))
    i = SENS_NWAKE_LIST.index(nw)
    if r is not None:
        sens_wake_data[nw] = (r,
                              float(sens_nwake_CT[i]) if i < len(sens_nwake_CT) else np.nan,
                              float(sens_nwake_CP[i]) if i < len(sens_nwake_CP) else np.nan)

sens_disc_data = {}
for N in SENS_N_LIST:
    for dist in ["cosine", "constant"]:
        r = _sens_res(f"sens_disc_res_{N}_", dist)
        if r is not None:
            sens_disc_data[(N, dist)] = (r, np.nan, np.nan)   # CT/CP filled below
# Fill CT/CP for cosine from saved arrays
for i, N in enumerate(SENS_N_LIST):
    if (N, "cosine") in sens_disc_data and i < len(sens_N_CT_cos):
        res, _, _ = sens_disc_data[(N, "cosine")]
        sens_disc_data[(N, "cosine")] = (res,
                                          float(sens_N_CT_cos[i]),
                                          float(sens_N_CP_cos[i]))

# ── Optional: BEM data ─────────────────────────────────────────────────────────
_have_bem = os.path.exists(BEM_NPZ)
if _have_bem:
    print(f"Loading BEM results  →  {BEM_NPZ}")
    B = np.load(BEM_NPZ, allow_pickle=False)
    bem = {}
    for tsr in TSR_SWEEP_SPAN:
        key = f"sweep_res_{int(tsr)}"
        if key in B.files:
            bem[tsr] = B[key]
    rho_bem  = float(B["cfg_rho"])  if "cfg_rho"  in B.files else 1.0
    norm_bem = 0.5 * rho_bem * U0**2 * Radius
    bem_CT_span = {}; bem_CP_span = {}
    if "tsr_CT" in B.files:
        for i, tsr in enumerate(TSR_SWEEP_SPAN):
            if i < len(B["tsr_CT"]):
                bem_CT_span[tsr] = float(B["tsr_CT"][i])
    if "tsr_CP" in B.files:
        for i, tsr in enumerate(TSR_SWEEP_SPAN):
            if i < len(B["tsr_CP"]):
                bem_CP_span[tsr] = float(B["tsr_CP"][i])
    bem_tsrs_perf = B["sweep_tsrs_perf"] if "sweep_tsrs_perf" in B.files else \
                    B["sweep_tsrs"]       if "sweep_tsrs"      in B.files else \
                    np.array(TSR_SWEEP_SPAN, dtype=float)
    bem_CT_perf   = B["tsr_CT_perf"] if "tsr_CT_perf" in B.files else \
                    B["tsr_CT"]       if "tsr_CT"       in B.files else np.array([])
    bem_CP_perf   = B["tsr_CP_perf"] if "tsr_CP_perf" in B.files else \
                    B["tsr_CP"]       if "tsr_CP"       in B.files else np.array([])
else:
    print(f"  [INFO] {BEM_NPZ} not found — BEM comparison plots will be skipped.")
    bem = {}; norm_bem = norm_val; bem_CT_span = {}; bem_CP_span = {}
    bem_tsrs_perf = np.array([]); bem_CT_perf = np.array([]); bem_CP_perf = np.array([])

print(f"\nLoaded {len(sweep_data_span)} span-TSR cases: {list(sweep_data_span.keys())}")
print(f"Sensitivity: aw={len(sens_aw_data)}, dpsi={len(sens_dpsi_data)}, "
      f"nwake={len(sens_wake_data)}, disc={len(sens_disc_data)}")

n_span = len(TSR_SWEEP_SPAN)

# Scalar CT/CP at span TSRs (used in comparison)
ll_CT_span = {tsr: float(tsr_CT_span[i])
              for i, tsr in enumerate(TSR_SWEEP_SPAN) if i < len(tsr_CT_span)}
ll_CP_span = {tsr: float(tsr_CP_span[i])
              for i, tsr in enumerate(TSR_SWEEP_SPAN) if i < len(tsr_CP_span)}

# =============================================================================
# 5.  LL.1  —  Inflow angle and angle of attack
# =============================================================================

print("\n" + "="*60)
print("Plotting LL.1 — inflow angle and AoA")

if PLOT_LL_1 and sweep_data_span:
    for qty_col, ylabel, fname in [
            (7, r"$\phi$ [deg]",   "LL_1a_inflow_angle_vs_rR"),
            (6, r"$\alpha$ [deg]", "LL_1b_angle_of_attack_vs_rR")]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for k, tsr in enumerate(TSR_SWEEP_SPAN):
            if tsr not in sweep_data_span: continue
            res = sweep_data_span[tsr]
            ax.plot(res[:, 2], res[:, qty_col],
                    color=_tsr_color(k), lw=2, label=rf"$\lambda={tsr}$")
        ax.set_xlabel("r/R"); ax.set_ylabel(ylabel)
        ax.legend(); ax.grid(True)
        fig.tight_layout(); save_fig(fname)
elif PLOT_LL_1:
    _skip("LL.1", "span sweep data missing")

# =============================================================================
# 6.  LL.2  —  Induction factors
# =============================================================================

print("Plotting LL.2 — induction factors")

if PLOT_LL_2 and sweep_data_span:
    for qty_col, ylabel, fname in [
            (0, r"$a$ [-]",  "LL_2a_axial_induction_vs_rR"),
            (1, r"$a'$ [-]", "LL_2b_tangential_induction_vs_rR")]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for k, tsr in enumerate(TSR_SWEEP_SPAN):
            if tsr not in sweep_data_span: continue
            res = sweep_data_span[tsr]
            ax.plot(res[:, 2], res[:, qty_col],
                    color=_tsr_color(k), lw=2, label=rf"$\lambda={tsr}$")
        ax.set_xlabel("r/R"); ax.set_ylabel(ylabel)
        ax.legend(); ax.grid(True)
        fig.tight_layout(); save_fig(fname)
elif PLOT_LL_2:
    _skip("LL.2", "span sweep data missing")

# =============================================================================
# 7.  LL.3  —  Loading distributions
# =============================================================================

print("Plotting LL.3 — loading distributions")

if PLOT_LL_3 and sweep_data_span:
    for qty_col, ylabel, fname in [
            (3, r"$C_n = F_n\,/\,(\frac{1}{2}\rho U_\infty^2 R)$",
             "LL_3a_normal_loading_Cn_vs_rR"),
            (4, r"$C_t = F_t\,/\,(\frac{1}{2}\rho U_\infty^2 R)$",
             "LL_3b_azimuthal_loading_Ct_vs_rR")]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for k, tsr in enumerate(TSR_SWEEP_SPAN):
            if tsr not in sweep_data_span: continue
            res = sweep_data_span[tsr]
            ax.plot(res[:, 2], res[:, qty_col] / norm_val,
                    color=_tsr_color(k), lw=2, label=rf"$\lambda={tsr}$")
        ax.set_xlabel("r/R"); ax.set_ylabel(ylabel)
        ax.legend(); ax.grid(True)
        fig.tight_layout(); save_fig(fname)
elif PLOT_LL_3:
    _skip("LL.3", "span sweep data missing")

# =============================================================================
# 8.  LL.4  —  Circulation
# =============================================================================

print("Plotting LL.4 — circulation")

if PLOT_LL_4 and sweep_data_span:
    fig, ax = plt.subplots(figsize=(8, 5))
    for k, tsr in enumerate(TSR_SWEEP_SPAN):
        if tsr not in sweep_data_span: continue
        res   = sweep_data_span[tsr]
        Omega = U0 * tsr / Radius
        norm_G = np.pi * U0**2 / (NBlades * Omega)
        ax.plot(res[:, 2], res[:, 5] / norm_G,
                color=_tsr_color(k), lw=2, label=rf"$\lambda={tsr}$")
    ax.set_xlabel("r/R")
    ax.set_ylabel(r"$\Gamma\,/\,(\pi U_\infty^2 / (B\,\Omega))$ [-]")
    ax.legend(); ax.grid(True)
    fig.tight_layout(); save_fig("LL_4_circulation_vs_rR")
elif PLOT_LL_4:
    _skip("LL.4", "span sweep data missing")

# =============================================================================
# 9.  LL.5  —  CT and CP vs TSR  (wide performance sweep)
# =============================================================================

print("Plotting LL.5 — CT/CP vs TSR")

if PLOT_LL_5 and len(tsr_CT_perf) > 0:
    for vals, ylabel, fname, col in [
            (tsr_CT_perf, r"$C_T$ [-]", "LL_5a_CT_vs_TSR", "#0173b2"),
            (tsr_CP_perf, r"$C_P$ [-]", "LL_5b_CP_vs_TSR", "#029e73")]:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(TSR_SWEEP_PERF, vals, "o-", color=col, lw=2, ms=5, zorder=2)
        if "CP" in ylabel:
            ax.axhline(16/27, color="grey", ls=":", lw=1.2, label="Betz limit")
            ax.legend()
        ax.set_xlabel(r"Tip-speed ratio $\lambda$ [-]")
        ax.set_ylabel(ylabel)
        ax.set_xlim(min(TSR_SWEEP_PERF) - 0.5, max(TSR_SWEEP_PERF) + 0.5)
        ax.set_xticks(np.arange(4, 13, 1))
        ax.grid(True)
        fig.tight_layout(); save_fig(fname)
elif PLOT_LL_5:
    _skip("LL.5", "performance sweep data missing")

# =============================================================================
# 10.  BEM vs LL comparison  (requires bem_results.npz)
# =============================================================================

print("Plotting BEM/LL comparison")

if PLOT_BEM_COMPARE and _have_bem and sweep_data_span and bem:
    _TSR_COL = {tsr: _tsr_color(k) for k, tsr in enumerate(TSR_SWEEP_SPAN)}

    # ── Inflow angle: side-by-side subplots ────────────────────────────────
    for qty_col, ylabel, fname in [
            (7, r"$\phi$ [deg]",   "BEMvsLL_inflow_angle_vs_rR"),
            (6, r"$\alpha$ [deg]", "BEMvsLL_angle_of_attack_vs_rR")]:
        fig, axes = plt.subplots(1, len(TSR_SWEEP_SPAN), figsize=(14, 5), sharey=True)
        for ax, tsr in zip(axes, TSR_SWEEP_SPAN):
            col = _TSR_COL[tsr]
            if tsr in sweep_data_span:
                ax.plot(sweep_data_span[tsr][:, 2], sweep_data_span[tsr][:, qty_col],
                        color=col, lw=2, ls="-",  label=rf"LL ($\lambda={tsr}$)")
            if tsr in bem:
                ax.plot(bem[tsr][:, 2], bem[tsr][:, qty_col],
                        color="k",  lw=2, ls="--", label=rf"BEM ($\lambda={tsr}$)")
            ax.set_xlabel("r/R"); ax.legend(fontsize=9); ax.grid(True)
        axes[0].set_ylabel(ylabel)
        fig.tight_layout(); save_fig(fname)

    # ── Induction factors ──────────────────────────────────────────────────
    for qty_col, ylabel, fname in [
            (0, r"$a$ [-]",  "BEMvsLL_axial_induction_vs_rR"),
            (1, r"$a'$ [-]", "BEMvsLL_tangential_induction_vs_rR")]:
        fig, axes = plt.subplots(1, len(TSR_SWEEP_SPAN), figsize=(14, 5), sharey=True)
        for ax, tsr in zip(axes, TSR_SWEEP_SPAN):
            col = _TSR_COL[tsr]
            if tsr in sweep_data_span:
                ax.plot(sweep_data_span[tsr][:, 2], sweep_data_span[tsr][:, qty_col],
                        color=col, lw=2, ls="-",  label=rf"LL ($\lambda={tsr}$)")
            if tsr in bem:
                ax.plot(bem[tsr][:, 2], bem[tsr][:, qty_col],
                        color="k",  lw=2, ls="--", label=rf"BEM ($\lambda={tsr}$)")
            ax.set_xlabel("r/R"); ax.legend(fontsize=9); ax.grid(True)
        axes[0].set_ylabel(ylabel)
        fig.tight_layout(); save_fig(fname)

    # ── Loading distributions ──────────────────────────────────────────────
    for qty_col, ylabel, fname in [
            (3, r"$C_n$ [-]", "BEMvsLL_normal_loading_Cn_vs_rR"),
            (4, r"$C_t$ [-]", "BEMvsLL_azimuthal_loading_Ct_vs_rR")]:
        fig, axes = plt.subplots(1, len(TSR_SWEEP_SPAN), figsize=(14, 5), sharey=True)
        for ax, tsr in zip(axes, TSR_SWEEP_SPAN):
            col = _TSR_COL[tsr]
            if tsr in sweep_data_span:
                ax.plot(sweep_data_span[tsr][:, 2],
                        sweep_data_span[tsr][:, qty_col] / norm_val,
                        color=col, lw=2, ls="-",  label=rf"LL ($\lambda={tsr}$)")
            if tsr in bem:
                ax.plot(bem[tsr][:, 2], bem[tsr][:, qty_col] / norm_bem,
                        color="k",  lw=2, ls="--", label=rf"BEM ($\lambda={tsr}$)")
            ax.set_xlabel("r/R"); ax.legend(fontsize=9); ax.grid(True)
        axes[0].set_ylabel(ylabel)
        fig.tight_layout(); save_fig(fname)

    # ── CT/CP vs TSR comparison ────────────────────────────────────────────
    for ll_v, bem_v, bem_tsr, ylabel, fname, col in [
            (tsr_CT_perf, bem_CT_perf, bem_tsrs_perf,
             r"$C_T$ [-]", "BEMvsLL_CT_vs_TSR", "#0173b2"),
            (tsr_CP_perf, bem_CP_perf, bem_tsrs_perf,
             r"$C_P$ [-]", "BEMvsLL_CP_vs_TSR", "#029e73")]:
        fig, ax = plt.subplots(figsize=(9, 5))
        if len(ll_v) > 0:
            ax.plot(TSR_SWEEP_PERF, ll_v, "o-", color=col,  lw=2, ms=5, label="LL")
        if len(bem_v) > 0:
            ax.plot(bem_tsr, bem_v, "s--", color="k", lw=2, ms=5, label="BEM")
        if "CP" in ylabel:
            ax.axhline(16/27, color="grey", ls=":", lw=1.2, label="Betz limit")
        ax.set_xlabel(r"Tip-speed ratio $\lambda$ [-]")
        ax.set_ylabel(ylabel)
        ax.set_xlim(3.5, 12.5); ax.set_xticks(np.arange(4, 13, 1))
        ax.legend(); ax.grid(True)
        fig.tight_layout(); save_fig(fname)

    # ── Scalar comparison table ────────────────────────────────────────────
    print("\n  CT/CP comparison (LL vs BEM):")
    print(f"  {'TSR':>4} {'CT_LL':>8} {'CT_BEM':>8} {'dCT%':>7} "
          f"{'CP_LL':>8} {'CP_BEM':>8} {'dCP%':>7}")
    print("  " + "-"*56)
    for tsr in TSR_SWEEP_SPAN:
        ct_ll  = ll_CT_span.get(tsr,  float("nan"))
        ct_bem = bem_CT_span.get(tsr, float("nan"))
        cp_ll  = ll_CP_span.get(tsr,  float("nan"))
        cp_bem = bem_CP_span.get(tsr, float("nan"))
        dct = (ct_ll - ct_bem) / ct_bem * 100 if ct_bem else float("nan")
        dcp = (cp_ll - cp_bem) / cp_bem * 100 if cp_bem else float("nan")
        print(f"  {tsr:>4} {ct_ll:>8.4f} {ct_bem:>8.4f} {dct:>+7.2f}% "
              f"{cp_ll:>8.4f} {cp_bem:>8.4f} {dcp:>+7.2f}%")

elif PLOT_BEM_COMPARE and not _have_bem:
    _skip("BEM/LL compare", "bem_results.npz not found")

# =============================================================================
# 11.  SENSITIVITY: wake convection speed (a_w)
# =============================================================================

print("\n" + "="*60)
print("Plotting sensitivity: convection speed a_w")

if PLOT_SENS_AW and sens_aw_data:
    n_aw      = len(SENS_AW_LIST)
    aw_vals   = [aw for aw in SENS_AW_LIST if aw in sens_aw_data]
    ct_vals   = [sens_aw_data[aw][1] for aw in aw_vals]
    cp_vals   = [sens_aw_data[aw][2] for aw in aw_vals]
    aw_arr    = np.array(aw_vals)
    ct_arr    = np.array(ct_vals)
    cp_arr    = np.array(cp_vals)

    # Mean axial induction from span-averaged column 0
    meana_vals = []
    for aw in aw_vals:
        res, _, _ = sens_aw_data[aw]
        dr = np.gradient(res[:, 2])   # proxy weight (uniform r/R spacing)
        meana_vals.append(float(np.average(res[:, 0], weights=np.abs(dr))))
    meana_arr = np.array(meana_vals)
    resid_arr = meana_arr - aw_arr

    # Self-consistent a_w (zero crossing of residual)
    try:
        sort_idx = np.argsort(resid_arr)
        sc_aw = float(np.interp(0.0, resid_arr[sort_idx], aw_arr[sort_idx]))
        sc_CT = float(np.interp(sc_aw, aw_arr, ct_arr))
        sc_CP = float(np.interp(sc_aw, aw_arr, cp_arr))
        has_sc = True
        print(f"  Self-consistent a_w = {sc_aw:.4f}  CT={sc_CT:.4f}  CP={sc_CP:.4f}")
    except Exception:
        has_sc = False; sc_aw = A_WAKE

    # CT vs a_w
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(aw_arr, ct_arr, "o-", color="#0173b2", lw=2, ms=5, zorder=3)
    ax.axvline(A_WAKE, color="#d55e00", ls="--", lw=1.5,
               label=rf"Fixed $a_w = {A_WAKE}$")
    if has_sc:
        ax.axvline(sc_aw, color="#029e73", ls=":", lw=2,
                   label=rf"Self-consistent $a_w = {sc_aw:.3f}$")
    ax.set_xlabel(r"Wake convection induction $a_w$ [-]")
    ax.set_ylabel(r"$C_T$ [-]")
    ax.legend(fontsize=9); ax.grid(True)
    fig.tight_layout(); save_fig("Sens_AW_1_CT_vs_aw")

    # CP vs a_w
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(aw_arr, cp_arr, "o-", color="#029e73", lw=2, ms=5, zorder=3)
    ax.axvline(A_WAKE, color="#d55e00", ls="--", lw=1.5,
               label=rf"Fixed $a_w = {A_WAKE}$")
    if has_sc:
        ax.axvline(sc_aw, color="#0173b2", ls=":", lw=2,
                   label=rf"Self-consistent $a_w = {sc_aw:.3f}$")
    ax.set_xlabel(r"Wake convection induction $a_w$ [-]")
    ax.set_ylabel(r"$C_P$ [-]")
    ax.legend(fontsize=9); ax.grid(True)
    fig.tight_layout(); save_fig("Sens_AW_2_CP_vs_aw")

    # mean_a vs a_w  (self-consistency plot)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(aw_arr, meana_arr, "o-", color="#0173b2", lw=2, ms=5,
            label=r"$\bar{a}$ from LL")
    ax.plot(aw_arr, aw_arr, "--", color="#949494", lw=1.5,
            label=r"$\bar{a} = a_w$ (self-consistent line)")
    if has_sc:
        ax.axvline(sc_aw, color="#029e73", ls=":", lw=2,
                   label=rf"Self-consistent $a_w = {sc_aw:.3f}$")
        ax.scatter([sc_aw], [sc_aw], color="#029e73", s=70, zorder=5)
    ax.set_xlabel(r"Input $a_w$ [-]"); ax.set_ylabel(r"$\bar{a}$ [-]")
    ax.legend(fontsize=9); ax.grid(True)
    fig.tight_layout(); save_fig("Sens_AW_3_mean_a_vs_aw")

    # Residual vs a_w
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(aw_arr, resid_arr, "o-", color="#d55e00", lw=2, ms=5)
    ax.axhline(0, color="#949494", ls="--", lw=1.5)
    if has_sc:
        ax.axvline(sc_aw, color="#029e73", ls=":", lw=2,
                   label=rf"Zero crossing $a_w = {sc_aw:.3f}$")
        ax.scatter([sc_aw], [0.0], color="#029e73", s=70, zorder=5)
        ax.legend(fontsize=9)
    ax.set_xlabel(r"Input $a_w$ [-]")
    ax.set_ylabel(r"Residual $\bar{a} - a_w$ [-]")
    ax.grid(True)
    fig.tight_layout(); save_fig("Sens_AW_4_residual_vs_aw")

    # Spanwise distributions
    for qty_col, ylabel, fname, use_meana_label in [
            (0, r"$a$ [-]",
             "Sens_AW_5_axial_induction_vs_rR", True),
            (5, r"$\Gamma$ [m$^2$/s]",
             "Sens_AW_6_circulation_vs_rR", False),
            (3, r"$C_n = F_n\,/\,(\frac{1}{2}\rho U_\infty^2 R)$ [-]",
             "Sens_AW_7_normal_loading_vs_rR", False),
            (7, r"$\phi$ [deg]",
             "Sens_AW_8_inflow_angle_vs_rR", False)]:
        fig, ax = plt.subplots(figsize=(9, 5))
        for idx, aw in enumerate(aw_vals):
            res, CT, CP = sens_aw_data[aw]
            y = res[:, qty_col] / norm_val if qty_col == 3 else res[:, qty_col]
            lbl = (rf"$a_w={aw:.2f}$  $\bar{{a}}={meana_vals[idx]:.3f}$"
                   if use_meana_label else
                   rf"$a_w={aw:.2f}$  $C_T={CT:.3f}$")
            ax.plot(res[:, 2], y, color=_sens_color(idx, n_aw), lw=1.8, label=lbl)
        ax.set_xlabel("r/R"); ax.set_ylabel(ylabel)
        ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
        ax.grid(True)
        fig.tight_layout(); save_fig(fname)

elif PLOT_SENS_AW:
    _skip("Sens_AW", "convection speed sensitivity data missing")

# =============================================================================
# 12.  SENSITIVITY: blade discretisation
# =============================================================================

print("Plotting sensitivity: discretisation")

if PLOT_SENS_DISC and sens_disc_data:
    n_N = len(SENS_N_LIST)

    _DIST_STYLE = {
        "cosine":   {"color": "#0173b2", "ls": "-",  "marker": "o",
                     "label_prefix": "Cosine"},
        "constant": {"color": "#d55e00", "ls": "--", "marker": "s",
                     "label_prefix": "Constant"},
    }

    # Spanwise Cn at each panel count  (cosine only)
    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, N in enumerate(SENS_N_LIST):
        if (N, "cosine") not in sens_disc_data: continue
        res, CT, CP = sens_disc_data[(N, "cosine")]
        ax.plot(res[:, 2], res[:, 3] / norm_val, color=_sens_color(idx, n_N), lw=2,
                label=rf"N={N}  $C_T={CT:.3f}$" if not np.isnan(CT) else f"N={N}")
    ax.set_xlabel("r/R"); ax.set_ylabel(r"$C_n$ [-]")
    ax.legend(fontsize=8); ax.grid(True)
    fig.tight_layout(); save_fig("Sens_Disc_a1_Cn_panel_count_cosine")

    # Spanwise a at each panel count  (cosine only)
    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, N in enumerate(SENS_N_LIST):
        if (N, "cosine") not in sens_disc_data: continue
        res, CT, CP = sens_disc_data[(N, "cosine")]
        ax.plot(res[:, 2], res[:, 0], color=_sens_color(idx, n_N), lw=2,
                label=rf"N={N}  $C_P={CP:.3f}$" if not np.isnan(CP) else f"N={N}")
    ax.set_xlabel("r/R"); ax.set_ylabel(r"$a$ [-]")
    ax.legend(fontsize=8); ax.grid(True)
    fig.tight_layout(); save_fig("Sens_Disc_a2_a_panel_count_cosine")

    # Cosine vs constant at baseline N: Cn
    fig, ax = plt.subplots(figsize=(8, 5))
    for dist, st in _DIST_STYLE.items():
        if (N_PANELS, dist) not in sens_disc_data: continue
        res, CT, CP = sens_disc_data[(N_PANELS, dist)]
        lbl = rf"{st['label_prefix']}  $C_T={CT:.3f}$" if not np.isnan(CT) \
              else st['label_prefix']
        ax.plot(res[:, 2], res[:, 3] / norm_val,
                color=st["color"], lw=2, ls=st["ls"], label=lbl)
    ax.set_xlabel("r/R"); ax.set_ylabel(r"$C_n$ [-]")
    ax.legend(fontsize=8); ax.grid(True)
    fig.tight_layout(); save_fig("Sens_Disc_b1_Cn_cosine_vs_constant")

    # Cosine vs constant at baseline N: a
    fig, ax = plt.subplots(figsize=(8, 5))
    for dist, st in _DIST_STYLE.items():
        if (N_PANELS, dist) not in sens_disc_data: continue
        res, CT, CP = sens_disc_data[(N_PANELS, dist)]
        lbl = rf"{st['label_prefix']}  $C_P={CP:.3f}$" if not np.isnan(CP) \
              else st['label_prefix']
        ax.plot(res[:, 2], res[:, 0],
                color=st["color"], lw=2, ls=st["ls"], label=lbl)
    ax.set_xlabel("r/R"); ax.set_ylabel(r"$a$ [-]")
    ax.legend(fontsize=8); ax.grid(True)
    fig.tight_layout(); save_fig("Sens_Disc_b2_a_cosine_vs_constant")

    # CT / CP convergence vs N  (both distributions)
    N_vals_cos = [N for N in SENS_N_LIST if (N, "cosine")   in sens_disc_data]
    N_vals_con = [N for N in SENS_N_LIST if (N, "constant") in sens_disc_data]
    ct_cos = np.array([sens_disc_data[(N, "cosine")][1]   for N in N_vals_cos])
    cp_cos = np.array([sens_disc_data[(N, "cosine")][2]   for N in N_vals_cos])
    ct_con = np.array([sens_disc_data[(N, "constant")][1] for N in N_vals_con])
    cp_con = np.array([sens_disc_data[(N, "constant")][2] for N in N_vals_con])

    # Use saved cosine CT/CP from npz where available (more reliable)
    if len(sens_N_CT_cos) > 0:
        for i, N in enumerate(SENS_N_LIST):
            if N in N_vals_cos and i < len(sens_N_CT_cos):
                j = N_vals_cos.index(N)
                ct_cos[j] = float(sens_N_CT_cos[i])
            if N in N_vals_cos and i < len(sens_N_CP_cos):
                j = N_vals_cos.index(N)
                cp_cos[j] = float(sens_N_CP_cos[i])

    for vals_cos, vals_con, ylabel, fname in [
            (ct_cos, ct_con, r"$C_T$ [-]", "Sens_Disc_c1_CT_convergence_vs_N"),
            (cp_cos, cp_con, r"$C_P$ [-]", "Sens_Disc_c2_CP_convergence_vs_N")]:
        fig, ax = plt.subplots(figsize=(8, 5))
        if len(N_vals_cos) > 0 and not np.all(np.isnan(vals_cos)):
            ax.plot(N_vals_cos, vals_cos, "o-",
                    color=_DIST_STYLE["cosine"]["color"], lw=2, label="Cosine")
        if len(N_vals_con) > 0 and not np.all(np.isnan(vals_con)):
            ax.plot(N_vals_con, vals_con, "s--",
                    color=_DIST_STYLE["constant"]["color"], lw=2, label="Constant")
        ax.axvline(N_PANELS, color="#949494", ls=":", lw=1.2,
                   label=rf"Baseline $N={N_PANELS}$")
        ax.set_xlabel(r"$N_\mathrm{panels}$ [-]")
        ax.set_ylabel(ylabel); ax.legend(fontsize=9); ax.grid(True)
        fig.tight_layout(); save_fig(fname)

    # Relative error vs N  (cosine, log scale)
    if len(N_vals_cos) > 1 and not np.all(np.isnan(ct_cos)):
        valid_ct = ct_cos[~np.isnan(ct_cos)]
        ct_ref   = valid_ct[-1]
        cp_ref   = cp_cos[~np.isnan(cp_cos)][-1]
        err_CT   = np.where(ct_cos == ct_ref, np.nan,
                            np.abs(ct_cos - ct_ref) / ct_ref * 100)
        err_CP   = np.where(cp_cos == cp_ref, np.nan,
                            np.abs(cp_cos - cp_ref) / cp_ref * 100)
        fig, ax = plt.subplots(figsize=(8, 5))
        mask_ct = ~np.isnan(err_CT)
        mask_cp = ~np.isnan(err_CP)
        if mask_ct.any():
            ax.semilogy(np.array(N_vals_cos)[mask_ct], err_CT[mask_ct],
                        "o-", color="#0173b2", lw=2, label=r"$\epsilon_{C_T}$ cosine")
        if mask_cp.any():
            ax.semilogy(np.array(N_vals_cos)[mask_cp], err_CP[mask_cp],
                        "s--", color="#0173b2", lw=2, alpha=0.6,
                        label=r"$\epsilon_{C_P}$ cosine")
        ax.axvline(N_PANELS, color="#949494", ls=":", lw=1.2,
                   label=rf"Baseline $N={N_PANELS}$")
        ax.set_xlabel(r"$N_\mathrm{panels}$ [-]")
        ax.set_ylabel(r"Relative error w.r.t. finest cosine [%]")
        ax.legend(fontsize=8); ax.grid(True, which="both")
        fig.tight_layout(); save_fig("Sens_Disc_d1_error_vs_N")

    # Multi-N comparison: cosine vs constant at 3 representative panel counts
    _CMP_N = [N for N in (5, 20, 70) if (N, "cosine")   in sens_disc_data
                                      and (N, "constant") in sens_disc_data]
    if _CMP_N:
        _n_colors = [_CB_PALETTE[i] for i in range(len(_CMP_N))]
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for ci, N in enumerate(_CMP_N):
            col = _n_colors[ci]
            res_cos, CT_cos, _ = sens_disc_data[(N, "cosine")]
            res_con, CT_con, _ = sens_disc_data[(N, "constant")]
            rR_cos = res_cos[:, 2];  Cn_cos = res_cos[:, 3] / norm_val
            rR_con = res_con[:, 2];  Cn_con = res_con[:, 3] / norm_val
            Cn_con_interp = np.interp(rR_cos, rR_con, Cn_con)
            ax.fill_between(rR_cos, Cn_cos, Cn_con_interp,
                            where=(Cn_con_interp > Cn_cos), color=col,
                            alpha=0.12, zorder=1, interpolate=True)
            lbl_cos = (rf"$N={N}$ cosine  $C_T={CT_cos:.3f}$"
                       if not np.isnan(CT_cos) else rf"$N={N}$ cosine")
            lbl_con = (rf"$N={N}$ constant  $C_T={CT_con:.3f}$"
                       if not np.isnan(CT_con) else rf"$N={N}$ constant")
            ax.plot(rR_cos, Cn_cos, color=col, lw=2.0, ls="-",
                    marker="o", ms=4, zorder=3, label=lbl_cos)
            ax.plot(rR_con, Cn_con, color=col, lw=2.0, ls="--",
                    marker="s", ms=4, alpha=0.55, zorder=2, label=lbl_con)
        ax.set_xlabel("r/R")
        ax.set_ylabel(r"$C_n = F_n\,/\,(\frac{1}{2}\rho U_\infty^2 R)$ [-]")
        ax.grid(True); ax.legend(fontsize=8, loc="upper left")
        ax.text(0.02, 0.02,
                "Shaded band: constant > cosine\n(over-predicted loading)",
                transform=ax.transAxes, fontsize=8, va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="#949494", alpha=0.85))
        fig.tight_layout(); save_fig("Sens_Disc_f1_Cn_cosine_vs_constant_multiN")

elif PLOT_SENS_DISC:
    _skip("Sens_Disc", "discretisation sensitivity data missing")

# =============================================================================
# 13.  SENSITIVITY: azimuthal step
# =============================================================================

print("Plotting sensitivity: azimuthal step")

if PLOT_SENS_DPSI and sens_dpsi_data:
    n_dpsi  = len(SENS_DPSI_LIST)
    dp_vals = [dp for dp in SENS_DPSI_LIST if dp in sens_dpsi_data]
    ct_dpsi = np.array([sens_dpsi_data[dp][1] for dp in dp_vals])
    cp_dpsi = np.array([sens_dpsi_data[dp][2] for dp in dp_vals])

    # Spanwise distributions
    for qty_col, ylabel, fname in [
            (0, r"$a$ [-]",                                       "Sens_DPSI_a_axial_induction"),
            (3, r"$C_n = F_n\,/\,(\frac{1}{2}\rho U_\infty^2 R)$ [-]", "Sens_DPSI_b_normal_loading")]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for idx, dp in enumerate(dp_vals):
            res, CT, CP = sens_dpsi_data[dp]
            y = res[:, qty_col] / norm_val if qty_col == 3 else res[:, qty_col]
            ax.plot(res[:, 2], y, color=_sens_color(idx, n_dpsi), lw=2,
                    label=rf"$\Delta\psi={dp:.0f}°$  $C_T={CT:.3f}$")
        ax.set_xlabel("r/R"); ax.set_ylabel(ylabel)
        ax.legend(fontsize=8); ax.grid(True)
        fig.tight_layout(); save_fig(fname)

    # CT vs dpsi
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dp_vals, ct_dpsi, "o-", color="#0173b2", lw=2)
    ax.axvline(DPSI_DEG, color="#949494", ls="--", lw=1.2,
               label=rf"Baseline $\Delta\psi={DPSI_DEG}°$")
    ax.set_xlabel(r"$\Delta\psi$ [deg]"); ax.set_ylabel(r"$C_T$ [-]")
    ax.legend(fontsize=9); ax.grid(True)
    fig.tight_layout(); save_fig("Sens_DPSI_c1_CT_vs_dpsi")

    # CP vs dpsi
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dp_vals, cp_dpsi, "o-", color="#029e73", lw=2)
    ax.axvline(DPSI_DEG, color="#949494", ls="--", lw=1.2,
               label=rf"Baseline $\Delta\psi={DPSI_DEG}°$")
    ax.set_xlabel(r"$\Delta\psi$ [deg]"); ax.set_ylabel(r"$C_P$ [-]")
    ax.legend(fontsize=9); ax.grid(True)
    fig.tight_layout(); save_fig("Sens_DPSI_c2_CP_vs_dpsi")

    # Relative error vs dpsi  (log scale, reference = finest/smallest step)
    ct_ref_d = ct_dpsi[0]; cp_ref_d = cp_dpsi[0]
    err_CT_d = np.where(ct_dpsi == ct_ref_d, np.nan,
                        np.abs(ct_dpsi - ct_ref_d) / ct_ref_d * 100)
    err_CP_d = np.where(cp_dpsi == cp_ref_d, np.nan,
                        np.abs(cp_dpsi - cp_ref_d) / cp_ref_d * 100)
    fig, ax = plt.subplots(figsize=(8, 5))
    m = ~np.isnan(err_CT_d)
    if m.any():
        ax.semilogy(np.array(dp_vals)[m], err_CT_d[m], "o-",
                    color="#0173b2", lw=2, label=r"$\epsilon_{C_T}$ [%]")
    m = ~np.isnan(err_CP_d)
    if m.any():
        ax.semilogy(np.array(dp_vals)[m], err_CP_d[m], "s-",
                    color="#029e73", lw=2, label=r"$\epsilon_{C_P}$ [%]")
    ax.axvline(DPSI_DEG, color="#949494", ls="--", lw=1.2,
               label=rf"Baseline $\Delta\psi={DPSI_DEG}°$")
    ax.set_xlabel(r"$\Delta\psi$ [deg]")
    ax.set_ylabel(r"Relative error w.r.t. finest case [%]")
    ax.legend(fontsize=9); ax.grid(True, which="both")
    fig.tight_layout(); save_fig("Sens_DPSI_d1_error_vs_dpsi")

elif PLOT_SENS_DPSI:
    _skip("Sens_DPSI", "azimuthal step sensitivity data missing")

# =============================================================================
# 14.  SENSITIVITY: wake length
# =============================================================================

print("Plotting sensitivity: wake length")

if PLOT_SENS_WAKE and sens_wake_data:
    n_nw    = len(SENS_NWAKE_LIST)
    nw_vals = [nw for nw in SENS_NWAKE_LIST if nw in sens_wake_data]
    ct_wake = np.array([sens_wake_data[nw][1] for nw in nw_vals])
    cp_wake = np.array([sens_wake_data[nw][2] for nw in nw_vals])

    # Convergence markers (0.5% threshold vs finest)
    _ct_ref   = ct_wake[-1]; _cp_ref = cp_wake[-1]; _tol = 0.005
    _nw_conv_CT = next((nw for nw, ct in zip(nw_vals, ct_wake)
                        if abs(ct - _ct_ref) < _tol), nw_vals[-1])
    _nw_conv_CP = next((nw for nw, cp in zip(nw_vals, cp_wake)
                        if abs(cp - _cp_ref) < _tol), nw_vals[-1])

    # Spanwise distributions
    for qty_col, ylabel, fname in [
            (0, r"$a$ [-]",                                       "Sens_Wake_a_axial_induction"),
            (3, r"$C_n = F_n\,/\,(\frac{1}{2}\rho U_\infty^2 R)$ [-]", "Sens_Wake_b_normal_loading")]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for idx, nw in enumerate(nw_vals):
            res, CT, CP = sens_wake_data[nw]
            y = res[:, qty_col] / norm_val if qty_col == 3 else res[:, qty_col]
            ax.plot(res[:, 2], y, color=_sens_color(idx, n_nw), lw=2,
                    label=rf"$N_{{wake}}={nw}$  $C_T={CT:.3f}$  $C_P={CP:.3f}$")
        ax.text(0.97, 0.97,
                rf"$C_T$ converged at $N_{{wake}}={_nw_conv_CT}$" + "\n"
                + rf"$C_P$ converged at $N_{{wake}}={_nw_conv_CP}$",
                transform=ax.transAxes, fontsize=8, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="#949494", alpha=0.85))
        ax.set_xlabel("r/R"); ax.set_ylabel(ylabel)
        ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
        ax.grid(True)
        fig.tight_layout(); save_fig(fname)

    # CT convergence vs N_wake
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(nw_vals, ct_wake, "o-", color="#0173b2", lw=2)
    ax.axvline(_nw_conv_CT, color="#949494", ls="--", lw=1.2,
               label=rf"Converged at $N_{{wake}}={_nw_conv_CT}$")
    ax.axhline(_ct_ref, color="#949494", ls=":", lw=1.0)
    ax.set_xlabel(r"Wake length $N_\mathrm{wake}$ [rotations]")
    ax.set_ylabel(r"$C_T$ [-]"); ax.legend(fontsize=9); ax.grid(True)
    fig.tight_layout(); save_fig("Sens_Wake_c1_CT_convergence_vs_Nwake")

    # CP convergence vs N_wake
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(nw_vals, cp_wake, "o-", color="#029e73", lw=2)
    ax.axvline(_nw_conv_CP, color="#949494", ls="--", lw=1.2,
               label=rf"Converged at $N_{{wake}}={_nw_conv_CP}$")
    ax.axhline(_cp_ref, color="#949494", ls=":", lw=1.0)
    ax.set_xlabel(r"Wake length $N_\mathrm{wake}$ [rotations]")
    ax.set_ylabel(r"$C_P$ [-]"); ax.legend(fontsize=9); ax.grid(True)
    fig.tight_layout(); save_fig("Sens_Wake_c2_CP_convergence_vs_Nwake")

elif PLOT_SENS_WAKE:
    _skip("Sens_Wake", "wake length sensitivity data missing")

# =============================================================================
# 15.  DONE
# =============================================================================

print("\n" + "="*60)
print(f"All plots saved  →  {OUT_DIR}")
print("="*60)