"""
PLOTTING_COMBINED.py  —  AE4135 Rotor/Wake Aerodynamics, Assignment 2
BEM vs Lifting Line comparison + LL sensitivity study plots.

Authors: Douwe de Jong (5313899), Martijn van Leeuwen (5614422)
================================================================
Loads:
    bem_results.npz      — from BEM_FINAL.py
    LLM_results.npz      — from LiftingLine_FINAL.py

Produces all plots for:
    Chapter 4  — LL results + BEM/LL comparison
    Chapter 5  — LL sensitivity study

Run:  python PLOTTING_COMBINED.py
"""

import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# 0.  PATHS  —  edit if your .npz files live elsewhere
# =============================================================================

_HERE        = os.path.dirname(os.path.abspath(__file__))
BEM_NPZ      = os.path.join(_HERE, "bem_results.npz")
LL_NPZ       = os.path.join(_HERE, "LLM_results.npz")
OUT_DIR      = os.path.join(_HERE, "combined_plots")
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# 1.  GLOBAL PLOT STYLE  (matches LiftingLine_FINAL.py)
# =============================================================================

mpl.rcParams.update({
    "text.usetex"       : False,
    "font.family"       : "serif",
    "font.serif"        : ["CMU Serif", "Computer Modern Roman",
                           "Latin Modern Roman", "DejaVu Serif"],
    "mathtext.fontset"  : "cm",
    "axes.labelsize"    : 12,
    "legend.fontsize"   : 10,
    "xtick.labelsize"   : 10,
    "ytick.labelsize"   : 10,
    "axes.titlesize"    : 12,
    "savefig.bbox"      : "tight",
    "savefig.pad_inches": 0.02,
    "legend.frameon"    : True,
})

# Seaborn colorblind palette  —  same as LiftingLine_FINAL.py
_CB  = sns.color_palette("colorblind").as_hex()

# Fixed per-TSR colours (indices into _CB)
_TSR_COL = {6: _CB[0], 8: _CB[1], 10: _CB[2]}

# Fixed per-parameter colours for sensitivity sweeps
_SENS_COL = lambda idx: _CB[idx % len(_CB)]

SHOW_PLOTS = False   # True → plt.show() after each save

# =============================================================================
# 2.  SAVE HELPER
# =============================================================================

def _savefig(fname: str) -> None:
    path = os.path.join(OUT_DIR, fname if fname.endswith(".pdf") else fname + ".pdf")
    plt.savefig(path)
    print(f"  Saved: {os.path.basename(path)}")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

# =============================================================================
# 3.  LOAD DATA
# =============================================================================

print(f"Loading BEM results  → {BEM_NPZ}")
if not os.path.exists(BEM_NPZ):
    sys.exit(f"  ERROR: {BEM_NPZ} not found. Run BEM_FINAL.py first.")
B = np.load(BEM_NPZ, allow_pickle=False)

print(f"Loading LL  results  → {LL_NPZ}")
if not os.path.exists(LL_NPZ):
    sys.exit(f"  ERROR: {LL_NPZ} not found. Run LiftingLine_FINAL.py first.")
L = np.load(LL_NPZ, allow_pickle=False)

# ── BEM per-TSR arrays --------------------------------------------------------
#  BEM column order: [a, aline, r_mid, fnorm, ftan, gamma, alpha, phi, cl, cd]
TSR_SPAN = [6, 8, 10]

def _bem(tsr: int) -> np.ndarray:
    key = f"sweep_res_{tsr}"
    if key not in B.files:
        sys.exit(f"  ERROR: key '{key}' missing from {BEM_NPZ}")
    return B[key]

bem = {tsr: _bem(tsr) for tsr in TSR_SPAN}

# BEM scalar performance (wide TSR sweep for 4.4 comparison)
bem_tsrs_perf = B["sweep_tsrs_perf"] if "sweep_tsrs_perf" in B.files else np.array(TSR_SPAN, dtype=float)
bem_CT_perf   = B["tsr_CT_perf"]     if "tsr_CT_perf"    in B.files else np.array([B["tsr_CT"][i] for i in range(3)])
bem_CP_perf   = B["tsr_CP_perf"]     if "tsr_CP_perf"    in B.files else np.array([B["tsr_CP"][i] for i in range(3)])

# Normalisation used in BEM loading plots
Radius   = float(B["cfg_Radius"])  if "cfg_Radius" in B.files else 50.0
U0       = float(B["cfg_U0"])      if "cfg_U0"     in B.files else 10.0
rho_bem  = float(B["cfg_rho"])     if "cfg_rho"    in B.files else 1.0
norm_bem = 0.5 * rho_bem * U0**2 * Radius   # matches BEM_FINAL.py

# BEM scalar CT/CP at the 3 span TSRs (for the comparison table)
bem_CT_span = {tsr: float(np.sum(bem[tsr][:,3] * np.gradient(bem[tsr][:,2]*Radius)
                                  * 3 / (0.5*rho_bem*U0**2*np.pi*Radius**2)))
               for tsr in TSR_SPAN}
if "tsr_CT" in B.files:
    for i, tsr in enumerate(TSR_SPAN):
        bem_CT_span[tsr] = float(B["tsr_CT"][i])
if "tsr_CP" in B.files:
    bem_CP_span = {tsr: float(B["tsr_CP"][i]) for i, tsr in enumerate(TSR_SPAN)}
else:
    bem_CP_span = {}

# ── LL per-TSR arrays ---------------------------------------------------------
#  LL column order: [a, aprime, r/R, Fn, Ft, Gamma, alpha, phi, Cl, Cd]
def _ll(tsr: int) -> np.ndarray:
    key = f"sweep_res_{tsr}"
    if key not in L.files:
        sys.exit(f"  ERROR: key '{key}' missing from {LL_NPZ}")
    return L[key]

ll = {tsr: _ll(tsr) for tsr in TSR_SPAN}

# LL normalisation  (rho=1.0 as set in LiftingLine_FINAL.py)
rho_ll   = 1.0
norm_ll  = 0.5 * rho_ll * U0**2 * Radius

# LL scalar performance (wide TSR sweep, λ=4..12)
ll_tsrs_perf = L["perf_tsrs"] if "perf_tsrs" in L.files else np.array(TSR_SPAN, dtype=float)
ll_CT_perf   = L["perf_CT"]   if "perf_CT"   in L.files else np.array([L["tsr_CT"][i] for i in range(3)])
ll_CP_perf   = L["perf_CP"]   if "perf_CP"   in L.files else np.array([L["tsr_CP"][i] for i in range(3)])

# LL scalar CT/CP at the 3 span TSRs
ll_CT_span = {tsr: float(L["tsr_CT"][i]) for i, tsr in enumerate(TSR_SPAN)} \
             if "tsr_CT" in L.files else {}
ll_CP_span = {tsr: float(L["tsr_CP"][i]) for i, tsr in enumerate(TSR_SPAN)} \
             if "tsr_CP" in L.files else {}

# ── LL sensitivity arrays ------------------------------------------------------
def _lget(key, default=None):
    return L[key] if key in L.files else default

sens_aw_vals    = _lget("sens_aw_vals",    np.array([]))
sens_aw_CT      = _lget("sens_aw_CT",      np.array([]))
sens_aw_CP      = _lget("sens_aw_CP",      np.array([]))

sens_dpsi_vals  = _lget("sens_dpsi_vals",  np.array([]))
sens_dpsi_CT    = _lget("sens_dpsi_CT",    np.array([]))
sens_dpsi_CP    = _lget("sens_dpsi_CP",    np.array([]))

sens_nwake_vals = _lget("sens_nwake_vals", np.array([]))
sens_nwake_CT   = _lget("sens_nwake_CT",   np.array([]))
sens_nwake_CP   = _lget("sens_nwake_CP",   np.array([]))

sens_N_vals     = _lget("sens_N_vals",     np.array([]))
sens_N_CT_cos   = _lget("sens_N_CT_cosine",np.array([]))
sens_N_CP_cos   = _lget("sens_N_CP_cosine",np.array([]))

print(f"  LL sensitivity aw   : {sens_aw_vals}")
print(f"  LL sensitivity dpsi : {sens_dpsi_vals}")
print(f"  LL sensitivity nwake: {sens_nwake_vals}")
print(f"  LL sensitivity N    : {sens_N_vals}")

# helper to load per-span sensitivity result arrays
def _sens_arr(prefix, val):
    key = f"{prefix}{val}"
    return L[key] if key in L.files else None

# =============================================================================
# 4.  CHAPTER 4  —  LL RESULTS + BEM COMPARISON
# =============================================================================

print("\n" + "="*60)
print("Chapter 4 — LL results + BEM comparison")
print("="*60)

# ---------------------------------------------------------------------------
# 4.1  Inflow angle and AoA
# ---------------------------------------------------------------------------

# ── Fig 4.1-A: LL only — inflow angle ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
for tsr in TSR_SPAN:
    ax.plot(ll[tsr][:,2], ll[tsr][:,7],
            color=_TSR_COL[tsr], lw=2, label=rf"$\lambda={tsr}$")
ax.set_xlabel("r/R")
ax.set_ylabel(r"$\phi$ [deg]")
ax.legend(); ax.grid(True)
fig.tight_layout(); _savefig("4_1a_LL_inflow_angle_vs_rR")

# ── Fig 4.1-B: LL only — angle of attack ───────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
for tsr in TSR_SPAN:
    ax.plot(ll[tsr][:,2], ll[tsr][:,6],
            color=_TSR_COL[tsr], lw=2, label=rf"$\lambda={tsr}$")
ax.set_xlabel("r/R")
ax.set_ylabel(r"$\alpha$ [deg]")
ax.legend(); ax.grid(True)
fig.tight_layout(); _savefig("4_1b_LL_angle_of_attack_vs_rR")

# ── Fig 4.1-C: BEM vs LL — inflow angle (subplots per TSR) ─────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for ax, tsr in zip(axes, TSR_SPAN):
    col = _TSR_COL[tsr]
    ax.plot(ll[tsr][:,2],  ll[tsr][:,7],  color=col, lw=2,   ls="-",  label=rf"LLM ($\lambda={tsr}$)")
    ax.plot(bem[tsr][:,2], bem[tsr][:,7], color="k", lw=2,   ls="--", label=rf"BEM ($\lambda={tsr}$)")
    ax.set_xlabel("r/R")
    ax.legend(fontsize=9); ax.grid(True)
axes[0].set_ylabel(r"$\phi$ [deg]")
fig.tight_layout(); _savefig("4_1c_BEMvsLL_inflow_angle_vs_rR")

# ── Fig 4.1-D: BEM vs LL — angle of attack ─────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for ax, tsr in zip(axes, TSR_SPAN):
    col = _TSR_COL[tsr]
    ax.plot(ll[tsr][:,2],  ll[tsr][:,6],  color=col, lw=2, ls="-",  label=rf"LLM ($\lambda={tsr}$)")
    ax.plot(bem[tsr][:,2], bem[tsr][:,6], color="k", lw=2, ls="--", label=rf"BEM ($\lambda={tsr}$)")
    ax.set_xlabel("r/R")
    ax.legend(fontsize=9); ax.grid(True)
axes[0].set_ylabel(r"$\alpha$ [deg]")
fig.tight_layout(); _savefig("4_1d_BEMvsLL_angle_of_attack_vs_rR")

# ---------------------------------------------------------------------------
# 4.2  Induction factors
# ---------------------------------------------------------------------------

# ── Fig 4.2-A: LL only — axial induction ───────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
for tsr in TSR_SPAN:
    ax.plot(ll[tsr][:,2], ll[tsr][:,0],
            color=_TSR_COL[tsr], lw=2, label=rf"$\lambda={tsr}$")
ax.set_xlabel("r/R"); ax.set_ylabel(r"$a$ [-]")
ax.legend(); ax.grid(True)
fig.tight_layout(); _savefig("4_2a_LL_axial_induction_vs_rR")

# ── Fig 4.2-B: LL only — tangential induction ──────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
for tsr in TSR_SPAN:
    ax.plot(ll[tsr][:,2], ll[tsr][:,1],
            color=_TSR_COL[tsr], lw=2, label=rf"$\lambda={tsr}$")
ax.set_xlabel("r/R"); ax.set_ylabel(r"$a'$ [-]")
ax.legend(); ax.grid(True)
fig.tight_layout(); _savefig("4_2b_LL_tangential_induction_vs_rR")

# ── Fig 4.2-C: BEM vs LL — axial induction ─────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for ax, tsr in zip(axes, TSR_SPAN):
    col = _TSR_COL[tsr]
    ax.plot(ll[tsr][:,2],  ll[tsr][:,0],  color=col, lw=2, ls="-",  label=rf"LLM ($\lambda={tsr}$)")
    ax.plot(bem[tsr][:,2], bem[tsr][:,0], color="k", lw=2, ls="--", label=rf"BEM ($\lambda={tsr}$)")
    ax.set_xlabel("r/R")
    ax.legend(fontsize=9); ax.grid(True)
axes[0].set_ylabel(r"$a$ [-]")
fig.tight_layout(); _savefig("4_2c_BEMvsLL_axial_induction_vs_rR")

# ── Fig 4.2-D: BEM vs LL — tangential induction ────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for ax, tsr in zip(axes, TSR_SPAN):
    col = _TSR_COL[tsr]
    ax.plot(ll[tsr][:,2],  ll[tsr][:,1],  color=col, lw=2, ls="-",  label=rf"LLM ($\lambda={tsr}$)")
    ax.plot(bem[tsr][:,2], bem[tsr][:,1], color="k", lw=2, ls="--", label=rf"BEM ($\lambda={tsr}$)")
    ax.set_xlabel("r/R")
    ax.legend(fontsize=9); ax.grid(True)
axes[0].set_ylabel(r"$a'$ [-]")
fig.tight_layout(); _savefig("4_2d_BEMvsLL_tangential_induction_vs_rR")

# ---------------------------------------------------------------------------
# 4.3  Loading distributions
# ---------------------------------------------------------------------------

# ── Fig 4.3-A: LL only — axial loading ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
for tsr in TSR_SPAN:
    ax.plot(ll[tsr][:,2], ll[tsr][:,3] / norm_ll,
            color=_TSR_COL[tsr], lw=2, label=rf"$\lambda={tsr}$")
ax.set_xlabel("r/R")
ax.set_ylabel(r"$C_n = F_n\,/\,(\frac{1}{2}\rho U_0^2 R)$")
ax.legend(); ax.grid(True)
fig.tight_layout(); _savefig("4_3a_LL_normal_loading_Cn_vs_rR")

# ── Fig 4.3-B: LL only — azimuthal loading ─────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
for tsr in TSR_SPAN:
    ax.plot(ll[tsr][:,2], ll[tsr][:,4] / norm_ll,
            color=_TSR_COL[tsr], lw=2, label=rf"$\lambda={tsr}$")
ax.set_xlabel("r/R")
ax.set_ylabel(r"$C_t = F_t\,/\,(\frac{1}{2}\rho U_0^2 R)$")
ax.legend(); ax.grid(True)
fig.tight_layout(); _savefig("4_3b_LL_azimuthal_loading_Ct_vs_rR")

# ── Fig 4.3-C: BEM vs LL — axial loading ───────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for ax, tsr in zip(axes, TSR_SPAN):
    col = _TSR_COL[tsr]
    ax.plot(ll[tsr][:,2],  ll[tsr][:,3]  / norm_ll,  color=col, lw=2, ls="-",  label=rf"LLM ($\lambda={tsr}$)")
    ax.plot(bem[tsr][:,2], bem[tsr][:,3] / norm_bem, color="k", lw=2, ls="--", label=rf"BEM ($\lambda={tsr}$)")
    ax.set_xlabel("r/R")
    ax.legend(fontsize=9); ax.grid(True)
axes[0].set_ylabel(r"$C_n$ [-]")
fig.tight_layout(); _savefig("4_3c_BEMvsLL_normal_loading_Cn_vs_rR")

# ── Fig 4.3-D: BEM vs LL — azimuthal loading ───────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for ax, tsr in zip(axes, TSR_SPAN):
    col = _TSR_COL[tsr]
    ax.plot(ll[tsr][:,2],  ll[tsr][:,4]  / norm_ll,  color=col, lw=2, ls="-",  label=rf"LLM ($\lambda={tsr}$)")
    ax.plot(bem[tsr][:,2], bem[tsr][:,4] / norm_bem, color="k", lw=2, ls="--", label=rf"BEM ($\lambda={tsr}$)")
    ax.set_xlabel("r/R")
    ax.legend(fontsize=9); ax.grid(True)
axes[0].set_ylabel(r"$C_t$ [-]")
fig.tight_layout(); _savefig("4_3d_BEMvsLL_azimuthal_loading_Ct_vs_rR")

# ---------------------------------------------------------------------------
# 4.4  Total performance CT and CP
# ---------------------------------------------------------------------------

# ── Fig 4.4-A: LL only — CT vs λ (4..12) ──────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ll_tsrs_perf, ll_CT_perf, "o-", color=_CB[0], lw=2, ms=5)
for tsr in TSR_SPAN:
    if tsr in ll_CT_span:
        ax.plot(tsr, ll_CT_span[tsr], "o", color=_CB[0],
                ms=10, mec="black", mew=1.5,
                label=rf"$\lambda={tsr}$ (span sweep)")
ax.set_xlabel(r"Tip-speed ratio $\lambda$ [-]")
ax.set_ylabel(r"$C_T$ [-]")
ax.set_xlim(3.5, 12.5); ax.set_xticks(np.arange(4, 13, 1))
ax.legend(fontsize=9); ax.grid(True)
fig.tight_layout(); _savefig("4_4a_LL_CT_vs_TSR")

# ── Fig 4.4-B: LL only — CP vs λ (4..12) ──────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ll_tsrs_perf, ll_CP_perf, "o-", color=_CB[2], lw=2, ms=5)
ax.axhline(16/27, color="grey", ls=":", lw=1.2, label="Betz limit")
for tsr in TSR_SPAN:
    if tsr in ll_CP_span:
        ax.plot(tsr, ll_CP_span[tsr], "o", color=_CB[2],
                ms=10, mec="black", mew=1.5,
                label=rf"$\lambda={tsr}$ (span sweep)")
ax.set_xlabel(r"Tip-speed ratio $\lambda$ [-]")
ax.set_ylabel(r"$C_P$ [-]")
ax.set_xlim(3.5, 12.5); ax.set_xticks(np.arange(4, 13, 1))
ax.legend(fontsize=9); ax.grid(True)
fig.tight_layout(); _savefig("4_4b_LL_CP_vs_TSR")

# ── Fig 4.4-C: BEM vs LL — CT comparison ────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ll_tsrs_perf,  ll_CT_perf,  "o-", color=_CB[0], lw=2, ms=5, label="LLM")
ax.plot(bem_tsrs_perf, bem_CT_perf, "s--",color="k", lw=2, ms=5, label="BEM")
ax.set_xlabel(r"Tip-speed ratio $\lambda$ [-]")
ax.set_ylabel(r"$C_T$ [-]")
ax.set_xlim(3.5, 12.5); ax.set_xticks(np.arange(4, 13, 1))
ax.legend(); ax.grid(True)
fig.tight_layout(); _savefig("4_4c_BEMvsLL_CT_vs_TSR")

# ── Fig 4.4-D: BEM vs LL — CP comparison ────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ll_tsrs_perf,  ll_CP_perf,  "o-", color=_CB[2], lw=2, ms=5, label="LLM")
ax.plot(bem_tsrs_perf, bem_CP_perf, "s--",color="k", lw=2, ms=5, label="BEM")
ax.axhline(16/27, color="grey", ls=":", lw=1.2, label="Betz limit")
ax.set_xlabel(r"Tip-speed ratio $\lambda$ [-]")
ax.set_ylabel(r"$C_P$ [-]")
ax.set_xlim(3.5, 12.5); ax.set_xticks(np.arange(4, 13, 1))
ax.legend(); ax.grid(True)
fig.tight_layout(); _savefig("4_4d_BEMvsLL_CP_vs_TSR")

# ── Scalar CT/CP comparison table (printed to terminal) ─────────────────────
print("\n  CT / CP comparison table (LL vs BEM):")
print(f"  {'TSR':>4} {'CT_LL':>8} {'CT_BEM':>8} {'dCT%':>7} {'CP_LL':>8} {'CP_BEM':>8} {'dCP%':>7}")
print("  " + "-"*56)
for tsr in TSR_SPAN:
    ct_ll  = ll_CT_span.get(tsr, float("nan"))
    ct_bem = bem_CT_span.get(tsr, float("nan"))
    cp_ll  = ll_CP_span.get(tsr, float("nan"))
    cp_bem = bem_CP_span.get(tsr, float("nan"))
    dct = (ct_ll - ct_bem) / ct_bem * 100 if ct_bem else float("nan")
    dcp = (cp_ll - cp_bem) / cp_bem * 100 if cp_bem else float("nan")
    print(f"  {tsr:>4} {ct_ll:>8.4f} {ct_bem:>8.4f} {dct:>+7.2f}% "
          f"{cp_ll:>8.4f} {cp_bem:>8.4f} {dcp:>+7.2f}%")

# ---------------------------------------------------------------------------
# 4.5  Circulation distribution  (LL only — no BEM equivalent)
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 5))
for tsr in TSR_SPAN:
    r_R    = ll[tsr][:,2]
    Gamma  = ll[tsr][:,5]
    Omega  = U0 * tsr / Radius
    B    = 3
    Gamma_hat = Gamma * B * Omega / (np.pi * U0**2)
    ax.plot(r_R, Gamma_hat, color=_TSR_COL[tsr], lw=2, label=rf"$\lambda={tsr}$")
ax.set_xlabel("r/R")
ax.set_ylabel(r"$\hat{\Gamma} = \Gamma\,B\,\Omega\,/\,(\pi U_0^2)$ [-]")
ax.legend(); ax.grid(True)
fig.tight_layout(); _savefig("4_5_LL_circulation_vs_rR")

# =============================================================================
# 5.  CHAPTER 5  —  LL SENSITIVITY STUDY
# =============================================================================

print("\n" + "="*60)
print("Chapter 5 — LL sensitivity study")
print("="*60)

_BL_TSR  = 8
_BL_CT   = float(ll_CT_span.get(_BL_TSR, float("nan")))
_BL_CP   = float(ll_CP_span.get(_BL_TSR, float("nan")))
_BL_RES  = ll[8]

def _Cn(res): return res[:,3] / norm_ll
def _a(res):  return res[:,0]
def _rR(res): return res[:,2]

# ---------------------------------------------------------------------------
# 5.1  Sensitivity: wake convection speed a_w
# ---------------------------------------------------------------------------
print("\n  5.1 — convection speed (a_w)")

if len(sens_aw_vals) > 0:
    aw_res = [_sens_arr("sens_aw_res_", aw) for aw in sens_aw_vals]

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, (aw, res) in enumerate(zip(sens_aw_vals, aw_res)):
        if res is None: continue
        ct_val = float(sens_aw_CT[idx]) if idx < len(sens_aw_CT) else float("nan")
        ax.plot(_rR(res), _Cn(res), color=_SENS_COL(idx), lw=2,
                label=rf"$a_w={aw:.2f}$  $C_T={ct_val:.3f}$")
    ax.set_xlabel("r/R")
    ax.set_ylabel(r"$C_n = F_n\,/\,(\frac{1}{2}\rho U_0^2 R)$")
    ax.legend(fontsize=9); ax.grid(True)
    fig.tight_layout(); _savefig("5_1a_sens_aw_normal_loading_vs_rR")

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, (aw, res) in enumerate(zip(sens_aw_vals, aw_res)):
        if res is None: continue
        ax.plot(_rR(res), _a(res), color=_SENS_COL(idx), lw=2,
                label=rf"$a_w={aw:.2f}$")
    ax.set_xlabel("r/R"); ax.set_ylabel(r"$a$ [-]")
    ax.legend(fontsize=9); ax.grid(True)
    fig.tight_layout(); _savefig("5_1b_sens_aw_axial_induction_vs_rR")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(sens_aw_vals, sens_aw_CT, "o-", color=_CB[0], lw=2)
    axes[0].axhline(_BL_CT, color="grey", ls="--", lw=1, label=f"baseline CT={_BL_CT:.4f}")
    axes[0].set_xlabel(r"Wake induction $a_w$ [-]")
    axes[0].set_ylabel(r"$C_T$ [-]")
    axes[0].legend(fontsize=9); axes[0].grid(True)

    axes[1].plot(sens_aw_vals, sens_aw_CP, "o-", color=_CB[2], lw=2)
    axes[1].axhline(_BL_CP, color="grey", ls="--", lw=1, label=f"baseline CP={_BL_CP:.4f}")
    axes[1].set_xlabel(r"Wake induction $a_w$ [-]")
    axes[1].set_ylabel(r"$C_P$ [-]")
    axes[1].legend(fontsize=9); axes[1].grid(True)
    fig.tight_layout(); _savefig("5_1c_sens_aw_CT_CP_convergence")
else:
    print("    [SKIP] a_w sensitivity arrays not found in LL npz")

# ---------------------------------------------------------------------------
# 5.2  Sensitivity: blade discretisation (N + cosine vs constant)
# ---------------------------------------------------------------------------
print("\n  5.2 — blade discretisation (N, spacing)")

N_vals_avail = [int(n) for n in sens_N_vals] if len(sens_N_vals) > 0 else []

if len(N_vals_avail) >= 2:
    N_lo = N_vals_avail[0]
    N_hi = N_vals_avail[-1]

    def _disc_res(N, dist):
        key = f"sens_disc_res_{N}_{dist}"
        return L[key] if key in L.files else None

    fig, ax = plt.subplots(figsize=(8, 5))
    pairs = [(N_lo, "cosine", _CB[0], "-"),
             (N_hi, "cosine", _CB[0], "--"),
             (N_lo, "constant", _CB[1], "-"),
             (N_hi, "constant", _CB[1], "--")]
    for N, dist, col, ls in pairs:
        res = _disc_res(N, dist)
        if res is None: continue
        ax.plot(_rR(res), _Cn(res), color=col, lw=2, ls=ls,
                label=f"N={N}, {dist}")
    
    from matplotlib.lines import Line2D
    custom = [Line2D([0],[0], color=_CB[0], lw=2, label="Cosine"),
              Line2D([0],[0], color=_CB[1], lw=2, label="Constant"),
              Line2D([0],[0], color="k",    lw=2, ls="-",  label=f"N={N_lo}"),
              Line2D([0],[0], color="k",    lw=2, ls="--", label=f"N={N_hi}")]
    ax.legend(handles=custom, fontsize=9); ax.grid(True)
    ax.set_xlabel("r/R")
    ax.set_ylabel(r"$C_n = F_n\,/\,(\frac{1}{2}\rho U_0^2 R)$")
    fig.tight_layout(); _savefig("5_2a_sens_disc_normal_loading_Cn_vs_rR")

    fig, ax = plt.subplots(figsize=(8, 5))
    for N, dist, col, ls in pairs:
        res = _disc_res(N, dist)
        if res is None: continue
        ax.plot(_rR(res), _a(res), color=col, lw=2, ls=ls,
                label=f"N={N}, {dist}")
    ax.legend(handles=custom, fontsize=9); ax.grid(True)
    ax.set_xlabel("r/R"); ax.set_ylabel(r"$a$ [-]")
    fig.tight_layout(); _savefig("5_2b_sens_disc_axial_induction_vs_rR")

    if len(sens_N_vals) > 0 and len(sens_N_CT_cos) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(sens_N_vals, sens_N_CT_cos, "o-", color=_CB[0], lw=2, label="Cosine")
        axes[0].axhline(_BL_CT, color="grey", ls="--", lw=1,
                         label=f"baseline CT={_BL_CT:.4f}")
        axes[0].set_xlabel("Number of panels N")
        axes[0].set_ylabel(r"$C_T$ [-]")
        axes[0].legend(fontsize=9); axes[0].grid(True)

        axes[1].plot(sens_N_vals, sens_N_CP_cos, "o-", color=_CB[2], lw=2, label="Cosine")
        axes[1].axhline(_BL_CP, color="grey", ls="--", lw=1,
                         label=f"baseline CP={_BL_CP:.4f}")
        axes[1].set_xlabel("Number of panels N")
        axes[1].set_ylabel(r"$C_P$ [-]")
        axes[1].legend(fontsize=9); axes[1].grid(True)
        fig.tight_layout(); _savefig("5_2c_sens_disc_CT_CP_convergence_vs_N")
else:
    print("    [SKIP] disc sensitivity arrays not found or insufficient N values")

# ---------------------------------------------------------------------------
# 5.3  Sensitivity: azimuthal step Δψ
# ---------------------------------------------------------------------------
print("\n  5.3 — azimuthal step (dpsi)")

if len(sens_dpsi_vals) > 0:
    dpsi_res = [_sens_arr("sens_dpsi_res_", float(dp)) for dp in sens_dpsi_vals]

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, (dp, res) in enumerate(zip(sens_dpsi_vals, dpsi_res)):
        if res is None: continue
        ct_val = float(sens_dpsi_CT[idx]) if idx < len(sens_dpsi_CT) else float("nan")
        n_steps = int(round(360.0 / dp))
        ax.plot(_rR(res), _Cn(res), color=_SENS_COL(idx), lw=2,
                label=rf"$\Delta\psi={dp:.0f}°$ ({n_steps} steps)  $C_T={ct_val:.3f}$")
    ax.set_xlabel("r/R")
    ax.set_ylabel(r"$C_n = F_n\,/\,(\frac{1}{2}\rho U_0^2 R)$")
    ax.legend(fontsize=9); ax.grid(True)
    fig.tight_layout(); _savefig("5_3a_sens_dpsi_normal_loading_vs_rR")

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, (dp, res) in enumerate(zip(sens_dpsi_vals, dpsi_res)):
        if res is None: continue
        ax.plot(_rR(res), _a(res), color=_SENS_COL(idx), lw=2,
                label=rf"$\Delta\psi={dp:.0f}°$")
    ax.set_xlabel("r/R"); ax.set_ylabel(r"$a$ [-]")
    ax.legend(fontsize=9); ax.grid(True)
    fig.tight_layout(); _savefig("5_3b_sens_dpsi_axial_induction_vs_rR")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(sens_dpsi_vals, sens_dpsi_CT, "o-", color=_CB[0], lw=2)
    axes[0].axhline(_BL_CT, color="grey", ls="--", lw=1,
                     label=f"baseline CT={_BL_CT:.4f}")
    axes[0].set_xlabel(r"Azimuthal step $\Delta\psi$ [deg]")
    axes[0].set_ylabel(r"$C_T$ [-]")
    axes[0].legend(fontsize=9); axes[0].grid(True)

    axes[1].plot(sens_dpsi_vals, sens_dpsi_CP, "o-", color=_CB[2], lw=2)
    axes[1].axhline(_BL_CP, color="grey", ls="--", lw=1,
                     label=f"baseline CP={_BL_CP:.4f}")
    axes[1].set_xlabel(r"Azimuthal step $\Delta\psi$ [deg]")
    axes[1].set_ylabel(r"$C_P$ [-]")
    axes[1].legend(fontsize=9); axes[1].grid(True)
    fig.tight_layout(); _savefig("5_3c_sens_dpsi_CT_CP_convergence")
else:
    print("    [SKIP] dpsi sensitivity arrays not found in LL npz")

# ---------------------------------------------------------------------------
# 5.4  Sensitivity: wake length N_wake
# ---------------------------------------------------------------------------
print("\n  5.4 — wake length (N_wake)")

if len(sens_nwake_vals) > 0:
    nwake_res = [_sens_arr("sens_nwake_res_", int(nw)) for nw in sens_nwake_vals]

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, (nw, res) in enumerate(zip(sens_nwake_vals, nwake_res)):
        if res is None: continue
        ct_val = float(sens_nwake_CT[idx]) if idx < len(sens_nwake_CT) else float("nan")
        ax.plot(_rR(res), _Cn(res), color=_SENS_COL(idx), lw=2,
                label=rf"$N_{{wake}}={int(nw)}$  $C_T={ct_val:.3f}$")
    ax.set_xlabel("r/R")
    ax.set_ylabel(r"$C_n = F_n\,/\,(\frac{1}{2}\rho U_0^2 R)$")
    ax.legend(fontsize=9); ax.grid(True)
    fig.tight_layout(); _savefig("5_4a_sens_nwake_normal_loading_vs_rR")

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, (nw, res) in enumerate(zip(sens_nwake_vals, nwake_res)):
        if res is None: continue
        ax.plot(_rR(res), _a(res), color=_SENS_COL(idx), lw=2,
                label=rf"$N_{{wake}}={int(nw)}$")
    ax.set_xlabel("r/R"); ax.set_ylabel(r"$a$ [-]")
    ax.legend(fontsize=9); ax.grid(True)
    fig.tight_layout(); _savefig("5_4b_sens_nwake_axial_induction_vs_rR")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(sens_nwake_vals, sens_nwake_CT, "o-", color=_CB[0], lw=2)
    axes[0].axhline(_BL_CT, color="grey", ls="--", lw=1,
                     label=f"baseline CT={_BL_CT:.4f}")
    axes[0].set_xlabel(r"Wake length $N_\mathrm{wake}$ [rotations]")
    axes[0].set_ylabel(r"$C_T$ [-]")
    axes[0].legend(fontsize=9); axes[0].grid(True)

    axes[1].plot(sens_nwake_vals, sens_nwake_CP, "o-", color=_CB[2], lw=2)
    axes[1].axhline(_BL_CP, color="grey", ls="--", lw=1,
                     label=f"baseline CP={_BL_CP:.4f}")
    axes[1].set_xlabel(r"Wake length $N_\mathrm{wake}$ [rotations]")
    axes[1].set_ylabel(r"$C_P$ [-]")
    axes[1].legend(fontsize=9); axes[1].grid(True)
    fig.tight_layout(); _savefig("5_4c_sens_nwake_CT_CP_convergence")
else:
    print("    [SKIP] N_wake sensitivity arrays not found in LL npz")

# =============================================================================
# 6.  SUMMARY
# =============================================================================

print("\n" + "="*60)
print(f"All plots saved to: {OUT_DIR}")
print("="*60)
print("\nFile list:")
for f in sorted(os.listdir(OUT_DIR)):
    if f.endswith(".pdf"):
        print(f"  {f}")