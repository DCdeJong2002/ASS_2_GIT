"""
PLOTTING_BEM_FINAL.py  —  Standalone plotting script

Authors: Douwe de Jong(5313899), Martijn van Leeuwen(5614422)
================================================
Loads bem_results.npz and opt_results.npz (produced by BEM_FINAL.py) and
reproduces all plots without re-running any BEM. In this way the full
optimization does not need to be run to verify the results.

Usage
-----
    python PLOTTING_BEM_FINAL.py                           # uses bem_results.npz and opt_results.npz in cwd

Plots saved to ./plotting_plots_assignment/
"""

import os, sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns

# =============================================================================
# GLOBAL PLOT STYLE
# =============================================================================

mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["CMU Serif", "Computer Modern Roman", "Latin Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",

    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.titlesize": 12,

    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,

    "legend.frameon": True,
})

# Seaborn colorblind palette (8 colours)
_CB_PALETTE = sns.color_palette("colorblind", 8)
_CB_HEX     = [mpl.colors.to_hex(c) for c in _CB_PALETTE]

# =============================================================================
# MENU — choose which plots to produce and whether to display them
# =============================================================================

SHOW_PLOTS = False   # False -> save and close immediately
                     # True  -> plt.show() after each save

PLOT_4_1   = True
PLOT_4_2   = True
PLOT_4_3   = True
PLOT_4_4   = True
PLOT_5     = True
PLOT_6     = True
PLOT_7     = True
PLOT_8     = True
PLOT_9     = True
PLOT_9_CMP = True
PLOT_10    = True

# =============================================================================
# 1.  LOAD RESULTS — two separate npz files
# =============================================================================

_script_dir = os.path.dirname(os.path.abspath(__file__))

if len(sys.argv) >= 3:
    bem_path = sys.argv[1]
    opt_path = sys.argv[2]
elif len(sys.argv) == 2:
    bem_path = sys.argv[1]
    opt_path = os.path.join(_script_dir, "opt_results.npz")
else:
    bem_path = os.path.join(_script_dir, "bem_results.npz")
    opt_path = os.path.join(_script_dir, "opt_results.npz")

if not os.path.exists(bem_path):
    raise FileNotFoundError(
        f"Cannot find BEM results file: '{bem_path}'\n"
        "Run BEM_FINAL.py with SAVE_BEM_RESULTS=True first.")

_opt_available = os.path.exists(opt_path)
if not _opt_available:
    print(f"  [INFO] opt_results.npz not found at '{opt_path}' — "
          "PLOT_8/9/10 will be skipped.")

print(f"Loading BEM results from : {bem_path}")
B = np.load(bem_path, allow_pickle=False)

if _opt_available:
    print(f"Loading opt results from : {opt_path}")
    O = np.load(opt_path, allow_pickle=False)
else:
    O = {}

def _bget(key, default=None):
    try:    return B[key]
    except KeyError: return default

def _oget(key, default=None):
    try:    return O[key]
    except (KeyError, TypeError): return default

def _getf(D, key, default=None):
    try:
        v = D[key]; return float(v)
    except (KeyError, TypeError): return default

def _arr_or_none(D, key):
    try:
        v = D[key]; return v if v.size > 0 else None
    except (KeyError, TypeError): return None

# ── Configuration scalars ─────────────────────────────────────────────────────
_cfg = O if _opt_available else B
Radius         = _getf(_cfg, "cfg_Radius",         50.0)
NBlades        = int(_getf(_cfg, "cfg_NBlades",    3))
U0             = _getf(_cfg, "cfg_U0",             10.0)
rho            = _getf(_cfg, "cfg_rho",            1.0)
RootLocation_R = _getf(_cfg, "cfg_RootLocation_R", 0.2)
TipLocation_R  = _getf(_cfg, "cfg_TipLocation_R",  1.0)
Pitch          = _getf(_cfg, "cfg_Pitch",          -2.0)
CHORD_ROOT     = _getf(_cfg, "cfg_CHORD_ROOT",     3.4)
CHORD_MIN      = _getf(_cfg, "cfg_CHORD_MIN",      0.3)
CT_TARGET      = _getf(_cfg, "cfg_CT_TARGET",      0.75)
TSR_DESIGN     = _getf(_cfg, "cfg_TSR_DESIGN",     8.0)
DELTA_R_R      = _getf(_cfg, "cfg_DELTA_R_R",      0.005)

# ── Polar ─────────────────────────────────────────────────────────────────────
polar_alpha = B["polar_alpha"]
polar_cl    = B["polar_cl"]
polar_cd    = B["polar_cd"]

# ── Spanwise sweep ────────────────────────────────────────────────────────────
_all_span_tsrs  = [int(t) for t in B["sweep_tsrs"]]
_span_mask      = [f"sweep_res_{int(t)}" in B.files for t in _all_span_tsrs]
TSR_SWEEP_SPAN  = [t for t, ok in zip(_all_span_tsrs, _span_mask) if ok]
tsr_CT_span     = B["tsr_CT"][_span_mask]
tsr_CP_span     = B["tsr_CP"][_span_mask]
sweep_data_span = {t: B[f"sweep_res_{t}"] for t in TSR_SWEEP_SPAN}

# ── Performance sweep ─────────────────────────────────────────────────────────
_perf_tsrs = _arr_or_none(B, "sweep_tsrs_perf")
if _perf_tsrs is not None and len(_perf_tsrs) > 0:
    _raw_tsrs = [float(t) for t in _perf_tsrs]
    _raw_ct   = list(B["tsr_CT_perf"])
    _raw_cp   = list(B["tsr_CP_perf"])
    _seen = {}
    for t, ct, cp in zip(_raw_tsrs, _raw_ct, _raw_cp):
        _seen[round(t, 6)] = (t, ct, cp)
    _sorted = sorted(_seen.values(), key=lambda x: x[0])
    TSR_SWEEP_PERF = [t   for t, ct, cp in _sorted]
    tsr_CT_perf    = np.array([ct for t, ct, cp in _sorted])
    tsr_CP_perf    = np.array([cp for t, ct, cp in _sorted])
else:
    TSR_SWEEP_PERF = TSR_SWEEP_SPAN
    tsr_CT_perf    = tsr_CT_span
    tsr_CP_perf    = tsr_CP_span

# ── TSR=8 specific ────────────────────────────────────────────────────────────
results_tsr8 = _arr_or_none(B, "results_tsr8")
res_nc       = _arr_or_none(B, "res_nc")
ct_hist_tsr8 = _arr_or_none(B, "ct_hist_tsr8")
F_tsr8       = _arr_or_none(B, "F_tsr8")

# ── Section-6: annuli sensitivity + spacing ───────────────────────────────────
_annuli_N_arr  = _arr_or_none(B, "annuli_N_list")
_annuli_CT_arr = _arr_or_none(B, "annuli_CT_list")
_annuli_CP_arr = _arr_or_none(B, "annuli_CP_list")

_ANNULI_N_KEYS = [4, 8, 16, 32, 64, 160]
annuli_results = {}
for n in _ANNULI_N_KEYS:
    v = _arr_or_none(B, f"annuli_N{n}")
    if v is not None:
        annuli_results[n] = v

if _annuli_N_arr is not None and _annuli_CT_arr is not None:
    annuli_CT_scalar = {int(n): float(ct) for n, ct in zip(_annuli_N_arr, _annuli_CT_arr)}
    annuli_CP_scalar = {int(n): float(cp) for n, cp in zip(_annuli_N_arr, _annuli_CP_arr)}
else:
    annuli_CT_scalar = {}
    annuli_CP_scalar = {}

_spacing_N_arr = _arr_or_none(B, "spacing_N")
N_SPACING = int(_spacing_N_arr[0]) if _spacing_N_arr is not None else 20

spacing_results = {k: v for k, v in
                   [("Constant", _arr_or_none(B, "spacing_constant")),
                    ("Cosine",   _arr_or_none(B, "spacing_cosine"))] if v is not None}

def _first(*arrays):
    for a in arrays:
        if a is not None: return a
    return None

# ── Geometry nodes ────────────────────────────────────────────────────────────
r_base  = _first(_arr_or_none(O, "r_base"),  _arr_or_none(B, "r_base"))
c_base  = _first(_arr_or_none(O, "c_base"),  _arr_or_none(B, "c_base"))
tw_base = _first(_arr_or_none(O, "tw_base"), _arr_or_none(B, "tw_base"))
r_anal  = _arr_or_none(O, "r_anal");  c_anal  = _arr_or_none(O, "c_anal");  tw_anal  = _arr_or_none(O, "tw_anal")
r_cubic = _arr_or_none(O, "r_cubic"); c_cubic = _arr_or_none(O, "c_cubic"); tw_cubic = _arr_or_none(O, "tw_cubic")
r_qrt   = _arr_or_none(O, "r_qrt");   c_qrt   = _arr_or_none(O, "c_qrt");   tw_qrt   = _arr_or_none(O, "tw_qrt")

# ── BEM results ───────────────────────────────────────────────────────────────
res_base  = _first(_arr_or_none(O, "res_base"), _arr_or_none(B, "res_base"))
res_anal  = _arr_or_none(O, "res_anal")
res_cubic = _arr_or_none(O, "res_cubic")
res_qrt   = _arr_or_none(O, "res_qrt")

# ── Scalar performance ────────────────────────────────────────────────────────
def _first_f(*vals):
    for v in vals:
        if v is not None and not (isinstance(v, float) and np.isnan(v)): return v
    return None

CT_base  = _first_f(_getf(O, "CT_base"),  _getf(B, "CT_base"))
CP_base  = _first_f(_getf(O, "CP_base"),  _getf(B, "CP_base"))
CT_anal  = _getf(O, "CT_anal");  CP_anal  = _getf(O, "CP_anal")
CT_cubic = _getf(O, "CT_cubic"); CP_cubic = _getf(O, "CP_cubic")
CT_qrt   = _getf(O, "CT_qrt");   CP_qrt   = _getf(O, "CP_qrt")
CP_ad    = _getf(O, "CP_ad")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"  Span TSR sweep : {TSR_SWEEP_SPAN}")
print(f"  Perf TSR sweep : {TSR_SWEEP_PERF}")
print(f"  Baseline       : CT={CT_base:.4f}  CP={CP_base:.4f}" if CT_base else "  Baseline       : not in file")
print(f"  Analytical     : CT={CT_anal:.4f}  CP={CP_anal:.4f}" if CT_anal else "  Analytical     : not in file")
print(f"  Cubic poly     : CT={CT_cubic:.4f}  CP={CP_cubic:.4f}" if CT_cubic else "  Cubic poly     : not in file")
print(f"  Quartic poly   : CT={CT_qrt:.4f}  CP={CP_qrt:.4f}" if CT_qrt else "  Quartic poly   : not in file")

# =============================================================================
# 2.  COLOR SCHEME  (seaborn colorblind palette)
# =============================================================================

def _tsr_color(idx, n):
    """Colour for the idx-th TSR line, drawn from colorblind palette."""
    return _CB_HEX[idx % len(_CB_HEX)]

_DESIGN_COLOR_MAP = {
    "Baseline":     _CB_HEX[0],   # blue
    "Analytical":   _CB_HEX[1],   # orange
    "Cubic poly":   _CB_HEX[2],   # green
    "Quartic poly": _CB_HEX[3],   # red
}

def _design_color(label, n=None):
    return _DESIGN_COLOR_MAP.get(label, _CB_HEX[4])

# Annuli sensitivity (6 levels drawn from palette)
_ANNULI_COLS  = {n: _CB_HEX[i] for i, n in enumerate(_ANNULI_N_KEYS)}
# Spacing comparison
_SPACING_COLS = {"Constant": _CB_HEX[0], "Cosine": _CB_HEX[1]}

# =============================================================================
# 3.  HELPER FUNCTIONS
# =============================================================================

def find_optimal_alpha():
    alphas = np.linspace(polar_alpha[0], polar_alpha[-1], 2000)
    cl_v   = np.interp(alphas, polar_alpha, polar_cl)
    cd_v   = np.interp(alphas, polar_alpha, polar_cd)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(cd_v > 1e-6, cl_v / cd_v, 0.0)
    idx = int(np.argmax(ratio))
    return float(alphas[idx]), float(cl_v[idx]), float(cd_v[idx])


def _skip(name, reason):
    print(f"  [SKIP] {name} — {reason}")
    plt.close("all")


def _lighten(hex_color, amount=0.45):
    """Return a lighter (white-blended) version of hex_color."""
    r = int(hex_color[1:3], 16) / 255
    g = int(hex_color[3:5], 16) / 255
    b = int(hex_color[5:7], 16) / 255
    return (f"#{int((r+(1-r)*amount)*255):02x}"
            f"{int((g+(1-g)*amount)*255):02x}"
            f"{int((b+(1-b)*amount)*255):02x}")


def _build_designs(r_R_dense):
    designs = []
    if r_base is not None:
        designs.append(("Baseline",
                         np.interp(r_R_dense, r_base/Radius, c_base),
                         np.interp(r_R_dense, r_base/Radius, tw_base),
                         res_base, CT_base, CP_base))
    if r_anal is not None and res_anal is not None:
        designs.append(("Analytical",
                         np.interp(r_R_dense, r_anal/Radius, c_anal),
                         np.interp(r_R_dense, r_anal/Radius, tw_anal),
                         res_anal, CT_anal, CP_anal))
    else:
        print("  [INFO] analytical design absent — skipping that curve")
    if r_cubic is not None and res_cubic is not None:
        designs.append(("Cubic poly",
                         np.interp(r_R_dense, r_cubic/Radius, c_cubic),
                         np.interp(r_R_dense, r_cubic/Radius, tw_cubic),
                         res_cubic, CT_cubic, CP_cubic))
    else:
        print("  [INFO] cubic poly absent — skipping that curve")
    if r_qrt is not None and res_qrt is not None:
        designs.append(("Quartic poly",
                         np.interp(r_R_dense, r_qrt/Radius, c_qrt),
                         np.interp(r_R_dense, r_qrt/Radius, tw_qrt),
                         res_qrt, CT_qrt, CP_qrt))
    else:
        print("  [INFO] quartic poly absent — skipping that curve")
    return designs


def _tangential_induction_comparison(designs):
    fig, ax = plt.subplots(figsize=(9, 5))
    for lbl, _, _, res, *_ in designs:
        ax.plot(res[:,2], res[:,1], color=_design_color(lbl), lw=2, label=lbl)
    ax.set_xlabel("r/R"); ax.set_ylabel(r"$a'$ [-]")
    ax.legend(); ax.grid(True)
    fig.tight_layout()
    return fig


def _chord_cl_comparison(r_R_dense, designs):
    targets = {"Analytical", "Quartic poly"}
    subset  = [(lbl, c_d, tw_d, res, CT, CP)
               for (lbl, c_d, tw_d, res, CT, CP) in designs if lbl in targets]
    if not subset:
        print("  [INFO] chord/Cl comparison skipped — neither analytical nor quartic available")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: primary = Chord, secondary = Cl
    ax_l  = axes[0]; ax_lr = ax_l.twinx()
    for lbl, c_d, _, res, *_ in subset:
        col = _design_color(lbl)
        ax_l.plot(r_R_dense,  c_d,      color=col, lw=2,       label=f"{lbl} — chord")
        ax_lr.plot(res[:,2], res[:,8],   color=col, lw=2, ls="--", label=rf"{lbl} — $C_l$")
    ax_l.axhline(CHORD_MIN,  color="grey", ls=":",  lw=1,  label=f"Min chord {CHORD_MIN} m")
    ax_l.axhline(CHORD_ROOT, color="k",   ls="--", lw=0.8, label=f"Root chord {CHORD_ROOT} m")
    ax_l.set_xlabel("r/R"); ax_l.set_ylabel("Chord [m]"); ax_lr.set_ylabel(r"$C_l$ [-]")
    ax_l.set_zorder(ax_lr.get_zorder() + 1); ax_l.patch.set_visible(False)
    h1, l1 = ax_l.get_legend_handles_labels(); h2, l2 = ax_lr.get_legend_handles_labels()
    ax_l.legend(h1+h2, l1+l2, fontsize=8); ax_l.grid(True)

    # Right: primary = Cl, secondary = Chord
    ax_r  = axes[1]; ax_rr = ax_r.twinx()
    for lbl, c_d, _, res, *_ in subset:
        col = _design_color(lbl)
        ax_r.plot(res[:,2],  res[:,8],  color=col, lw=2,       label=rf"{lbl} — $C_l$")
        ax_rr.plot(r_R_dense, c_d,     color=col, lw=2, ls="--", label=f"{lbl} — chord")
    ax_r.set_xlabel("r/R"); ax_r.set_ylabel(r"$C_l$ [-]"); ax_rr.set_ylabel("Chord [m]")
    ax_r.set_zorder(ax_rr.get_zorder() + 1); ax_r.patch.set_visible(False)
    h1, l1 = ax_r.get_legend_handles_labels(); h2, l2 = ax_rr.get_legend_handles_labels()
    ax_r.legend(h1+h2, l1+l2, fontsize=8); ax_r.grid(True)

    fig.tight_layout()
    return fig

# =============================================================================
# 4.  SAVE / SHOW HELPER
# =============================================================================

save_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "BEM_plotting_plots")
os.makedirs(save_folder, exist_ok=True)

def save_fig(name):
    """Save figure as both PDF and PNG."""
    base = os.path.splitext(name)[0]
    plt.savefig(os.path.join(save_folder, base + ".pdf"), dpi=300, bbox_inches="tight")
    print(f"  Saved: {base}.pdf")
    if SHOW_PLOTS: plt.show()
    else:          plt.close()

norm_val = 0.5 * U0**2 * Radius
n_span   = len(TSR_SWEEP_SPAN)

# =============================================================================
# 5.  SECTION 4.1  (alpha and inflow angle)
# =============================================================================

if PLOT_4_1 and sweep_data_span:
    for qty_col, ylabel, fname in [
            (6, r"$\alpha$ [deg]", "4_1a_angle_of_attack_vs_rR"),
            (7, r"$\phi$ [deg]",   "4_1b_inflow_angle_vs_rR")]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for k, TSR in enumerate(TSR_SWEEP_SPAN):
            res = sweep_data_span[TSR]
            ax.plot(res[:,2], res[:,qty_col], color=_tsr_color(k, n_span), lw=2,
                    label=rf"$\lambda={TSR}$")
        ax.set_xlabel("r/R"); ax.set_ylabel(ylabel)
        ax.legend(); ax.grid(True)
        fig.tight_layout(); save_fig(fname)
elif PLOT_4_1:
    _skip("PLOT_4_1", "span sweep data missing from npz")

# =============================================================================
# 6.  SECTION 4.2  (induction factors)
# =============================================================================

if PLOT_4_2 and sweep_data_span:
    for qty_col, ylabel, fname in [
            (0, r"$a$ [-]",  "4_2a_axial_induction_vs_rR"),
            (1, r"$a'$ [-]", "4_2b_tangential_induction_vs_rR")]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for k, TSR in enumerate(TSR_SWEEP_SPAN):
            res = sweep_data_span[TSR]
            ax.plot(res[:,2], res[:,qty_col], color=_tsr_color(k, n_span), lw=2,
                    label=rf"$\lambda={TSR}$")
        ax.set_xlabel("r/R"); ax.set_ylabel(ylabel)
        ax.legend(); ax.grid(True)
        fig.tight_layout(); save_fig(fname)
elif PLOT_4_2:
    _skip("PLOT_4_2", "span sweep data missing from npz")

# =============================================================================
# 7.  SECTION 4.3  (loading)
# =============================================================================

if PLOT_4_3 and sweep_data_span:
    for qty_col, ylabel, fname in [
            (3, r"$C_n = F_n\,/\,(½\rho U_\infty^2 R)$", "4_3a_normal_loading_Cn_vs_rR"),
            (4, r"$C_t = F_t\,/\,(½\rho U_\infty^2 R)$", "4_3b_azimuthal_loading_Ct_vs_rR")]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for k, TSR in enumerate(TSR_SWEEP_SPAN):
            res = sweep_data_span[TSR]
            ax.plot(res[:,2], res[:,qty_col]/norm_val, color=_tsr_color(k, n_span), lw=2,
                    label=rf"$\lambda={TSR}$")
        ax.set_xlabel("r/R"); ax.set_ylabel(ylabel)
        ax.legend(); ax.grid(True)
        fig.tight_layout(); save_fig(fname)
elif PLOT_4_3:
    _skip("PLOT_4_3", "span sweep data missing from npz")

# =============================================================================
# 8.  SECTION 4.4  (CT, CQ, CP vs TSR)
# =============================================================================

if PLOT_4_4 and len(TSR_SWEEP_PERF) > 0:
    CQ_perf = tsr_CP_perf / np.array(TSR_SWEEP_PERF, dtype=float)
    for vals, ylabel, fname, cb_idx in [
            (tsr_CT_perf, r"$C_T$ [-]", "4_4a_thrust_coefficient_CT_vs_TSR",  0),
            (CQ_perf,     r"$C_Q$ [-]", "4_4b_torque_coefficient_CQ_vs_TSR",  3),
            (tsr_CP_perf, r"$C_P$ [-]", "4_4c_power_coefficient_CP_vs_TSR",   2)]:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(TSR_SWEEP_PERF, vals, "o-", color=_CB_HEX[cb_idx], lw=2)
        ax.set_xlabel(r"Tip-speed ratio $\lambda$ [-]"); ax.set_ylabel(ylabel)
        ax.grid(True)
        fig.tight_layout(); save_fig(fname)
elif PLOT_4_4:
    _skip("PLOT_4_4", "performance sweep data missing from npz")

# =============================================================================
# 9.  SECTION 5  (tip correction)
# =============================================================================

if PLOT_5 and results_tsr8 is not None and res_nc is not None:
    r_R8 = results_tsr8[:,2]
    for qty_col, ylabel, fname in [
            (0, r"$a$ [-]",   "5a_axial_induction_tip_correction_comparison"),
            (3, r"$C_n$ [-]", "5b_normal_loading_tip_correction_comparison")]:
        fig, ax = plt.subplots(figsize=(8, 5))
        yc  = results_tsr8[:,qty_col] if qty_col == 0 else results_tsr8[:,qty_col]/norm_val
        ync = res_nc[:,0]             if qty_col == 0 else res_nc[:,3]/norm_val
        ax.plot(r_R8,        yc,  color=_CB_HEX[0], lw=2, label="With Prandtl correction")
        ax.plot(res_nc[:,2], ync, color=_CB_HEX[3], lw=2, linestyle="--", label="No correction (F=1)")
        ax.set_xlabel("r/R"); ax.set_ylabel(ylabel)
        ax.legend(); ax.grid(True)
        fig.tight_layout(); save_fig(fname)
elif PLOT_5:
    _skip("PLOT_5", "res_nc or results_tsr8 missing (run with RUN_NO_CORRECTION=True)")

# =============================================================================
# 10.  SECTION 6  (annuli sensitivity + spacing study)
# =============================================================================

if PLOT_6 and results_tsr8 is not None and annuli_results and spacing_results:

    def _annuli_marker(N):
        return ("o", 4) if N <= 16 else (None, None)

    def _plot_annuli_quantity(col_idx, ylabel, fname, xlim=None):
        fig, ax = plt.subplots(figsize=(8, 5))
        for N, res_N in sorted(annuli_results.items()):
            mk, ms = _annuli_marker(N)
            ax.plot(res_N[:,2],
                    res_N[:,col_idx]/norm_val if col_idx in (3, 4) else res_N[:,col_idx],
                    color=_ANNULI_COLS.get(N, _CB_HEX[4]), lw=2,
                    marker=mk, markersize=ms, label=f"N={N}")
        ax.set_xlabel("r/R"); ax.set_ylabel(ylabel)
        if xlim: ax.set_xlim(*xlim)
        ax.legend(); ax.grid(True)
        fig.tight_layout(); save_fig(fname)

    def _plot_spacing_quantity(col_idx, ylabel, fname, xlim=None):
        fig, ax = plt.subplots(figsize=(8, 5))
        for lbl, res_sp in spacing_results.items():
            ax.plot(res_sp[:,2],
                    res_sp[:,col_idx]/norm_val if col_idx in (3, 4) else res_sp[:,col_idx],
                    "-o", markersize=5, color=_SPACING_COLS.get(lbl, _CB_HEX[4]),
                    lw=2, label=lbl)
        ax.set_xlabel("r/R"); ax.set_ylabel(ylabel)
        if xlim: ax.set_xlim(*xlim)
        ax.legend(); ax.grid(True)
        fig.tight_layout(); save_fig(fname)

    # Annuli sensitivity
    _plot_annuli_quantity(3, r"$C_n = F_n\,/\,(½\rho U_\infty^2 R)$",
                          "6a1_Cn_vs_rR_annuli_sensitivity")
    _plot_annuli_quantity(4, r"$C_t = F_t\,/\,(½\rho U_\infty^2 R)$",
                          "6a2_Ct_vs_rR_annuli_sensitivity")
    _plot_annuli_quantity(0, r"$a$ [-]",
                          "6a3_axial_induction_vs_rR_annuli_sensitivity")
    _plot_annuli_quantity(6, r"$\alpha$ [deg]",
                          "6a4_alpha_vs_rR_annuli_sensitivity")

    # CT vs N convergence
    if annuli_CT_scalar:
        _N_list = sorted(annuli_CT_scalar.keys())
        ref_CT  = annuli_CT_scalar.get(160)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(_N_list, [annuli_CT_scalar[n] for n in _N_list], "o-", color=_CB_HEX[3], lw=2)
        if ref_CT:
            ax.axhline(ref_CT, color="k", ls="--", lw=0.8,
                       label=f"N=160 reference ({ref_CT:.4f})")
            ax.legend()
        ax.set_xlabel("Number of annuli N"); ax.set_ylabel(r"$C_T$ [-]")
        ax.grid(True); fig.tight_layout(); save_fig("6a5_CT_vs_N_annuli_convergence")

    # CP vs N convergence
    if annuli_CP_scalar:
        _N_list = sorted(annuli_CP_scalar.keys())
        ref_CP  = annuli_CP_scalar.get(160)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(_N_list, [annuli_CP_scalar[n] for n in _N_list], "o-", color=_CB_HEX[2], lw=2)
        if ref_CP:
            ax.axhline(ref_CP, color="k", ls="--", lw=0.8,
                       label=f"N=160 reference ({ref_CP:.4f})")
            ax.legend()
        ax.set_xlabel("Number of annuli N"); ax.set_ylabel(r"$C_P$ [-]")
        ax.grid(True); fig.tight_layout(); save_fig("6a6_CP_vs_N_annuli_convergence")

    # Cn tip zoom
    _plot_annuli_quantity(3, r"$C_n = F_n\,/\,(½\rho U_\infty^2 R)$",
                          "6a7_Cn_tip_zoom_annuli_sensitivity", xlim=(0.8, 1.0))

    # Spacing comparison
    _plot_spacing_quantity(3, r"$C_n = F_n\,/\,(½\rho U_\infty^2 R)$",
                           "6b1_Cn_vs_rR_spacing_comparison")
    _plot_spacing_quantity(3, r"$C_n = F_n\,/\,(½\rho U_\infty^2 R)$",
                           "6b2_Cn_tip_zoom_spacing_comparison", xlim=(0.8, 1.0))
    _plot_spacing_quantity(4, r"$C_t = F_t\,/\,(½\rho U_\infty^2 R)$",
                           "6b3_Ct_vs_rR_spacing_comparison")
    _plot_spacing_quantity(0, r"$a$ [-]",
                           "6b4_axial_induction_vs_rR_spacing_comparison")
    _plot_spacing_quantity(6, r"$\alpha$ [deg]",
                           "6b5_alpha_vs_rR_spacing_comparison")

    # Iteration convergence
    if ct_hist_tsr8 is not None:
        n_show = min(60, len(ct_hist_tsr8))
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(range(1, len(ct_hist_tsr8)+1), ct_hist_tsr8, color=_CB_HEX[0], lw=2)
        ax.set_xlim(1, n_show); ax.set_xlabel("Iteration"); ax.set_ylabel(r"$C_T$ [-]")
        ax.grid(True); fig.tight_layout(); save_fig("6c_CT_convergence_history")

        resid = np.abs(np.diff(ct_hist_tsr8))
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogy(range(2, len(ct_hist_tsr8)+1), resid, color=_CB_HEX[3], lw=2,
                    label=r"$|C_{T,i}-C_{T,i-1}|$")
        ax.axhline(1e-5, color="k", ls="--", lw=0.8, label="Tolerance = 1e-5")
        ax.set_xlim(1, n_show); ax.set_xlabel("Iteration"); ax.set_ylabel(r"$|\Delta C_T|$")
        ax.legend(); ax.grid(True, which="both")
        fig.tight_layout(); save_fig("6d_CT_convergence_residuals_log_scale")

elif PLOT_6:
    _skip("PLOT_6", "annuli/spacing data missing (run with RUN_TSR_SWEEP_SPAN=True)")

# =============================================================================
# 11.  SECTION 7  (stagnation pressure)
# =============================================================================

def _plot_stagnation_pressure_for_design(res_arr, design_name, file_tag):
    if res_arr is None:
        print(f"  [INFO] stagnation-pressure plot skipped for {design_name} — result array missing")
        return

    r_R = res_arr[:, 2]
    a_R = res_arr[:, 0]
    q_inf   = 0.5 * rho * U0**2
    P0_12   = np.ones(len(r_R))
    P0_34   = (1.0 - 2.0 * a_R) ** 2
    P0_up   = q_inf * P0_12
    P0_down = q_inf * P0_34
    eps     = 0.003 * q_inf

    # Four stations
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(r_R, P0_up,       color=_CB_HEX[0], lw=2.5,
            label=rf"$P_0^{{\infty,\uparrow}}$ ({design_name}, $\infty$ upwind)")
    ax.plot(r_R, P0_down,     color=_CB_HEX[3], lw=2.5,
            label=rf"$P_0^{{\infty,\downarrow}}$ ({design_name}, $\infty$ downwind)")
    ax.plot(r_R, P0_up+eps,   color=_CB_HEX[7], lw=1.8, ls="--", alpha=0.95,
            label=rf"$P_0^{{+}}$ ({design_name}, rotor upwind)")
    ax.plot(r_R, P0_down+eps, color=_CB_HEX[2], lw=1.8, ls="--", alpha=0.95,
            label=rf"$P_0^{{-}}$ ({design_name}, rotor downwind)")
    ax.set_xlabel("r/R"); ax.set_ylabel(r"$P_0$ [Pa]")
    ax.grid(True); ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(f"7_{file_tag}_stagnation_pressure_four_stations")

    # Normalised drop
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(r_R, P0_12, color=_CB_HEX[0], lw=2.5,
            label=r"Upstream $P_0/q_\infty = 1$")
    ax.plot(r_R, P0_34, color=_CB_HEX[3], lw=2.5,
            label=r"Downstream $P_0/q_\infty = (1-2a)^2$")
    ax.fill_between(r_R, P0_34, P0_12,
                    color=_CB_HEX[1], alpha=0.25, label=r"$\Delta P_0$")
    ax.set_xlabel("r/R"); ax.set_ylabel(r"$P_0/q_\infty$ [-]")
    ax.grid(True); ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(f"7_{file_tag}_stagnation_pressure_drop")


if PLOT_7 and results_tsr8 is not None:
    _plot_stagnation_pressure_for_design(results_tsr8, "Baseline", "baseline")
    _plot_stagnation_pressure_for_design(res_anal,     "Analytical",   "opt_analytical")
    _plot_stagnation_pressure_for_design(res_cubic,    "Cubic poly",   "opt_cubic_poly")
    _plot_stagnation_pressure_for_design(res_qrt,      "Quartic poly", "opt_quartic_poly")
elif PLOT_7:
    _skip("PLOT_7", "results_tsr8 missing (run with RUN_TSR_SWEEP_SPAN=True)")

# =============================================================================
# 12.  SECTION 8  (design comparison)
# =============================================================================

if PLOT_8 and res_base is not None:
    r_R_dense = np.linspace(RootLocation_R, TipLocation_R, 400)
    designs   = _build_designs(r_R_dense)

    # 8a — chord
    fig, ax = plt.subplots(figsize=(9, 5))
    for lbl, c_d, *_ in designs:
        ax.plot(r_R_dense, c_d, color=_design_color(lbl), lw=2, label=lbl)
    ax.axhline(CHORD_MIN,  color="grey", ls=":",  lw=1,  label=f"Min chord  {CHORD_MIN} m")
    ax.axhline(CHORD_ROOT, color="k",   ls="--", lw=0.8, label=f"Root chord  {CHORD_ROOT} m")
    ax.set_xlabel("r/R"); ax.set_ylabel("Chord [m]")
    ax.legend(); ax.grid(True)
    fig.tight_layout(); save_fig("8a_chord_distribution_design_comparison")

    # 8b — twist
    fig, ax = plt.subplots(figsize=(9, 5))
    for lbl, _, tw_d, *_ in designs:
        ax.plot(r_R_dense, tw_d, color=_design_color(lbl), lw=2, label=lbl)
    ax.set_xlabel("r/R"); ax.set_ylabel("Twist [deg]")
    ax.legend(); ax.grid(True)
    fig.tight_layout(); save_fig("8b_twist_distribution_design_comparison")

    # 8ab — chord and twist side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for lbl, c_d, tw_d, *_ in designs:
        col = _design_color(lbl)
        axes[0].plot(r_R_dense, c_d,  color=col, lw=2, label=lbl)
        axes[1].plot(r_R_dense, tw_d, color=col, lw=2, label=lbl)
    axes[0].axhline(CHORD_MIN,  color="grey", ls=":",  lw=1,  label=f"Min  {CHORD_MIN} m")
    axes[0].axhline(CHORD_ROOT, color="k",   ls="--", lw=0.8, label=f"Root  {CHORD_ROOT} m")
    axes[0].set_xlabel("r/R"); axes[0].set_ylabel("Chord [m]")
    axes[0].legend(); axes[0].grid(True)
    axes[1].set_xlabel("r/R"); axes[1].set_ylabel("Twist [deg]")
    axes[1].legend(); axes[1].grid(True)
    fig.tight_layout(); save_fig("8ab_chord_and_twist_design_comparison")

    # 8c — axial induction
    fig, ax = plt.subplots(figsize=(9, 5))
    for lbl, _, _, res, *_ in designs:
        ax.plot(res[:,2], res[:,0], color=_design_color(lbl), lw=2, label=lbl)
    ax.axhline(1/3, color="grey", ls=":", lw=0.8, label="a = 1/3  (Betz)")
    ax.set_xlabel("r/R"); ax.set_ylabel(r"$a$ [-]")
    ax.legend(); ax.grid(True)
    fig.tight_layout(); save_fig("8c_axial_induction_design_comparison")

    # 8c2 — tangential induction
    fig = _tangential_induction_comparison(designs)
    save_fig("8c2_tangential_induction_design_comparison")

    # 8d — normal loading
    fig, ax = plt.subplots(figsize=(9, 5))
    for lbl, _, _, res, *_ in designs:
        ax.plot(res[:,2], res[:,3]/norm_val, color=_design_color(lbl), lw=2, label=lbl)
    ax.set_xlabel("r/R"); ax.set_ylabel(r"$C_n = F_n\,/\,(½\rho U_\infty^2 R)$")
    ax.legend(); ax.grid(True)
    fig.tight_layout(); save_fig("8d_normal_loading_design_comparison")

    # 8e — angle of attack
    fig, ax = plt.subplots(figsize=(9, 5))
    for lbl, _, _, res, *_ in designs:
        ax.plot(res[:,2], res[:,6], color=_design_color(lbl), lw=2, label=lbl)
    ax.set_xlabel("r/R"); ax.set_ylabel(r"$\alpha$ [deg]")
    ax.legend(); ax.grid(True)
    fig.tight_layout(); save_fig("8e_angle_of_attack_design_comparison")

    # 8f — performance bar chart
    a_ad_  = 0.5 * (1.0 - np.sqrt(1.0 - CT_TARGET))
    cp_ad_ = CP_ad if CP_ad is not None else 4.0*a_ad_*(1.0-a_ad_)**2
    labels_b = [d[0] for d in designs] + ["Actuator disk"]
    cp_vals  = [d[5] for d in designs] + [cp_ad_]
    ct_vals  = [d[4] for d in designs] + [CT_TARGET]
    eff_vals = [v / cp_ad_ for v in cp_vals]
    bar_cols = [_design_color(d[0]) for d in designs] + [_CB_HEX[7]]
    x = np.arange(len(labels_b)); w = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x-w, cp_vals,  w, label=r"$C_P$",
                color=[c+"cc" for c in bar_cols])
    b2 = ax.bar(x,   ct_vals,  w, label=r"$C_T$",
                color=bar_cols, alpha=0.6)
    b3 = ax.bar(x+w, eff_vals, w, label=r"$C_P/C_{P,\mathrm{AD}}$",
                color=bar_cols, hatch="//", alpha=0.85)
    for bars in [b1, b2, b3]:
        for bar in bars: bar.set_edgecolor("white"); bar.set_linewidth(0.5)
    ax.set_xticks(x); ax.set_xticklabels(labels_b, rotation=15, ha="right")
    ax.set_ylabel("Coefficient [-]"); ax.legend()
    ax.bar_label(b1, fmt="%.4f", padding=3, fontsize=7.5)
    ax.bar_label(b2, fmt="%.4f", padding=3, fontsize=7.5)
    ax.bar_label(b3, fmt="%.3f", padding=3, fontsize=7.5)
    ax.grid(True, axis="y")
    fig.tight_layout(); save_fig("8f_performance_comparison_all_designs")

elif PLOT_8:
    _skip("PLOT_8", "res_base missing from npz")

# =============================================================================
# 13.  SECTION 9  (Cl, chord and circulation)
# =============================================================================

def _make_9a_axes(ax, r_mid, cl, chord, col, lbl):
    col_chord = _lighten(col, 0.45)
    ax.plot(r_mid, cl, color=col, lw=2, ls="-", label=rf"{lbl} — $C_l$")
    ax.set_xlabel("r/R"); ax.set_ylabel(r"$C_l$ [-]"); ax.grid(True)
    ax2 = ax.twinx()
    ax2.plot(r_mid, chord, color=col_chord, lw=2, ls="--", label=f"{lbl} — chord")
    ax2.set_ylabel("Chord [m]")
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, fontsize=8)

if PLOT_9:
    _opt_data = []
    for lbl, r_arr, c_arr, res_arr in [
            ("Baseline",     r_base,  c_base,  res_base),
            ("Analytical",   r_anal,  c_anal,  res_anal),
            ("Cubic poly",   r_cubic, c_cubic, res_cubic),
            ("Quartic poly", r_qrt,   c_qrt,   res_qrt)]:
        if res_arr is not None and r_arr is not None and res_arr.shape[1] >= 9:
            r_mid = res_arr[:,2]; cl = res_arr[:,8]
            chord = np.interp(r_mid, r_arr/Radius, c_arr)
            _opt_data.append((lbl, r_mid, cl, chord))

    if _opt_data:
        # 9a combined
        fig, ax = plt.subplots(figsize=(9, 5))
        ax2_comb = ax.twinx()
        for lbl, r_mid, cl, chord in _opt_data:
            col = _design_color(lbl); col_chord = _lighten(col, 0.45)
            ax.plot(r_mid, cl,    color=col,       lw=2, ls="-",  label=rf"{lbl} — $C_l$")
            ax2_comb.plot(r_mid, chord, color=col_chord, lw=2, ls="--", label=f"{lbl} — chord")
        ax.set_xlabel("r/R"); ax.set_ylabel(r"$C_l$ [-]"); ax.grid(True)
        ax2_comb.set_ylabel("Chord [m]")
        ax.set_zorder(ax2_comb.get_zorder() + 1); ax.patch.set_visible(False)
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2_comb.get_legend_handles_labels()
        ax.legend(h1+h2, l1+l2, fontsize=8)
        fig.tight_layout(); save_fig("9a_combined_Cl_and_chord_optimised_designs")

        # 9a individual
        for lbl, r_mid, cl, chord in _opt_data:
            fig, ax = plt.subplots(figsize=(9, 5))
            _make_9a_axes(ax, r_mid, cl, chord, _design_color(lbl), lbl)
            fig.tight_layout()
            save_fig("9a_" + lbl.lower().replace(" ", "_") + "_Cl_and_chord")

        # 9b circulation proxy
        fig, ax = plt.subplots(figsize=(9, 5))
        for lbl, r_mid, cl, chord in _opt_data:
            ax.plot(r_mid, cl*chord, color=_design_color(lbl), lw=2, label=lbl)
        ax.set_xlabel("r/R"); ax.set_ylabel(r"$C_l \cdot c$  [m]"); ax.grid(True)
        ax.legend(); fig.tight_layout()
        save_fig("9b_circulation_proxy_Cl_times_chord_optimised_designs")
    else:
        _skip("PLOT_9", "no optimised design results with Cl data available in npz")

# 9c — chord + Cl side-by-side, analytical vs quartic
if PLOT_9_CMP:
    if not (PLOT_8 and res_base is not None):
        r_R_dense = np.linspace(RootLocation_R, TipLocation_R, 400)
        designs   = _build_designs(r_R_dense)
    if len(designs) > 0:
        fig = _chord_cl_comparison(r_R_dense, designs)
        if fig is not None:
            save_fig("9c_chord_and_Cl_analytical_vs_quartic_comparison")
    else:
        _skip("PLOT_9_CMP", "no design data available in npz")

# =============================================================================
# 14.  SECTION 10  (Cl/Cd polar and glide ratio)
# =============================================================================

if PLOT_10:
    _pol_all = []; _pol_opt = []
    for lbl, res_arr in [("Baseline",     res_base),
                          ("Analytical",   res_anal),
                          ("Cubic poly",   res_cubic),
                          ("Quartic poly", res_qrt)]:
        if res_arr is not None and res_arr.shape[1] >= 10:
            _pol_all.append((lbl, res_arr))
            if lbl != "Baseline":
                _pol_opt.append((lbl, res_arr))

    if _pol_all:
        alpha_opt, cl_opt, cd_opt = find_optimal_alpha()
        alphas_d = np.linspace(polar_alpha[0], polar_alpha[-1], 500)
        cl_d  = np.interp(alphas_d, polar_alpha, polar_cl)
        cd_d  = np.interp(alphas_d, polar_alpha, polar_cd)
        ld_d  = cl_d / np.maximum(cd_d, 1e-8)
        _norm = mcolors.Normalize(vmin=RootLocation_R, vmax=TipLocation_R)
        _cmap = cm.viridis
        _MARKERS = {"Baseline":"o", "Analytical":"s",
                    "Cubic poly":"^", "Quartic poly":"D"}

        def _draw_polar(ax, designs):
            ax.plot(polar_cd, polar_cl, "k-", lw=1.5,
                    label="DU95W180 polar", zorder=1)
            for lbl, res_arr in designs:
                sc = ax.scatter(res_arr[:,9], res_arr[:,8],
                                c=res_arr[:,2], cmap=_cmap, norm=_norm,
                                s=30, marker=_MARKERS.get(lbl, "o"), zorder=5, label=lbl)
            plt.colorbar(sc, ax=ax, label="r/R")
            ax.set_xlabel(r"$C_d$ [-]"); ax.set_ylabel(r"$C_l$ [-]")
            ax.legend(fontsize=8); ax.grid(True)

        def _draw_glide(ax, designs):
            ax.plot(alphas_d, ld_d, "k-", lw=1.5,
                    label=r"$C_l/C_d$  DU95W180", zorder=1)
            ax.axvline(alpha_opt, color="k", ls="--", lw=0.8,
                       label=rf"$\alpha_{{opt}}={alpha_opt:.1f}°$,"
                             rf"  $(C_l/C_d)_{{max}}={cl_opt/cd_opt:.0f}$")
            for lbl, res_arr in designs:
                ld_ops = (np.interp(res_arr[:,6], polar_alpha, polar_cl)
                          / np.maximum(np.interp(res_arr[:,6], polar_alpha, polar_cd), 1e-8))
                ax.scatter(res_arr[:,6], ld_ops,
                           c=res_arr[:,2], cmap=_cmap, norm=_norm,
                           s=30, marker=_MARKERS.get(lbl, "o"), zorder=5, label=lbl)
            plt.colorbar(ax.collections[-1], ax=ax, label="r/R")
            ax.set_xlabel(r"$\alpha$ [deg]"); ax.set_ylabel(r"$C_l/C_d$ [-]")
            ax.legend(fontsize=8); ax.grid(True)

        # 10a combined
        fig, ax = plt.subplots(figsize=(9, 5))
        _draw_polar(ax, _pol_all); fig.tight_layout()
        save_fig("10a_combined_Cl_Cd_polar_all_designs")

        # 10a individual
        for lbl, res_arr in _pol_opt:
            fig, ax = plt.subplots(figsize=(9, 5))
            _draw_polar(ax, [(lbl, res_arr)]); fig.tight_layout()
            save_fig("10a_" + lbl.lower().replace(" ", "_") + "_Cl_Cd_polar")

        # 10b combined
        fig, ax = plt.subplots(figsize=(9, 5))
        _draw_glide(ax, _pol_all); fig.tight_layout()
        save_fig("10b_combined_glide_ratio_vs_alpha_all_designs")

        # 10b individual
        for lbl, res_arr in _pol_opt:
            fig, ax = plt.subplots(figsize=(9, 5))
            _draw_glide(ax, [(lbl, res_arr)]); fig.tight_layout()
            save_fig("10b_" + lbl.lower().replace(" ", "_") + "_glide_ratio_vs_alpha")

    else:
        _skip("PLOT_10", "no design BEM results with Cd data available in npz")

# =============================================================================
print("\n" + "="*60)
print(f"ALL PLOTS SAVED TO: {save_folder}")
print("="*60)
print("\nFile list:")
for f in sorted(os.listdir(save_folder)):
    if f.endswith((".pdf")):
        print(f"  {f}")



