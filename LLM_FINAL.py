"""
LLM_FINAL.py  —  AE4135 Rotor/Wake Aerodynamics, Assignment 2
Frozen Vortex Wake / Lifting Line model for the DU95W180 wind turbine rotor.

Authors: Douwe de Jong (5313899), Martijn van Leeuwen (5614422)
================================================================
Self-contained script producing all required plots and saving
results to ll_results.npz for use with PLOTTING_LLM_FINAL.py.

Run:  python LLM_FINAL.py
"""

import os, sys, time
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# 0.  CONFIGURATION
# =============================================================================

# ── Rotor geometry ─────────────────────────────
Radius         = 50.0
NBlades        = 3
U0             = 10.0
rho            = 1.0          # air density [kg/m³]
RootLocation_R = 0.2
TipLocation_R  = 1.0
Pitch          = -2.0         # blade pitch [deg]

# ── Baseline wake / discretisation parameters ────────────────────────────────
N_PANELS     = 20             # number of spanwise panels per blade
N_WAKE       = 5              # number of full wake rotations
DPSI_DEG     = 10.0           # azimuthal step size [deg]
A_WAKE       = 0.25           # frozen-wake axial induction (convection speed factor)
DISTRIBUTION = "cosine"       # spanwise panel distribution: "cosine" or "constant"
R_CORE       = 1e-3           # vortex core radius (Rankine) [m]

# ── Wake convection speed mode ────────────────────────────────────────────────
# False -> use A_WAKE directly (fixed frozen wake)
# True  -> iterate outer loop until a_w converges to mean axial induction
USE_ITERATED_AW = False
AW_ITER_TOL     = 1e-3        # outer loop convergence tolerance on a_w
AW_ITER_MAX     = 20          # maximum outer iterations
AW_ITER_INIT    = 0.25        # initial guess for a_w when iterating

# ── TSR sweeps ────────────────────────────────────────────────────────────────
TSR_SWEEP_SPAN = [6, 8, 10]   # used for all spanwise plots

# ── Performance sweep: λ=4..12 step 0.5 — ONLY used for CT/CP-λ plots ────────
TSR_SWEEP_PERF = list(np.arange(4.0, 12.5, 0.5))   # [4.0, 4.5, …, 12.0]

# ── Sensitivity sweep parameters (all run at TSR=8) ──────────────────────────
SENS_TSR          = 8
SENS_N_LIST       = [5, 8, 10, 13, 15, 18, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 125, 150]
SENS_AW_LIST      = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
SENS_DPSI_LIST    = [1.0, 2.5, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0,
                     60.0, 70.0, 80.0, 90.0]
SENS_NWAKE_LIST   = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

# ── Display / save ─────────────────────────────────────────────────────────────
SHOW_PLOTS = False

# ── Computations (toggle off to skip expensive runs) ─────────────────────────
RUN_TSR_SWEEP_SPAN   = True
RUN_TSR_SWEEP_PERF   = True
RUN_SENS_CONV_SPEED  = True
RUN_SENS_DISC        = True
RUN_SENS_AZIMUTHAL   = True
RUN_SENS_WAKE_LENGTH = True

# ── Plots ──────────────────────────────────────────────────────────────────────
PLOT_LL_1      = True
PLOT_LL_2      = True
PLOT_LL_3      = True
PLOT_LL_4      = True
PLOT_LL_5      = True
PLOT_LL_5_PERF = True
PLOT_SENS_AW   = True
PLOT_SENS_DISC = True
PLOT_SENS_DPSI = True
PLOT_SENS_WAKE = True

# ── Save ──────────────────────────────────────────────────────────────────────
SAVE_LL_RESULTS = True
SAVE_TABLES_PDF = True

# =============================================================================
# 1.  POLAR DATA
# =============================================================================

_df         = pd.read_excel("polar DU95W180 (3).xlsx", skiprows=3)
polar_alpha = _df["Alfa"].to_numpy()
polar_cl    = _df["Cl"].to_numpy()
polar_cd    = _df["Cd"].to_numpy()

# =============================================================================
# 2.  GLOBAL STYLE  &  COLOR SCHEME
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

_CB_PALETTE = sns.color_palette("colorblind").as_hex() + [
    '#8b0000',  # dark red
    '#556b2f',  # dark olive green
    '#4b0082',  # indigo
    '#ff6347',  # tomato
    '#20b2aa',  # light sea green
    '#8b4513',  # saddle brown
    '#2f4f4f',  # dark slate gray
    '#9400d3',  # dark violet
    '#6a5acd',  # slate blue
    '#b8860b',  # dark goldenrod
]

def _tsr_color(idx, n):
    return _CB_PALETTE[idx % len(_CB_PALETTE)]

def _sens_color(idx, n):
    return _CB_PALETTE[idx % len(_CB_PALETTE)]

# =============================================================================
# 3.  BLADE GEOMETRY
# =============================================================================

def blade_chord(r_R):
    return 3.0 * (1.0 - r_R) + 1.0

def blade_twist(r_R):
    return 14.0 * (1.0 - r_R) + Pitch

def make_panels(N=N_PANELS, distribution=DISTRIBUTION):
    r_root = RootLocation_R * Radius
    r_tip  = TipLocation_R  * Radius
    if distribution == "cosine":
        theta   = np.linspace(0.0, np.pi, N + 1)
        r_edges = r_root + 0.5 * (r_tip - r_root) * (1.0 - np.cos(theta))
    else:
        r_edges = np.linspace(r_root, r_tip, N + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    dr        = np.diff(r_edges)
    return r_edges, r_centers, dr

# =============================================================================
# 4.  WAKE GEOMETRY & VORTEX RING ASSEMBLY
# =============================================================================

def build_vortex_system(Omega, N=N_PANELS, N_wake=N_WAKE,
                        dpsi_deg=DPSI_DEG, a_w=A_WAKE,
                        distribution=DISTRIBUTION):
    r_edges, r_centers, dr = make_panels(N, distribution)
    U_wake  = U0 * (1.0 - a_w)
    dpsi    = np.radians(dpsi_deg)
    psi_arr = np.arange(0.0, N_wake * 2.0 * np.pi + dpsi * 0.5, dpsi)

    controlpoints = []
    rings         = []

    for k_blade in range(NBlades):
        angle_rot = 2.0 * np.pi / NBlades * k_blade
        cosR, sinR = np.cos(angle_rot), np.sin(angle_rot)

        for i in range(N):
            r   = r_centers[i]
            r_R = r / Radius
            chord     = blade_chord(r_R)
            twist_rad = np.radians(blade_twist(r_R))

            cp_y   = r * cosR
            cp_z   = r * sinR
            cp_rot = np.array([0.0, cp_y, cp_z])

            controlpoints.append({
                'r'        : r,
                'r_R'      : r_R,
                'chord'    : chord,
                'twist_rad': twist_rad,
                'coords'   : cp_rot,
                'dr'       : dr[i],
            })

            filaments = []

            # Bound vortex filament
            y_in  = r_edges[i]   * cosR;  z_in  = r_edges[i]   * sinR
            y_out = r_edges[i+1] * cosR;  z_out = r_edges[i+1] * sinR
            filaments.append({
                'x1': 0.0, 'y1': y_in,  'z1': z_in,
                'x2': 0.0, 'y2': y_out, 'z2': z_out,
            })

            # Trailing vortex at inner edge
            for j in range(len(psi_arr) - 1):
                t1 = psi_arr[j]   / Omega
                t2 = psi_arr[j+1] / Omega
                ri = r_edges[i]
                x1 = U_wake * t1
                y1 = ri * np.cos(-Omega * t1)
                z1 = ri * np.sin(-Omega * t1)
                x2 = U_wake * t2
                y2 = ri * np.cos(-Omega * t2)
                z2 = ri * np.sin(-Omega * t2)
                y1r = y1 * cosR - z1 * sinR;  z1r = y1 * sinR + z1 * cosR
                y2r = y2 * cosR - z2 * sinR;  z2r = y2 * sinR + z2 * cosR
                filaments.append({
                    'x1': x2, 'y1': y2r, 'z1': z2r,
                    'x2': x1, 'y2': y1r, 'z2': z1r,
                })

            # Trailing vortex at outer edge (opposite sign)
            for j in range(len(psi_arr) - 1):
                t1 = psi_arr[j]   / Omega
                t2 = psi_arr[j+1] / Omega
                ro = r_edges[i+1]
                x1 = U_wake * t1
                y1 = ro * np.cos(-Omega * t1)
                z1 = ro * np.sin(-Omega * t1)
                x2 = U_wake * t2
                y2 = ro * np.cos(-Omega * t2)
                z2 = ro * np.sin(-Omega * t2)
                y1r = y1 * cosR - z1 * sinR;  z1r = y1 * sinR + z1 * cosR
                y2r = y2 * cosR - z2 * sinR;  z2r = y2 * sinR + z2 * cosR
                filaments.append({
                    'x1': x1, 'y1': y1r, 'z1': z1r,
                    'x2': x2, 'y2': y2r, 'z2': z2r,
                })

            rings.append(filaments)

    return controlpoints, rings

# =============================================================================
# 5.  BIOT-SAVART KERNEL
# =============================================================================

def biot_savart_segment(x1, x2, xp, r_core=R_CORE):
    R1    = xp - x1
    R2    = xp - x2
    cross = np.cross(R1, R2)
    cross_sq = np.einsum('ij,ij->i', cross, cross)

    R1_mag = np.linalg.norm(R1, axis=1)
    R2_mag = np.linalg.norm(R2, axis=1)

    inside = (cross_sq < r_core**2) | (R1_mag < r_core) | (R2_mag < r_core)

    R0R1 = np.einsum('j,ij->i', x2 - x1, R1)
    R0R2 = np.einsum('j,ij->i', x2 - x1, R2)

    R1_mag   = np.where(R1_mag   < 1e-12, 1e-12, R1_mag)
    R2_mag   = np.where(R2_mag   < 1e-12, 1e-12, R2_mag)
    cross_sq = np.where(cross_sq < 1e-24, 1e-24, cross_sq)

    K = (1.0 / (4.0 * np.pi * cross_sq)) * (R0R1 / R1_mag - R0R2 / R2_mag)
    K = np.where(inside, 0.0, K)

    return K[:, None] * cross

# =============================================================================
# 6.  INFLUENCE MATRIX ASSEMBLY
# =============================================================================

def assemble_influence_matrix(controlpoints, rings):
    N_total = len(controlpoints)
    A_u = np.zeros((N_total, N_total))
    A_v = np.zeros((N_total, N_total))
    A_w = np.zeros((N_total, N_total))

    XP = np.array([cp['coords'] for cp in controlpoints])

    for j, ring in enumerate(rings):
        vel_j = np.zeros((N_total, 3))
        for fil in ring:
            x1 = np.array([fil['x1'], fil['y1'], fil['z1']])
            x2 = np.array([fil['x2'], fil['y2'], fil['z2']])
            vel_j += biot_savart_segment(x1, x2, XP)
        A_u[:, j] = vel_j[:, 0]
        A_v[:, j] = vel_j[:, 1]
        A_w[:, j] = vel_j[:, 2]

    return A_u, A_v, A_w

# =============================================================================
# 7.  CIRCULATION SOLVER  
# =============================================================================

def solve_circulation(A_u, A_v, A_w, controlpoints, Omega,
                      tol=1e-4, max_iter=1000):
    N_total = len(controlpoints)
    Gamma   = np.zeros(N_total)
    R_prev  = np.zeros(N_total)
    omega   = 0.1
    err     = np.inf

    for k in range(max_iter):
        u_ind = A_u @ Gamma
        v_ind = A_v @ Gamma
        w_ind = A_w @ Gamma

        Gamma_new = np.zeros(N_total)

        for i, cp in enumerate(controlpoints):
            r_cp  = cp['coords']
            v_rot = np.cross(np.array([-Omega, 0.0, 0.0]), r_cp)

            V_ax = U0 + u_ind[i] + v_rot[0]
            V_y  = v_ind[i] + v_rot[1]
            V_z  = w_ind[i] + v_rot[2]

            r_mag    = max(cp['r'], 1e-12)
            azim_dir = np.cross(np.array([-1.0 / r_mag, 0.0, 0.0]), r_cp)
            V_tan    = float(np.dot(azim_dir, [V_ax, V_y, V_z]))

            V_eff = np.sqrt(V_ax**2 + V_tan**2)
            phi   = np.arctan2(V_ax, V_tan)
            alpha = phi - cp['twist_rad']

            cl = float(np.interp(np.degrees(alpha), polar_alpha, polar_cl))
            Gamma_new[i] = 0.5 * cp['chord'] * V_eff * cl

        # Aitken relaxation
        R_k = Gamma_new - Gamma
        err = np.max(np.abs(R_k)) / max(np.max(np.abs(Gamma_new)), 1e-6)

        if err < tol:
            Gamma = Gamma + omega * R_k
            converged = True
            print(f"    Converged in {k+1} iterations  (err = {err:.2e})")
            return Gamma, A_u @ Gamma, A_v @ Gamma, A_w @ Gamma, k + 1, converged

        if k > 0:
            delta_R = R_k - R_prev
            den     = np.dot(delta_R, delta_R)
            if den > 1e-16:
                omega = -omega * np.dot(R_prev, delta_R) / den
            omega = float(np.clip(omega, 0.01, 0.5))

        Gamma  = Gamma + omega * R_k
        R_prev = R_k.copy()

    print(f"    [WARNING] Did not converge after {max_iter} iterations  (err = {err:.2e})")
    converged = False
    return Gamma, A_u @ Gamma, A_v @ Gamma, A_w @ Gamma, max_iter, converged

# =============================================================================
# 8.  POST-PROCESSING
# =============================================================================

def post_process(Gamma, u_ind, v_ind, w_ind, controlpoints, Omega):
    N        = len(controlpoints) // NBlades
    res_rows = []

    for i in range(N):
        cp    = controlpoints[i]
        r_cp  = cp['coords']
        v_rot = np.cross(np.array([-Omega, 0.0, 0.0]), r_cp)

        V_ax = U0 + u_ind[i] + v_rot[0]
        V_y  = v_ind[i] + v_rot[1]
        V_z  = w_ind[i] + v_rot[2]

        r_mag    = max(cp['r'], 1e-12)
        azim_dir = np.cross(np.array([-1.0 / r_mag, 0.0, 0.0]), r_cp)
        V_tan    = float(np.dot(azim_dir, [V_ax, V_y, V_z]))

        V_eff = np.sqrt(V_ax**2 + V_tan**2)
        phi   = np.arctan2(V_ax, V_tan)
        alpha = phi - cp['twist_rad']

        cl = float(np.interp(np.degrees(alpha), polar_alpha, polar_cl))
        cd = float(np.interp(np.degrees(alpha), polar_alpha, polar_cd))

        lift = 0.5 * rho * V_eff**2 * cp['chord'] * cl
        drag = 0.5 * rho * V_eff**2 * cp['chord'] * cd
        F_n  = lift * np.cos(phi) + drag * np.sin(phi)
        F_t  = lift * np.sin(phi) - drag * np.cos(phi)

        a_ax  = -(u_ind[i] + v_rot[0]) / U0
        a_tan = V_tan / (cp['r'] * Omega) - 1.0

        res_rows.append([
            a_ax, a_tan, cp['r_R'], F_n, F_t,
            Gamma[i], float(np.degrees(alpha)), float(np.degrees(phi)), cl, cd,
        ])

    res = np.array(res_rows, dtype=float)
    dr  = np.array([cp['dr'] for cp in controlpoints[:N]])

    T  = NBlades * np.sum(res[:, 3] * dr)
    Q  = NBlades * np.sum(res[:, 4] * np.array([cp['r'] for cp in controlpoints[:N]]) * dr)
    P  = Omega * Q
    A  = np.pi * Radius**2

    CT = T / (0.5 * rho * U0**2 * A)
    CP = P / (0.5 * rho * U0**3 * A)

    mean_a = float(np.sum(res[:, 0] * dr) / np.sum(dr))

    return res, CT, CP, mean_a

# =============================================================================
# 9.  SINGLE INNER SOLVE
# =============================================================================

def _single_solve(TSR, N, N_wake, dpsi_deg, a_w, distribution, verbose):
    Omega = U0 * TSR / Radius
    if verbose:
        print(f"    Building vortex system  (N={N}, N_wake={N_wake}, "
              f"dpsi={dpsi_deg}°, a_w={a_w:.4f}, dist={distribution}) ...")
    cps, rings = build_vortex_system(
        Omega, N=N, N_wake=N_wake, dpsi_deg=dpsi_deg,
        a_w=a_w, distribution=distribution)

    if verbose:
        print(f"    Assembling influence matrix ({len(cps)}×{len(cps)}) ...")
    A_u, A_v, A_w = assemble_influence_matrix(cps, rings)

    if verbose:
        print(f"    Solving circulation ...")
    Gamma, u_ind, v_ind, w_ind, n_iter, converged = solve_circulation(
        A_u, A_v, A_w, cps, Omega)

    res, CT, CP, mean_a = post_process(Gamma, u_ind, v_ind, w_ind, cps, Omega)
    return res, CT, CP, mean_a, n_iter, converged

# =============================================================================
# 10.  ROTOR EVALUATOR
# =============================================================================

_solver_meta = {}

def run_case(TSR, N=N_PANELS, N_wake=N_WAKE, dpsi_deg=DPSI_DEG,
             a_w=None, distribution=DISTRIBUTION, verbose=True,
             _meta_key=None):
    fixed_aw = a_w is not None
    iterate  = USE_ITERATED_AW and not fixed_aw

    if a_w is None:
        a_w_run = AW_ITER_INIT if iterate else A_WAKE
    else:
        a_w_run = a_w

    if iterate:
        if verbose:
            print(f"  [AW iter] TSR={TSR}  initial a_w={a_w_run:.4f}")
        n_outer = 0
        for outer_k in range(AW_ITER_MAX):
            n_outer += 1
            res, CT, CP, mean_a, n_iter, converged = _single_solve(
                TSR, N, N_wake, dpsi_deg, a_w_run, distribution, verbose)
            aw_err = abs(mean_a - a_w_run)
            if verbose:
                print(f"  [AW iter {outer_k+1}] a_w={a_w_run:.4f}  "
                      f"mean_a={mean_a:.4f}  err={aw_err:.2e}  "
                      f"CT={CT:.4f}  CP={CP:.4f}")
            if aw_err < AW_ITER_TOL:
                if verbose:
                    print(f"  [AW iter] Converged: a_w = {mean_a:.4f} "
                          f"after {outer_k+1} outer iteration(s)")
                a_w_run = mean_a
                break
            a_w_run = mean_a
        else:
            if verbose:
                print(f"  [AW iter] WARNING: outer loop did not converge "
                      f"after {AW_ITER_MAX} iterations  (err={aw_err:.2e})")
    else:
        if verbose:
            print(f"  TSR={TSR}  (fixed a_w={a_w_run:.4f})")
        res, CT, CP, mean_a, n_iter, converged = _single_solve(
            TSR, N, N_wake, dpsi_deg, a_w_run, distribution, verbose)
        n_outer = 1

    if verbose:
        mode_str = f"iterated a_w={a_w_run:.4f}" if iterate else f"fixed a_w={a_w_run:.4f}"
        print(f"  TSR={TSR}  CT={CT:.4f}  CP={CP:.4f}  "
              f"mean_a={mean_a:.4f}  ({mode_str})")

    if _meta_key is not None:
        _solver_meta[_meta_key] = (n_iter, converged, mean_a, a_w_run, n_outer)

    return res, CT, CP

# =============================================================================
# 11.  SAVE / SHOW HELPER
# =============================================================================

LL_RESULTS_PATH = "LLM_results.npz"

save_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "LLM_plots")
os.makedirs(save_folder, exist_ok=True)

tables_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "LLM_tables")
os.makedirs(tables_folder, exist_ok=True)

def save_fig(name):
    stem  = name.rsplit(".", 1)[0] if "." in name else name
    fpath = os.path.join(save_folder, stem + ".pdf")
    plt.savefig(fpath)
    print(f"  Saved: {stem}.pdf")
    if SHOW_PLOTS: plt.show()
    else:          plt.close()

def _skip(name, reason):
    print(f"  [SKIP] {name} — {reason}")
    plt.close("all")

norm_val = 0.5 * rho * U0**2 * Radius

# =============================================================================
# 11b.  ANNOTATION HELPER — iterative arrow placement
# =============================================================================

def _annotate_increasing_direction(ax, x_data_list, y_data_list,
                                   label="increasing", color="#333333",
                                   margin_frac=0.08, force_direction=None,
                                   x_pos=None, y_pos=None):
    """Place a single-headed arrow showing the direction in which curves shift
    as the parameter increases.

    Parameters
    ----------
    ax               : Axes to annotate
    x_data_list      : list of 1-D arrays, one per curve (r/R values)
    y_data_list      : list of 1-D arrays, one per curve (quantity values)
    label            : text to place beside the arrow
    color            : arrow and text colour
    margin_frac      : fraction of axis range to offset the annotation
    force_direction  : None (auto-detect from data), 'up', 'down', or 'left'.
                       'left' draws a horizontal arrow along the x-axis instead.
    x_pos            : manual r/R position for the vertical arrow (overrides
                       auto-detection); ignored for 'left' arrows.
    y_pos            : manual y position for the horizontal arrow y_level
                       (overrides auto-detection); ignored for vertical arrows.
    """
    if len(y_data_list) < 2:
        return

    # ── Horizontal (left) arrow — drawn at a fixed y position ────────────
    if force_direction == 'left':
        ax_xmin, ax_xmax = ax.get_xlim()
        ax_ymin, ax_ymax = ax.get_ylim()
        x_range = ax_xmax - ax_xmin
        y_range = ax_ymax - ax_ymin

        if y_pos is not None:
            y_level = float(y_pos)
        else:
            # Find the x position of maximum spread to anchor the y level
            x_ref   = x_data_list[0]
            y_first = y_data_list[0]
            y_last  = np.interp(x_ref, x_data_list[-1], y_data_list[-1])
            spread  = np.abs(y_last - y_first)
            idx_max = int(np.argmax(spread))
            y_level = float(y_first[idx_max])
            # Nudge y_level away from axes edges
            y_level = float(np.clip(y_level,
                                    ax_ymin + 0.08 * y_range,
                                    ax_ymax - 0.08 * y_range))

        # Arrow spans most of the x axis
        x_tail = ax_xmax - 0.05 * x_range
        x_head = ax_xmin + 0.05 * x_range

        ax.annotate(
            "", xy=(x_head, y_level), xytext=(x_tail, y_level),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
        )
        x_mid   = 0.5 * (x_tail + x_head)
        y_label = y_level + 0.04 * y_range
        ax.text(x_mid, y_label, label, fontsize=9, color=color,
                va="bottom", ha="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec=color, alpha=0.75, lw=0.8))
        return

    # ── Vertical arrow ────────────────────────────────────────────────────
    x_ref   = x_data_list[0]
    y_first = y_data_list[0]
    y_last  = np.interp(x_ref, x_data_list[-1], y_data_list[-1])

    spread  = np.abs(y_last - y_first)
    if x_pos is not None:
        # Snap to the nearest data point to the requested r/R
        idx_max = int(np.argmin(np.abs(x_ref - x_pos)))
    else:
        idx_max = int(np.argmax(spread))
    x_arrow = float(x_ref[idx_max])

    y0 = float(y_first[idx_max])
    y1 = float(y_last[idx_max])

    # Auto-detect or override direction
    if force_direction == 'up':
        direction_up = True
    elif force_direction == 'down':
        direction_up = False
    else:
        direction_up = y1 > y0

    y_low  = min(y0, y1)
    y_high = max(y0, y1)

    ax_ymin, ax_ymax = ax.get_ylim()
    y_range = ax_ymax - ax_ymin
    offset  = margin_frac * y_range

    if direction_up:
        y_tail = y_low  - offset
        y_head = y_high + offset
    else:
        y_tail = y_high + offset
        y_head = y_low  - offset

    guard  = 0.02 * y_range
    y_tail = float(np.clip(y_tail, ax_ymin + guard, ax_ymax - guard))
    y_head = float(np.clip(y_head, ax_ymin + guard, ax_ymax - guard))

    # Single-headed arrow: tail -> head
    ax.annotate(
        "", xy=(x_arrow, y_head), xytext=(x_arrow, y_tail),
        arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
    )

    y_mid   = 0.5 * (y_tail + y_head)
    x_label = x_arrow + 0.03 * (ax.get_xlim()[1] - ax.get_xlim()[0])
    ax.text(x_label, y_mid, label, fontsize=9, color=color,
            va="center", ha="left",
            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                      ec=color, alpha=0.75, lw=0.8))

# =============================================================================
# 11c.  PDF TABLE HELPER
# =============================================================================

def _make_table_pdf(filename, title, headers, rows, col_widths=None,
                    footnote=None):
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle,
                                     Paragraph, Spacer)
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER

    fpath    = os.path.join(tables_folder, filename + ".pdf")
    pagesize = landscape(A4) if len(headers) > 6 else A4
    doc = SimpleDocTemplate(
        fpath, pagesize=pagesize,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )
    styles      = getSampleStyleSheet()
    title_style = ParagraphStyle("TableTitle", parent=styles["Heading2"],
                                  alignment=TA_CENTER, spaceAfter=6)
    foot_style  = ParagraphStyle("Footnote", parent=styles["Normal"],
                                  fontSize=8, textColor=colors.grey, spaceBefore=4)

    story = []
    story.append(Paragraph(title, title_style))
    story.append(Spacer(1, 0.3*cm))

    table_data = [headers] + rows
    page_w     = pagesize[0] - 4*cm
    if col_widths is None:
        col_widths = [page_w / len(headers)] * len(headers)

    tbl = Table(table_data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  colors.HexColor("#2c3e50")),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0),  10),
        ("ALIGN",         (0, 0), (-1, 0),  "CENTER"),
        ("BOTTOMPADDING", (0, 0), (-1, 0),  8),
        ("TOPPADDING",    (0, 0), (-1, 0),  8),
        ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",      (0, 1), (-1, -1), 9),
        ("ALIGN",         (0, 1), (-1, -1), "CENTER"),
        ("TOPPADDING",    (0, 1), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 5),
        *[("BACKGROUND",  (0, i), (-1, i),  colors.HexColor("#ecf0f1"))
          for i in range(2, len(table_data), 2)],
        ("GRID",  (0, 0), (-1, -1), 0.5, colors.HexColor("#bdc3c7")),
        ("BOX",   (0, 0), (-1, -1), 1.0, colors.HexColor("#2c3e50")),
        ("VALIGN",(0, 0), (-1, -1), "MIDDLE"),
    ]))

    story.append(tbl)
    if footnote:
        story.append(Spacer(1, 0.3*cm))
        story.append(Paragraph(footnote, foot_style))

    doc.build(story)
    print(f"  Saved table: {filename}.pdf")


def _meta_row(mk, default_aw):
    entry = _solver_meta.get(mk)
    if entry is None:
        return "—", True, "—", default_aw, 1
    n_iter, converged, mean_a, a_w_used, n_outer = entry
    return n_iter, converged, mean_a, a_w_used, n_outer


def save_all_tables():
    print("\n" + "="*60)
    print("Saving summary tables to LLM_tables/ ...")

    aw_mode_str = ("iterated" if USE_ITERATED_AW else f"fixed={A_WAKE}")

    if sweep_data_span:
        headers = [u"\u03bb", "CT", "CP", u"\u0101 [-]", "a_w used", "iters", "conv."]
        rows = []
        for TSR in TSR_SWEEP_SPAN:
            if TSR not in sweep_data_span: continue
            mk  = f"tsr_{TSR}"
            ni, cv, mean_a, aw_used, _ = _meta_row(mk, A_WAKE)
            CT  = tsr_CT_span[TSR_SWEEP_SPAN.index(TSR)]
            CP  = tsr_CP_span[TSR_SWEEP_SPAN.index(TSR)]
            mean_a_str = f"{mean_a:.4f}" if isinstance(mean_a, float) else mean_a
            rows.append([str(TSR), f"{CT:.4f}", f"{CP:.4f}",
                         mean_a_str, f"{aw_used:.4f}", str(ni),
                         u"\u2713" if cv else u"\u2717"])
        _make_table_pdf(
            "Table_1_TSR_sweep",
            "Table 1 — TSR Sweep Results (Lifting Line)", headers, rows,
            col_widths=[50, 70, 70, 65, 65, 55, 50],
            footnote=(f"Baseline: N={N_PANELS} panels (cosine), N_wake={N_WAKE} rotations, "
                      f"dpsi={DPSI_DEG} deg, a_w mode={aw_mode_str}, R={Radius} m, "
                      f"U0={U0} m/s, B={NBlades} blades. "
                      f"a-bar = span-averaged axial induction (area-weighted)."))

    if sens_aw_data:
        headers = ["a_w input", "CT", "CP", u"\u0101 [-]", "dCT", "dCP", "iters", "conv."]
        ref_CT  = sens_aw_data.get(A_WAKE, (None, None, None))[1]
        ref_CP  = sens_aw_data.get(A_WAKE, (None, None, None))[2]
        rows = []
        for aw in SENS_AW_LIST:
            if aw not in sens_aw_data: continue
            _, CT, CP = sens_aw_data[aw]
            mk = f"aw_{aw}"
            ni, cv, mean_a, aw_used, _ = _meta_row(mk, aw)
            dCT = f"{CT - ref_CT:+.4f}" if ref_CT is not None else "—"
            dCP = f"{CP - ref_CP:+.4f}" if ref_CP is not None else "—"
            mean_a_str = f"{mean_a:.4f}" if isinstance(mean_a, float) else mean_a
            rows.append([f"{aw:.2f}", f"{CT:.4f}", f"{CP:.4f}",
                         mean_a_str, dCT, dCP,
                         str(ni), u"\u2713" if cv else u"\u2717"])
        _make_table_pdf(
            "Table_2_sensitivity_aw",
            f"Table 2 — Sensitivity: Wake Convection Speed a_w  (TSR={SENS_TSR})",
            headers, rows, col_widths=[55, 65, 65, 60, 60, 60, 50, 45],
            footnote=(f"Note: a_w sensitivity always uses fixed input values. "
                      f"Reference: a_w={A_WAKE}. "
                      f"Other params: N={N_PANELS}, N_wake={N_WAKE}, dpsi={DPSI_DEG} deg. "
                      f"a-bar = span-averaged axial induction."))

    if sens_disc_data:
        headers = ["N", "dist.", "CT", "CP", u"\u0101 [-]", "dCT", "dCP", "iters", "conv."]
        ref_key = (N_PANELS, "cosine")
        ref_CT  = sens_disc_data[ref_key][1] if ref_key in sens_disc_data else None
        ref_CP  = sens_disc_data[ref_key][2] if ref_key in sens_disc_data else None
        rows = []
        for N in SENS_N_LIST:
            for dist in ["cosine", "constant"]:
                if (N, dist) not in sens_disc_data: continue
                _, CT, CP = sens_disc_data[(N, dist)]
                mk = f"disc_{N}_{dist}"
                ni, cv, mean_a, aw_used, _ = _meta_row(mk, A_WAKE)
                dCT = f"{CT - ref_CT:+.4f}" if ref_CT is not None else "—"
                dCP = f"{CP - ref_CP:+.4f}" if ref_CP is not None else "—"
                mean_a_str = f"{mean_a:.4f}" if isinstance(mean_a, float) else mean_a
                rows.append([str(N), dist.capitalize(), f"{CT:.4f}", f"{CP:.4f}",
                             mean_a_str, dCT, dCP,
                             str(ni), u"\u2713" if cv else u"\u2717"])
        _make_table_pdf(
            "Table_3_sensitivity_disc",
            f"Table 3 — Sensitivity: Blade Discretisation  (TSR={SENS_TSR})",
            headers, rows, col_widths=[35, 58, 58, 58, 55, 55, 55, 45, 42],
            footnote=(f"Reference: N={N_PANELS}, cosine. "
                      f"Other params: N_wake={N_WAKE}, dpsi={DPSI_DEG} deg, "
                      f"a_w mode={aw_mode_str}. "
                      f"a-bar = span-averaged axial induction."))

    if sens_dpsi_data:
        headers = ["dpsi [deg]", "N_steps", "CT", "CP", u"\u0101 [-]",
                   "dCT", "dCP", "iters", "conv."]
        ref_CT  = sens_dpsi_data.get(DPSI_DEG, (None, None, None))[1]
        ref_CP  = sens_dpsi_data.get(DPSI_DEG, (None, None, None))[2]
        rows = []
        for dpsi in SENS_DPSI_LIST:
            if dpsi not in sens_dpsi_data: continue
            _, CT, CP = sens_dpsi_data[dpsi]
            mk = f"dpsi_{dpsi}"
            ni, cv, mean_a, aw_used, _ = _meta_row(mk, A_WAKE)
            n_steps = int(round(360.0 / dpsi))
            dCT = f"{CT - ref_CT:+.4f}" if ref_CT is not None else "—"
            dCP = f"{CP - ref_CP:+.4f}" if ref_CP is not None else "—"
            mean_a_str = f"{mean_a:.4f}" if isinstance(mean_a, float) else mean_a
            rows.append([f"{dpsi:.1f}", str(n_steps), f"{CT:.4f}", f"{CP:.4f}",
                         mean_a_str, dCT, dCP,
                         str(ni), u"\u2713" if cv else u"\u2717"])
        _make_table_pdf(
            "Table_4_sensitivity_dpsi",
            f"Table 4 — Sensitivity: Azimuthal Step  (TSR={SENS_TSR})",
            headers, rows, col_widths=[58, 52, 58, 58, 55, 55, 55, 45, 42],
            footnote=(f"Reference: dpsi={DPSI_DEG} deg. "
                      f"Other params: N={N_PANELS}, N_wake={N_WAKE}, "
                      f"a_w mode={aw_mode_str}. "
                      f"a-bar = span-averaged axial induction."))

    if sens_wake_data:
        headers = ["N_wake", "N_fil.", "CT", "CP", u"\u0101 [-]",
                   "dCT", "dCP", "iters", "conv."]
        ref_CT  = sens_wake_data.get(N_WAKE, (None, None, None))[1]
        ref_CP  = sens_wake_data.get(N_WAKE, (None, None, None))[2]
        rows = []
        for nw in SENS_NWAKE_LIST:
            if nw not in sens_wake_data: continue
            _, CT, CP = sens_wake_data[nw]
            mk = f"nwake_{nw}"
            ni, cv, mean_a, aw_used, _ = _meta_row(mk, A_WAKE)
            n_fil = int(round(nw * 360.0 / DPSI_DEG))
            dCT = f"{CT - ref_CT:+.4f}" if ref_CT is not None else "—"
            dCP = f"{CP - ref_CP:+.4f}" if ref_CP is not None else "—"
            mean_a_str = f"{mean_a:.4f}" if isinstance(mean_a, float) else mean_a
            rows.append([str(nw), str(n_fil), f"{CT:.4f}", f"{CP:.4f}",
                         mean_a_str, dCT, dCP,
                         str(ni), u"\u2713" if cv else u"\u2717"])
        _make_table_pdf(
            "Table_5_sensitivity_wake",
            f"Table 5 — Sensitivity: Wake Length  (TSR={SENS_TSR})",
            headers, rows, col_widths=[52, 48, 58, 58, 55, 55, 55, 45, 42],
            footnote=(f"Reference: N_wake={N_WAKE}. "
                      f"Other params: N={N_PANELS}, dpsi={DPSI_DEG} deg, "
                      f"a_w mode={aw_mode_str}. "
                      f"a-bar = span-averaged axial induction."))

    print("All tables saved.")


# =============================================================================
# 12.  MAIN COMPUTATIONS
# =============================================================================

sweep_data_span = {}
tsr_CT_span     = []
tsr_CP_span     = []
tsr_CT_perf     = []
tsr_CP_perf     = []
sens_aw_data    = {}
sens_disc_data  = {}
sens_dpsi_data  = {}
sens_wake_data  = {}

timing_disc  = {}
timing_dpsi  = {}
timing_wake  = {}   # nw -> float seconds  [NEW]

print(f"\nWake convection mode: {'ITERATED a_w' if USE_ITERATED_AW else f'FIXED a_w = {A_WAKE}'}")

# ── Spanwise TSR sweep ────────────────────────────────────────────────────────
if RUN_TSR_SWEEP_SPAN:
    print("\n" + "="*60)
    print(f"Running spanwise TSR sweep {TSR_SWEEP_SPAN} ...")
    for TSR in TSR_SWEEP_SPAN:
        print(f"\n  TSR = {TSR}:")
        res, CT, CP = run_case(TSR, _meta_key=f"tsr_{TSR}")
        sweep_data_span[TSR] = res
        tsr_CT_span.append(CT)
        tsr_CP_span.append(CP)
    tsr_CT_span = np.array(tsr_CT_span)
    tsr_CP_span = np.array(tsr_CP_span)

# ── Performance sweep ─────────────────────────────────────────────────────────
if RUN_TSR_SWEEP_PERF:
    print("\n" + "="*60)
    print(f"Running performance TSR sweep λ = {TSR_SWEEP_PERF[0]:.1f} … "
          f"{TSR_SWEEP_PERF[-1]:.1f}  (step 0.5) ...")
    for TSR in TSR_SWEEP_PERF:
        print(f"\n  TSR = {TSR}:")
        if TSR in sweep_data_span:
            idx = TSR_SWEEP_SPAN.index(TSR)
            tsr_CT_perf.append(float(tsr_CT_span[idx]))
            tsr_CP_perf.append(float(tsr_CP_span[idx]))
            print(f"    (re-using span sweep result)  CT={tsr_CT_span[idx]:.4f}  "
                  f"CP={tsr_CP_span[idx]:.4f}")
        else:
            _, CT_p, CP_p = run_case(TSR, verbose=True)
            tsr_CT_perf.append(CT_p)
            tsr_CP_perf.append(CP_p)
    tsr_CT_perf = np.array(tsr_CT_perf)
    tsr_CP_perf = np.array(tsr_CP_perf)

# ── Sensitivity: convection speed ────────────────────────────────────────────
if RUN_SENS_CONV_SPEED:
    print("\n" + "="*60)
    print(f"Running sensitivity: convection speed (TSR={SENS_TSR}) ...")
    for a_w in SENS_AW_LIST:
        print(f"\n  a_w = {a_w}:")
        res, CT, CP = run_case(SENS_TSR, a_w=a_w, _meta_key=f"aw_{a_w}")
        sens_aw_data[a_w] = (res, CT, CP)

# ── Sensitivity: blade discretisation ────────────────────────────────────────
if RUN_SENS_DISC:
    print("\n" + "="*60)
    print(f"Running sensitivity: discretisation (TSR={SENS_TSR}) ...")
    for N in SENS_N_LIST:
        for dist in ["cosine", "constant"]:
            print(f"\n  N={N}, distribution={dist}:")
            _t0 = time.perf_counter()
            res, CT, CP = run_case(SENS_TSR, N=N, distribution=dist,
                                   _meta_key=f"disc_{N}_{dist}")
            timing_disc[(N, dist)] = time.perf_counter() - _t0
            sens_disc_data[(N, dist)] = (res, CT, CP)

# ── Sensitivity: azimuthal step ───────────────────────────────────────────────
if RUN_SENS_AZIMUTHAL:
    print("\n" + "="*60)
    print(f"Running sensitivity: azimuthal step (TSR={SENS_TSR}) ...")
    for dpsi in SENS_DPSI_LIST:
        print(f"\n  dpsi = {dpsi} deg:")
        _t0 = time.perf_counter()
        res, CT, CP = run_case(SENS_TSR, dpsi_deg=dpsi, _meta_key=f"dpsi_{dpsi}")
        timing_dpsi[dpsi] = time.perf_counter() - _t0
        sens_dpsi_data[dpsi] = (res, CT, CP)

# ── Sensitivity: wake length  (now with timing) ───────────────────────────────
if RUN_SENS_WAKE_LENGTH:
    print("\n" + "="*60)
    print(f"Running sensitivity: wake length (TSR={SENS_TSR}) ...")
    for nw in SENS_NWAKE_LIST:
        print(f"\n  N_wake = {nw} rotations:")
        _t0 = time.perf_counter()                          # <-- NEW
        res, CT, CP = run_case(SENS_TSR, N_wake=nw, _meta_key=f"nwake_{nw}")
        timing_wake[nw] = time.perf_counter() - _t0       # <-- NEW
        sens_wake_data[nw] = (res, CT, CP)

# =============================================================================
# 13.  PLOTS — LL.1  (inflow angle φ and AoA α)
#      Annotate with direction arrow showing increasing λ
# =============================================================================

n_span = len(TSR_SWEEP_SPAN)

if PLOT_LL_1 and sweep_data_span:
    for qty_col, ylabel, fname in [
            (7, r"$\phi$ [deg]",   "LL_1a_inflow_angle_vs_rR"),
            (6, r"$\alpha$ [deg]", "LL_1b_angle_of_attack_vs_rR")]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for k, TSR in enumerate(TSR_SWEEP_SPAN):
            res = sweep_data_span[TSR]
            ax.plot(res[:, 2], res[:, qty_col], color=_tsr_color(k, n_span), lw=2,
                    label=rf"$\lambda={TSR}$")
        ax.set_xlabel("r/R"); ax.set_ylabel(ylabel)
        ax.legend(); ax.grid(True)
        fig.tight_layout(); save_fig(fname)
elif PLOT_LL_1:
    _skip("PLOT_LL_1", "span sweep data missing")

# =============================================================================
# 14.  PLOTS — LL.2  (induction factors)
# =============================================================================

if PLOT_LL_2 and sweep_data_span:
    for qty_col, ylabel, fname in [
            (0, r"$a$ [-]",  "LL_2a_axial_induction_vs_rR"),
            (1, r"$a'$ [-]", "LL_2b_tangential_induction_vs_rR")]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for k, TSR in enumerate(TSR_SWEEP_SPAN):
            res = sweep_data_span[TSR]
            ax.plot(res[:, 2], res[:, qty_col], color=_tsr_color(k, n_span), lw=2,
                    label=rf"$\lambda={TSR}$")
        ax.set_xlabel("r/R"); ax.set_ylabel(ylabel)
        ax.legend(); ax.grid(True)
        fig.tight_layout(); save_fig(fname)
elif PLOT_LL_2:
    _skip("PLOT_LL_2", "span sweep data missing")

# =============================================================================
# 15.  PLOTS — LL.3  (loading distributions)
# =============================================================================

if PLOT_LL_3 and sweep_data_span:
    for qty_col, ylabel, fname in [
            (3, r"$C_n = F_n\,/\,(\frac{1}{2}\rho U_\infty^2 R)$", "LL_3a_normal_loading_Cn_vs_rR"),
            (4, r"$C_t = F_t\,/\,(\frac{1}{2}\rho U_\infty^2 R)$", "LL_3b_azimuthal_loading_Ct_vs_rR")]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for k, TSR in enumerate(TSR_SWEEP_SPAN):
            res = sweep_data_span[TSR]
            ax.plot(res[:, 2], res[:, qty_col] / norm_val, color=_tsr_color(k, n_span), lw=2,
                    label=rf"$\lambda={TSR}$")
        ax.set_xlabel("r/R"); ax.set_ylabel(ylabel)
        ax.legend(); ax.grid(True)
        fig.tight_layout(); save_fig(fname)
elif PLOT_LL_3:
    _skip("PLOT_LL_3", "span sweep data missing")

# =============================================================================
# 16.  PLOTS — LL.4  (circulation)
# =============================================================================

if PLOT_LL_4 and sweep_data_span:
    fig, ax = plt.subplots(figsize=(8, 5))
    for k, TSR in enumerate(TSR_SWEEP_SPAN):
        res    = sweep_data_span[TSR]
        Omega  = U0 * TSR / Radius
        norm_G = np.pi * U0**2 / (NBlades * Omega)
        ax.plot(res[:, 2], res[:, 5] / norm_G, color=_tsr_color(k, n_span), lw=2,
                label=rf"$\lambda={TSR}$")
    ax.set_xlabel("r/R")
    ax.set_ylabel(r"$\Gamma\,/\,(\pi U_\infty^2 / (B\,\Omega))$ [-]")
    ax.legend(); ax.grid(True)
    fig.tight_layout(); save_fig("LL_4_circulation_vs_rR")
elif PLOT_LL_4:
    _skip("PLOT_LL_4", "span sweep data missing")

# =============================================================================
# 17.  PLOTS — LL.5  (CT and CP vs TSR)
# =============================================================================

if PLOT_LL_5_PERF and len(tsr_CT_perf) > 0:
    for vals, ylabel, fname, col in [
            (tsr_CT_perf, r"$C_T$ [-]", "LL_5c_CT_vs_TSR", "#0173b2"),
            (tsr_CP_perf, r"$C_P$ [-]", "LL_5d_CP_vs_TSR", "#029e73")]:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(TSR_SWEEP_PERF, vals, "o-", color=col, lw=2, ms=5, zorder=2)
        ax.set_xlabel(r"Tip-speed ratio $\lambda$ [-]")
        ax.set_ylabel(ylabel)
        ax.set_xlim(3.5, 12.5)
        ax.set_xticks(np.arange(4, 13, 1))
        ax.grid(True)
        fig.tight_layout()
        save_fig(fname)
elif PLOT_LL_5_PERF:
    _skip("PLOT_LL_5_PERF", "performance sweep data missing")

# =============================================================================
# 18.  PLOTS — SENSITIVITY: convection speed
# =============================================================================

if PLOT_SENS_AW and sens_aw_data:
    n_aw    = len(SENS_AW_LIST)
    aw_vals = SENS_AW_LIST
    ct_vals = [sens_aw_data[aw][1] for aw in aw_vals]
    cp_vals = [sens_aw_data[aw][2] for aw in aw_vals]
    aw_arr  = np.array(aw_vals)

    meana_vals = []
    for aw in aw_vals:
        entry = _solver_meta.get(f"aw_{aw}")
        if entry is not None:
            meana_vals.append(entry[2])
        else:
            res, _, _ = sens_aw_data[aw]
            meana_vals.append(float(np.mean(res[:, 0])))
    meana_arr = np.array(meana_vals)
    resid_arr = meana_arr - aw_arr

    try:
        sc_aw = float(np.interp(0.0, resid_arr[::-1], aw_arr[::-1]))
        sc_CT = float(np.interp(sc_aw, aw_arr, np.array(ct_vals)))
        sc_CP = float(np.interp(sc_aw, aw_arr, np.array(cp_vals)))
        has_sc = True
        print(f"  Self-consistent a_w = {sc_aw:.4f}  CT={sc_CT:.4f}  CP={sc_CP:.4f}")
    except Exception:
        has_sc = False
        sc_aw  = A_WAKE

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(aw_arr, ct_vals, "o-", color="#0173b2", lw=2, ms=5, zorder=3)
    ax.axvline(A_WAKE, color="#d55e00", ls="--", lw=1.5,
               label=rf"Fixed $a_w = {A_WAKE}$")
    if has_sc:
        ax.axvline(sc_aw, color="#029e73", ls=":", lw=2,
                   label=rf"Self-consistent $a_w = {sc_aw:.3f}$")
    ax.set_xlabel(r"Wake convection induction $a_w$ [-]")
    ax.set_ylabel(r"$C_T$ [-]")
    ax.legend(fontsize=9); ax.grid(True)
    fig.tight_layout(); save_fig("Sens_AW_1_CT_vs_aw")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(aw_arr, cp_vals, "o-", color="#029e73", lw=2, ms=5, zorder=3)
    ax.axvline(A_WAKE, color="#d55e00", ls="--", lw=1.5,
               label=rf"Fixed $a_w = {A_WAKE}$")
    if has_sc:
        ax.axvline(sc_aw, color="#0173b2", ls=":", lw=2,
                   label=rf"Self-consistent $a_w = {sc_aw:.3f}$")
    ax.set_xlabel(r"Wake convection induction $a_w$ [-]")
    ax.set_ylabel(r"$C_P$ [-]")
    ax.legend(fontsize=9); ax.grid(True)
    fig.tight_layout(); save_fig("Sens_AW_2_CP_vs_aw")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(aw_arr, meana_arr, "o-", color="#0173b2", lw=2, ms=5,
            label=r"$\bar{a}$ from LL")
    ax.plot(aw_arr, aw_arr, "--", color="#949494", lw=1.5,
            label=r"$\bar{a} = a_w$  (self-consistent line)")
    if has_sc:
        ax.axvline(sc_aw, color="#029e73", ls=":", lw=2,
                   label=rf"Self-consistent $a_w = {sc_aw:.3f}$")
        ax.scatter([sc_aw], [sc_aw], color="#029e73", s=70, zorder=5)
    ax.set_xlabel(r"Input $a_w$ [-]"); ax.set_ylabel(r"$\bar{a}$ [-]")
    ax.legend(fontsize=9); ax.grid(True)
    fig.tight_layout(); save_fig("Sens_AW_3_mean_a_vs_aw")

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

    for qty_col, ylabel, fname, use_meana_label in [
            (0, r"$a$ [-]",
             "Sens_AW_5_axial_induction_vs_rR",   True),
            (5, r"$\Gamma$ [m$^2$/s]",
             "Sens_AW_6_circulation_vs_rR",        False),
            (3, r"$C_n = F_n\,/\,(\frac{1}{2}\rho U_\infty^2 R)$ [-]",
             "Sens_AW_7_normal_loading_vs_rR",     False),
            (7, r"$\phi$ [deg]",
             "Sens_AW_8_inflow_angle_vs_rR",       False)]:
        fig, ax = plt.subplots(figsize=(9, 5))
        x_list, y_list = [], []
        for idx, a_w in enumerate(SENS_AW_LIST):
            res, CT, CP = sens_aw_data[a_w]
            y = res[:, qty_col] / norm_val if qty_col == 3 else res[:, qty_col]
            if use_meana_label:
                lbl = rf"$a_w={a_w:.2f}$  $\bar{{a}}={meana_vals[idx]:.3f}$"
            else:
                lbl = rf"$a_w={a_w:.2f}$  $C_T={CT:.3f}$"
            ax.plot(res[:, 2], y, color=_sens_color(idx, n_aw), lw=1.8, label=lbl)
            x_list.append(res[:, 2])
            y_list.append(y)
        ax.set_xlabel("r/R"); ax.set_ylabel(ylabel)
        ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
        ax.grid(True)
        fig.tight_layout()
        # Direction and placement are data-driven per quantity:
        #   a (col 0):   UP   x_pos=0.26  (root, largest spread)
        #   Gamma (col 5): DOWN x_pos=0.39
        #   Cn (col 3):  DOWN x_pos=0.75  (mid-outboard)
        #   phi (col 7): DOWN x_pos=0.26
        _aw_dir  = 'up'   if qty_col == 0 else 'down'
        _aw_xpos = {0: 0.26, 5: 0.39, 3: 0.75, 7: 0.26}.get(qty_col, None)
        _annotate_increasing_direction(
            ax, x_list, y_list,
            label=r"increasing $a_w$",
            color="#555555",
            force_direction=_aw_dir,
            x_pos=_aw_xpos)
        save_fig(fname)

elif PLOT_SENS_AW:
    _skip("PLOT_SENS_AW", "convection speed sensitivity data missing")

# =============================================================================
# 19.  PLOTS — SENSITIVITY: discretisation
# =============================================================================

if PLOT_SENS_DISC and sens_disc_data:
    n_N = len(SENS_N_LIST)

    _DIST_STYLE = {
        "cosine":   {"color": "#0173b2", "ls": "-",  "marker": "o",
                     "label_prefix": "Cosine"},
        "constant": {"color": "#d55e00", "ls": "--", "marker": "s",
                     "label_prefix": "Constant"},
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, N in enumerate(SENS_N_LIST):
        if (N, "cosine") not in sens_disc_data: continue
        res, CT, CP = sens_disc_data[(N, "cosine")]
        ax.plot(res[:, 2], res[:, 3] / norm_val, color=_sens_color(idx, n_N), lw=2,
                label=rf"N={N}  $C_T={CT:.3f}$")
    ax.set_xlabel("r/R"); ax.set_ylabel(r"$C_n$ [-]")
    ax.legend(fontsize=8); ax.grid(True)
    fig.tight_layout(); save_fig("Sens_Disc_a1_Cn_panel_count_cosine")

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, N in enumerate(SENS_N_LIST):
        if (N, "cosine") not in sens_disc_data: continue
        res, CT, CP = sens_disc_data[(N, "cosine")]
        ax.plot(res[:, 2], res[:, 0], color=_sens_color(idx, n_N), lw=2,
                label=rf"N={N}  $C_P={CP:.3f}$")
    ax.set_xlabel("r/R"); ax.set_ylabel(r"$a$ [-]")
    ax.legend(fontsize=8); ax.grid(True)
    fig.tight_layout(); save_fig("Sens_Disc_a2_a_panel_count_cosine")

    N_ref = N_PANELS
    fig, ax = plt.subplots(figsize=(8, 5))
    for dist, st in _DIST_STYLE.items():
        if (N_ref, dist) not in sens_disc_data: continue
        res, CT, CP = sens_disc_data[(N_ref, dist)]
        ax.plot(res[:, 2], res[:, 3] / norm_val,
                color=st["color"], lw=2, ls=st["ls"],
                label=rf"{st['label_prefix']}  $C_T={CT:.3f}$")
    ax.set_xlabel("r/R"); ax.set_ylabel(r"$C_n$ [-]")
    ax.legend(fontsize=8); ax.grid(True)
    fig.tight_layout(); save_fig("Sens_Disc_b1_Cn_cosine_vs_constant")

    fig, ax = plt.subplots(figsize=(8, 5))
    for dist, st in _DIST_STYLE.items():
        if (N_ref, dist) not in sens_disc_data: continue
        res, CT, CP = sens_disc_data[(N_ref, dist)]
        ax.plot(res[:, 2], res[:, 0],
                color=st["color"], lw=2, ls=st["ls"],
                label=rf"{st['label_prefix']}  $C_P={CP:.3f}$")
    ax.set_xlabel("r/R"); ax.set_ylabel(r"$a$ [-]")
    ax.legend(fontsize=8); ax.grid(True)
    fig.tight_layout(); save_fig("Sens_Disc_b2_a_cosine_vs_constant")

    _CMP_N = [N for N in (5, 20, 70) if (N, "cosine") in sens_disc_data
              and (N, "constant") in sens_disc_data]

    if _CMP_N:
        _n_colors = [_CB_PALETTE[i] for i in range(len(_CMP_N))]

        for zoom, fname in [(False, "Sens_Disc_f1_Cn_cosine_vs_constant_multiN"),
                            (True,  "Sens_Disc_f2_Cn_cosine_vs_constant_multiN_zoom")]:
            fig, ax = plt.subplots(figsize=(9, 5.5))
            for ci, N in enumerate(_CMP_N):
                col = _n_colors[ci]
                res_cos, CT_cos, _ = sens_disc_data[(N, "cosine")]
                res_con, CT_con, _ = sens_disc_data[(N, "constant")]

                rR_cos = res_cos[:, 2];  Cn_cos = res_cos[:, 3] / norm_val
                rR_con = res_con[:, 2];  Cn_con = res_con[:, 3] / norm_val

                Cn_con_on_cos = np.interp(rR_cos, rR_con, Cn_con)
                ax.fill_between(rR_cos, Cn_cos, Cn_con_on_cos,
                                where=(Cn_con_on_cos > Cn_cos),
                                color=col, alpha=0.12, zorder=1,
                                interpolate=True)

                ax.plot(rR_cos, Cn_cos, color=col, lw=2.0, ls="-",
                        marker="o", ms=4, alpha=1.0, zorder=3,
                        label=rf"$N={N}$ cosine  $C_T={CT_cos:.3f}$")
                ax.plot(rR_con, Cn_con, color=col, lw=2.0, ls="--",
                        marker="s", ms=4, alpha=0.55, zorder=2,
                        label=rf"$N={N}$ constant  $C_T={CT_con:.3f}$")

            ax.set_xlabel("r/R")
            ax.set_ylabel(r"$C_n = F_n\,/\,(\frac{1}{2}\rho U_\infty^2 R)$ [-]")
            ax.grid(True)
            if zoom:
                ax.set_xlim(0.7, 1.0)
                res_fine, _, _ = sens_disc_data[(_CMP_N[-1], "cosine")]
                mask = res_fine[:, 2] >= 0.7
                y_tip = res_fine[mask, 3] / norm_val
                if y_tip.size:
                    ymin = 0.9 * float(np.min(y_tip))
                    ymax = 1.1 * float(np.max(y_tip))
                    ax.set_ylim(ymin, ymax)
            ax.legend(fontsize=8, loc="upper left")
            ax.text(0.02, 0.02,
                    "Shaded band: constant $>$ cosine\n(over-predicted loading)",
                    transform=ax.transAxes, fontsize=8, va="bottom", ha="left",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec="#949494", alpha=0.85))
            fig.tight_layout(); save_fig(fname)

    N_vals_cos = [N for N in SENS_N_LIST if (N, "cosine")   in sens_disc_data]
    N_vals_con = [N for N in SENS_N_LIST if (N, "constant") in sens_disc_data]
    N_vals_all = sorted(set(N_vals_cos) & set(N_vals_con))

    ct_cos = np.array([sens_disc_data[(N, "cosine")][1]   for N in N_vals_cos])
    cp_cos = np.array([sens_disc_data[(N, "cosine")][2]   for N in N_vals_cos])
    ct_con = np.array([sens_disc_data[(N, "constant")][1] for N in N_vals_con])
    cp_con = np.array([sens_disc_data[(N, "constant")][2] for N in N_vals_con])

    def _is_conv(N, dist):
        entry = _solver_meta.get(f"disc_{N}_{dist}")
        return bool(entry[1]) if entry is not None else True

    conv_cos = np.array([_is_conv(N, "cosine")   for N in N_vals_cos])
    conv_con = np.array([_is_conv(N, "constant") for N in N_vals_con])

    _conv_cos_N = [N for N, c in zip(N_vals_cos, conv_cos) if c]
    if _conv_cos_N:
        N_ref_conv = _conv_cos_N[-1]
        ct_ref = sens_disc_data[(N_ref_conv, "cosine")][1]
        cp_ref = sens_disc_data[(N_ref_conv, "cosine")][2]
    else:
        N_ref_conv = N_vals_cos[-1]
        ct_ref = ct_cos[-1];  cp_ref = cp_cos[-1]
    print(f"  Disc error reference = finest CONVERGED cosine case: "
          f"N={N_ref_conv}  CT={ct_ref:.4f}  CP={cp_ref:.4f}")

    def _pct_err(arr, ref):
        e = np.abs(arr - ref) / ref * 100.0
        return np.where(e == 0, np.nan, e)

    err_CT_cos = _pct_err(ct_cos, ct_ref)
    err_CP_cos = _pct_err(cp_cos, cp_ref)
    err_CT_con = _pct_err(ct_con, ct_ref)
    err_CP_con = _pct_err(cp_con, cp_ref)

    t_cos = np.array([timing_disc.get((N, "cosine"),   0.0) for N in N_vals_cos])
    t_con = np.array([timing_disc.get((N, "constant"), 0.0) for N in N_vals_con])
    t_max = max(t_cos.max(), t_con.max()) if max(t_cos.max(), t_con.max()) > 0 else 1.0
    t_hat_cos = t_cos / t_max
    t_hat_con = t_con / t_max
    has_timing = t_max > 0

    def _plot_conv_split(ax, x, y, conv, color, ls, marker, label,
                         logy=False, alpha=1.0):
        x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
        conv = np.asarray(conv, dtype=bool)
        plot_fn = ax.semilogy if logy else ax.plot
        plot_fn(x, y, ls=ls, color=color, lw=2, alpha=alpha, label=label, zorder=2)
        if conv.any():
            plot_fn(x[conv], y[conv], ls="none", marker=marker,
                    mfc=color, mec=color, ms=7, zorder=3)
        if (~conv).any():
            plot_fn(x[~conv], y[~conv], ls="none", marker=marker,
                    mfc="white", mec=color, mew=1.5, ms=7, zorder=3)
            plot_fn(x[~conv], y[~conv], ls="none", marker="x",
                    color=color, ms=6, mew=1.5, zorder=4)

    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_conv_split(ax, N_vals_cos, ct_cos, conv_cos,
                     _DIST_STYLE["cosine"]["color"],   "-",  "o", "Cosine")
    _plot_conv_split(ax, N_vals_con, ct_con, conv_con,
                     _DIST_STYLE["constant"]["color"], "--", "s", "Constant")
    ax.axvline(N_PANELS, color="#949494", ls=":", lw=1.2,
               label=rf"Baseline $N={N_PANELS}$")
    ax.plot([], [], ls="none", marker="x", color="#444",
            label="Not converged")
    ax.set_xlabel(r"$N_\mathrm{panels}$ [-]")
    ax.set_ylabel(r"$C_T$ [-]"); ax.legend(fontsize=9); ax.grid(True)
    fig.tight_layout(); save_fig("Sens_Disc_c1_CT_convergence_vs_N")

    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_conv_split(ax, N_vals_cos, cp_cos, conv_cos,
                     _DIST_STYLE["cosine"]["color"],   "-",  "o", "Cosine")
    _plot_conv_split(ax, N_vals_con, cp_con, conv_con,
                     _DIST_STYLE["constant"]["color"], "--", "s", "Constant")
    ax.axvline(N_PANELS, color="#949494", ls=":", lw=1.2,
               label=rf"Baseline $N={N_PANELS}$")
    ax.plot([], [], ls="none", marker="x", color="#444",
            label="Not converged")
    ax.set_xlabel(r"$N_\mathrm{panels}$ [-]")
    ax.set_ylabel(r"$C_P$ [-]"); ax.legend(fontsize=9); ax.grid(True)
    fig.tight_layout(); save_fig("Sens_Disc_c2_CP_convergence_vs_N")

    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_conv_split(ax, N_vals_cos, err_CT_cos, conv_cos,
                     _DIST_STYLE["cosine"]["color"],   "-",  "o",
                     r"$\epsilon_{C_T}$ cosine", logy=True)
    _plot_conv_split(ax, N_vals_cos, err_CP_cos, conv_cos,
                     _DIST_STYLE["cosine"]["color"],   "--", "o",
                     r"$\epsilon_{C_P}$ cosine", logy=True, alpha=0.6)
    _plot_conv_split(ax, N_vals_con, err_CT_con, conv_con,
                     _DIST_STYLE["constant"]["color"], "-",  "s",
                     r"$\epsilon_{C_T}$ constant", logy=True)
    _plot_conv_split(ax, N_vals_con, err_CP_con, conv_con,
                     _DIST_STYLE["constant"]["color"], "--", "s",
                     r"$\epsilon_{C_P}$ constant", logy=True, alpha=0.6)
    ax.axvline(N_PANELS, color="#949494", ls=":", lw=1.2,
               label=rf"Baseline $N={N_PANELS}$")
    ax.axvline(N_ref_conv, color="#444", ls="-.", lw=1.2,
               label=rf"Reference (finest converged) $N={N_ref_conv}$")
    ax.plot([], [], ls="none", marker="x", color="#444",
            label="Not converged")
    ax.set_xlabel(r"$N_\mathrm{panels}$ [-]")
    ax.set_ylabel(r"Relative error w.r.t. finest converged cosine [%]")
    ax.legend(fontsize=7); ax.grid(True, which="both")
    fig.tight_layout(); save_fig("Sens_Disc_d1_error_vs_N")

    if has_timing:
        fig, ax = plt.subplots(figsize=(8, 5))
        _plot_conv_split(ax, N_vals_cos, t_hat_cos, conv_cos,
                         _DIST_STYLE["cosine"]["color"],   "-",  "o", "Cosine")
        _plot_conv_split(ax, N_vals_con, t_hat_con, conv_con,
                         _DIST_STYLE["constant"]["color"], "--", "s", "Constant")
        ax.axvline(N_PANELS, color="#949494", ls=":", lw=1.2,
                   label=rf"Baseline $N={N_PANELS}$")
        ax.plot([], [], ls="none", marker="x", color="#444",
                label="Not converged")
        ax.set_xlabel(r"$N_\mathrm{panels}$ [-]")
        ax.set_ylabel(r"Normalised compute time $\hat{t}$ [-]")
        ax.set_ylim(0, 1.05); ax.legend(fontsize=9); ax.grid(True)
        fig.tight_layout(); save_fig("Sens_Disc_d2_time_vs_N")

    if has_timing:
        fig, ax = plt.subplots(figsize=(8, 5))
        for N_v, err_CT, t_hat, conv, dist in [
                (N_vals_cos, err_CT_cos, t_hat_cos, conv_cos, "cosine"),
                (N_vals_con, err_CT_con, t_hat_con, conv_con, "constant")]:
            st = _DIST_STYLE[dist]
            N_v = np.asarray(N_v)
            valid = ~np.isnan(err_CT)
            m_cv = valid & conv
            ax.scatter(t_hat[m_cv], err_CT[m_cv],
                       color=st["color"], marker=st["marker"],
                       s=80, zorder=3, edgecolors="#444", linewidths=0.5,
                       label=st["label_prefix"])
            m_nc = valid & (~conv)
            if m_nc.any():
                ax.scatter(t_hat[m_nc], err_CT[m_nc],
                           facecolors="white", edgecolors=st["color"],
                           marker=st["marker"], s=80, linewidths=1.5, zorder=3)
                ax.scatter(t_hat[m_nc], err_CT[m_nc],
                           color=st["color"], marker="x", s=50,
                           linewidths=1.5, zorder=4)
            for xi, yi, Ni in zip(t_hat[valid], err_CT[valid], N_v[valid]):
                ax.annotate(str(int(Ni)), (xi, yi),
                            textcoords="offset points", xytext=(5, 4),
                            fontsize=7, color="#333")
        for dist in ["cosine", "constant"]:
            N_v = N_vals_cos if dist == "cosine" else N_vals_con
            t_h = t_hat_cos  if dist == "cosine" else t_hat_con
            err = err_CT_cos if dist == "cosine" else err_CT_con
            st  = _DIST_STYLE[dist]
            if N_PANELS in N_v:
                idx_b = list(N_v).index(N_PANELS)
                ax.scatter([t_h[idx_b]], [err[idx_b]],
                           color=st["color"], marker="*",
                           s=200, zorder=5,
                           label=rf"Baseline {st['label_prefix']} $N={N_PANELS}$")
        ax.plot([], [], ls="none", marker="x", color="#444",
                label="Not converged")
        ax.set_xlabel(r"Normalised compute time $\hat{t}$ [-]")
        ax.set_ylabel(r"$\epsilon_{C_T}$ rel. to finest converged cosine [%]")
        ax.set_yscale("log"); ax.legend(fontsize=8); ax.grid(True, which="both")
        fig.tight_layout(); save_fig("Sens_Disc_e1_accuracy_vs_cost_CT")

    if has_timing:
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax2 = ax1.twinx()
        _plot_conv_split(ax1, N_vals_cos, err_CT_cos, conv_cos,
                         _DIST_STYLE["cosine"]["color"],   "-", "o",
                         r"$\epsilon_{C_T}$ cosine", logy=True)
        _plot_conv_split(ax1, N_vals_con, err_CT_con, conv_con,
                         _DIST_STYLE["constant"]["color"], "-", "s",
                         r"$\epsilon_{C_T}$ constant", logy=True)
        l3, = ax2.plot(N_vals_cos, t_hat_cos, "o--",
                       color=_DIST_STYLE["cosine"]["color"],   lw=1.5, alpha=0.5,
                       label=r"$\hat{t}$ cosine")
        l4, = ax2.plot(N_vals_con, t_hat_con, "s--",
                       color=_DIST_STYLE["constant"]["color"], lw=1.5, alpha=0.5,
                       label=r"$\hat{t}$ constant")
        ax1.axvline(N_PANELS, color="#949494", ls=":", lw=1.5,
                    label=rf"Baseline $N={N_PANELS}$")
        ax1.axvline(N_ref_conv, color="#444", ls="-.", lw=1.2,
                    label=rf"Reference $N={N_ref_conv}$")
        ax1.plot([], [], ls="none", marker="x", color="#444",
                 label="Not converged")
        ax1.set_xlabel(r"$N_\mathrm{panels}$ [-]")
        ax1.set_ylabel(r"$\epsilon_{C_T}$ relative error [%]")
        ax2.set_ylabel(r"Normalised compute time $\hat{t}$ [-]")
        ax2.set_ylim(0, 1.15)
        ax1.legend(fontsize=7, loc="upper right")
        ax1.grid(True, which="both")
        fig.tight_layout(); save_fig("Sens_Disc_e2_combined_error_time_vs_N")

elif PLOT_SENS_DISC:
    _skip("PLOT_SENS_DISC", "discretisation sensitivity data missing")

# =============================================================================
# 20.  PLOTS — SENSITIVITY: azimuthal step
# =============================================================================

if PLOT_SENS_DPSI and sens_dpsi_data:
    n_dpsi    = len(SENS_DPSI_LIST)
    dpsi_vals = SENS_DPSI_LIST
    ct_vals   = np.array([sens_dpsi_data[d][1] for d in dpsi_vals])
    cp_vals   = np.array([sens_dpsi_data[d][2] for d in dpsi_vals])

    for qty_col, ylabel, fname in [
            (0, r"$a$ [-]",
             "Sens_DPSI_a_axial_induction"),
            (3, r"$C_n = F_n\,/\,(\frac{1}{2}\rho U_\infty^2 R)$ [-]",
             "Sens_DPSI_b_normal_loading")]:
        fig, ax = plt.subplots(figsize=(8, 5))
        x_list, y_list = [], []
        for idx, dpsi in enumerate(SENS_DPSI_LIST):
            res, CT, CP = sens_dpsi_data[dpsi]
            y = res[:, qty_col] / norm_val if qty_col == 3 else res[:, qty_col]
            ax.plot(res[:, 2], y, color=_sens_color(idx, n_dpsi), lw=2,
                    label=rf"$\Delta\psi={dpsi}°$  $C_T={CT:.3f}$")
            x_list.append(res[:, 2])
            y_list.append(y)
        ax.set_xlabel("r/R"); ax.set_ylabel(ylabel)
        ax.legend(fontsize=8); ax.grid(True)
        fig.tight_layout()
        if fname == "Sens_DPSI_b_normal_loading":
            # y_pos=0.48: just below the flat mid-span bundle (r/R~0.34)
            # where all curves are nearly identical and clear space exists below
            _annotate_increasing_direction(
                ax, x_list, y_list,
                label=r"increasing $\Delta\psi$",
                color="#555555",
                force_direction='left',
                y_pos=0.48)
        save_fig(fname)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dpsi_vals, ct_vals, "o-", color="#0173b2", lw=2)
    ax.axvline(DPSI_DEG, color="#949494", ls="--", lw=1.2,
               label=rf"Baseline $\Delta\psi={DPSI_DEG}°$")
    ax.set_xlabel(r"$\Delta\psi$ [deg]")
    ax.set_ylabel(r"$C_T$ [-]"); ax.legend(fontsize=9); ax.grid(True)
    fig.tight_layout(); save_fig("Sens_DPSI_c1_CT_vs_dpsi")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dpsi_vals, cp_vals, "o-", color="#029e73", lw=2)
    ax.axvline(DPSI_DEG, color="#949494", ls="--", lw=1.2,
               label=rf"Baseline $\Delta\psi={DPSI_DEG}°$")
    ax.set_xlabel(r"$\Delta\psi$ [deg]")
    ax.set_ylabel(r"$C_P$ [-]"); ax.legend(fontsize=9); ax.grid(True)
    fig.tight_layout(); save_fig("Sens_DPSI_c2_CP_vs_dpsi")

    ct_ref_dpsi = ct_vals[0];  cp_ref_dpsi = cp_vals[0]
    err_CT_dpsi = np.abs(ct_vals - ct_ref_dpsi) / ct_ref_dpsi * 100.0
    err_CP_dpsi = np.abs(cp_vals - cp_ref_dpsi) / cp_ref_dpsi * 100.0
    err_CT_dpsi = np.where(err_CT_dpsi == 0, np.nan, err_CT_dpsi)
    err_CP_dpsi = np.where(err_CP_dpsi == 0, np.nan, err_CP_dpsi)

    t_dpsi     = np.array([timing_dpsi.get(d, 0.0) for d in dpsi_vals])
    t_max_dpsi = t_dpsi.max() if t_dpsi.max() > 0 else 1.0
    t_hat_dpsi = t_dpsi / t_max_dpsi

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(dpsi_vals, err_CT_dpsi, "o-", color="#0173b2", lw=2,
                label=r"$\epsilon_{C_T}$ [%]")
    ax.semilogy(dpsi_vals, err_CP_dpsi, "s-", color="#029e73", lw=2,
                label=r"$\epsilon_{C_P}$ [%]")
    ax.axvline(DPSI_DEG, color="#949494", ls="--", lw=1.2,
               label=rf"Baseline $\Delta\psi={DPSI_DEG}°$")
    ax.set_xlabel(r"$\Delta\psi$ [deg]")
    ax.set_ylabel(r"Relative error w.r.t. finest case [%]")
    ax.legend(fontsize=9); ax.grid(True, which="both")
    fig.tight_layout(); save_fig("Sens_DPSI_d1_error_vs_dpsi")

    if t_dpsi.max() > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(dpsi_vals, t_hat_dpsi, "o-", color="#de8f05", lw=2)
        ax.axvline(DPSI_DEG, color="#949494", ls="--", lw=1.2,
                   label=rf"Baseline $\Delta\psi={DPSI_DEG}°$")
        ax.set_xlabel(r"$\Delta\psi$ [deg]")
        ax.set_ylabel(r"Normalised compute time $\hat{t}$ [-]")
        ax.set_ylim(0, 1.05); ax.legend(fontsize=9); ax.grid(True)
        fig.tight_layout(); save_fig("Sens_DPSI_d2_time_vs_dpsi")

    if t_dpsi.max() > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        sc = ax.scatter(t_hat_dpsi, err_CT_dpsi,
                        c=dpsi_vals, cmap="plasma_r",
                        s=80, zorder=3,
                        edgecolors="#444", linewidths=0.5)
        for xi, yi, di in zip(t_hat_dpsi, err_CT_dpsi, dpsi_vals):
            if not np.isnan(yi):
                ax.annotate(f"{di:g}°", (xi, yi),
                            textcoords="offset points", xytext=(5, 4),
                            fontsize=8, color="#333")
        idx_base_d = dpsi_vals.index(DPSI_DEG) if DPSI_DEG in dpsi_vals else None
        if idx_base_d is not None:
            ax.scatter([t_hat_dpsi[idx_base_d]], [err_CT_dpsi[idx_base_d]],
                       s=140, color="#d55e00", zorder=5,
                       label=rf"Baseline $\Delta\psi={DPSI_DEG}°$")
            ax.legend(fontsize=9)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(r"$\Delta\psi$ [deg]", fontsize=10)
        ax.set_xlabel(r"Normalised compute time $\hat{t}$ [-]")
        ax.set_ylabel(r"$\epsilon_{C_T}$ relative to finest case [%]")
        ax.set_yscale("log"); ax.grid(True, which="both")
        fig.tight_layout(); save_fig("Sens_DPSI_e1_accuracy_vs_cost_CT")

    if t_dpsi.max() > 0:
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax2 = ax1.twinx()
        l1, = ax1.semilogy(dpsi_vals, err_CT_dpsi, "o-", color="#0173b2",
                           lw=2, label=r"$\epsilon_{C_T}$ [%]")
        l2, = ax1.semilogy(dpsi_vals, err_CP_dpsi, "s-", color="#029e73",
                           lw=2, label=r"$\epsilon_{C_P}$ [%]")
        l3, = ax2.plot(dpsi_vals, t_hat_dpsi, "^--", color="#de8f05",
                       lw=2, label=r"Normalised time $\hat{t}$")
        vl  = ax1.axvline(DPSI_DEG, color="#949494", ls=":", lw=1.5)
        ax1.set_xlabel(r"$\Delta\psi$ [deg]")
        ax1.set_ylabel(r"Relative error [%]")
        ax2.set_ylabel(r"Normalised compute time $\hat{t}$ [-]")
        ax2.set_ylim(0, 1.15)
        lines  = [l1, l2, l3, vl]
        labels = [r"$\epsilon_{C_T}$ [%]", r"$\epsilon_{C_P}$ [%]",
                  r"Norm. time $\hat{t}$",
                  rf"Baseline $\Delta\psi={DPSI_DEG}°$"]
        ax1.legend(lines, labels, fontsize=9, loc="upper left")
        ax1.grid(True, which="both")
        fig.tight_layout(); save_fig("Sens_DPSI_e2_combined_error_time_vs_dpsi")

elif PLOT_SENS_DPSI:
    _skip("PLOT_SENS_DPSI", "azimuthal step sensitivity data missing")

# =============================================================================
# 21.  PLOTS — SENSITIVITY: wake length
# =============================================================================

if PLOT_SENS_WAKE and sens_wake_data:
    n_nw    = len(SENS_NWAKE_LIST)
    nw_vals = SENS_NWAKE_LIST
    ct_vals = [sens_wake_data[nw][1] for nw in nw_vals]
    cp_vals = [sens_wake_data[nw][2] for nw in nw_vals]
    ct_arr  = np.array(ct_vals)
    cp_arr  = np.array(cp_vals)

    # Convergence threshold annotation (same as original)
    _ct_ref = ct_vals[-1]
    _cp_ref = cp_vals[-1]
    _conv_tol = 0.005
    _nw_conv_CT = next((nw for nw, ct in zip(nw_vals, ct_vals)
                        if abs(ct - _ct_ref) < _conv_tol), nw_vals[-1])
    _nw_conv_CP = next((nw for nw, cp in zip(nw_vals, cp_vals)
                        if abs(cp - _cp_ref) < _conv_tol), nw_vals[-1])

    # ── Spanwise distributions ────────────────────────────────────────────
    for qty_col, ylabel, fname in [
            (0, r"$a$ [-]",
             "Sens_Wake_a_axial_induction"),
            (3, r"$C_n = F_n\,/\,(\frac{1}{2}\rho U_\infty^2 R)$ [-]",
             "Sens_Wake_b_normal_loading")]:
        fig, ax = plt.subplots(figsize=(8, 5))
        x_list, y_list = [], []
        for idx, nw in enumerate(SENS_NWAKE_LIST):
            res, CT, CP = sens_wake_data[nw]
            y = res[:, qty_col] / norm_val if qty_col == 3 else res[:, qty_col]
            ax.plot(res[:, 2], y, color=_sens_color(idx, n_nw), lw=2,
                    label=rf"$N_{{wake}}={nw}$  $C_T={CT:.3f}$  $C_P={CP:.3f}$")
            x_list.append(res[:, 2])
            y_list.append(y)
        ax.text(0.97, 0.97,
                rf"$C_T$ converged at $N_{{wake}}={_nw_conv_CT}$" + "\n"
                + rf"$C_P$ converged at $N_{{wake}}={_nw_conv_CP}$",
                transform=ax.transAxes, fontsize=8,
                va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#949494",
                          alpha=0.85))
        ax.set_xlabel("r/R"); ax.set_ylabel(ylabel)
        ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
        ax.grid(True)
        fig.tight_layout()
        if fname == "Sens_Wake_a_axial_induction":
            # x_pos=0.51: clean mid-span position, avoids root spike and
            # the convergence text box sitting at top-right
            _annotate_increasing_direction(
                ax, x_list, y_list,
                label=r"increasing $N_{wake}$",
                color="#555555",
                force_direction='up',
                x_pos=0.51)
        save_fig(fname)

    # ── CT convergence vs N_wake ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(nw_vals, ct_vals, "o-", color="#0173b2", lw=2)
    ax.axvline(_nw_conv_CT, color="#949494", ls="--", lw=1.2,
               label=rf"Converged at $N_{{wake}}={_nw_conv_CT}$")
    ax.axhline(_ct_ref, color="#949494", ls=":", lw=1.0)
    ax.set_xlabel(r"Wake length $N_{wake}$ [rotations]")
    ax.set_ylabel(r"$C_T$ [-]"); ax.legend(fontsize=9); ax.grid(True)
    fig.tight_layout(); save_fig("Sens_Wake_c1_CT_convergence_vs_Nwake")

    # ── CP convergence vs N_wake ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(nw_vals, cp_vals, "o-", color="#029e73", lw=2)
    ax.axvline(_nw_conv_CP, color="#949494", ls="--", lw=1.2,
               label=rf"Converged at $N_{{wake}}={_nw_conv_CP}$")
    ax.axhline(_cp_ref, color="#949494", ls=":", lw=1.0)
    ax.set_xlabel(r"Wake length $N_{wake}$ [rotations]")
    ax.set_ylabel(r"$C_P$ [-]"); ax.legend(fontsize=9); ax.grid(True)
    fig.tight_layout(); save_fig("Sens_Wake_c2_CP_convergence_vs_Nwake")

    # =========================================================================
    # NEW: Relative error, normalised time, accuracy-vs-cost, combined
    # Reference = finest N_wake (longest wake = most accurate)
    # =========================================================================

    # ── Relative error arrays ─────────────────────────────────────────────
    ct_ref_wake = ct_arr[-1]
    cp_ref_wake = cp_arr[-1]
    err_CT_wake = np.abs(ct_arr - ct_ref_wake) / abs(ct_ref_wake) * 100.0
    err_CP_wake = np.abs(cp_arr - cp_ref_wake) / abs(cp_ref_wake) * 100.0
    # Zero out the reference point (suppress in log plot)
    err_CT_wake = np.where(err_CT_wake == 0.0, np.nan, err_CT_wake)
    err_CP_wake = np.where(err_CP_wake == 0.0, np.nan, err_CP_wake)

    # ── Timing arrays ─────────────────────────────────────────────────────
    t_wake     = np.array([timing_wake.get(nw, 0.0) for nw in nw_vals])
    t_max_wake = t_wake.max() if t_wake.max() > 0 else 1.0
    t_hat_wake = t_wake / t_max_wake
    has_timing_wake = t_max_wake > 0

    # ── d1: Relative error vs N_wake (log scale) ──────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(nw_vals, err_CT_wake, "o-", color="#0173b2", lw=2,
                label=r"$\epsilon_{C_T}$ [%]")
    ax.semilogy(nw_vals, err_CP_wake, "s-", color="#029e73", lw=2,
                label=r"$\epsilon_{C_P}$ [%]")
    ax.axvline(N_WAKE, color="#949494", ls="--", lw=1.2,
               label=rf"Baseline $N_{{wake}}={N_WAKE}$")
    ax.axvline(nw_vals[-1], color="#444", ls="-.", lw=1.2,
               label=rf"Reference $N_{{wake}}={nw_vals[-1]}$")
    ax.set_xlabel(r"Wake length $N_{wake}$ [rotations]")
    ax.set_ylabel(r"Relative error w.r.t. longest wake [%]")
    ax.legend(fontsize=9); ax.grid(True, which="both")
    fig.tight_layout(); save_fig("Sens_Wake_d1_error_vs_Nwake")

    # ── d2: Normalised compute time vs N_wake ─────────────────────────────
    if has_timing_wake:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(nw_vals, t_hat_wake, "o-", color="#de8f05", lw=2)
        ax.axvline(N_WAKE, color="#949494", ls="--", lw=1.2,
                   label=rf"Baseline $N_{{wake}}={N_WAKE}$")
        ax.set_xlabel(r"Wake length $N_{wake}$ [rotations]")
        ax.set_ylabel(r"Normalised compute time $\hat{t}$ [-]")
        ax.set_ylim(0, 1.05); ax.legend(fontsize=9); ax.grid(True)
        fig.tight_layout(); save_fig("Sens_Wake_d2_time_vs_Nwake")

    # ── e1: Accuracy vs cost scatter (CT error) ────────────────────────────
    if has_timing_wake:
        fig, ax = plt.subplots(figsize=(8, 5))
        sc = ax.scatter(t_hat_wake, err_CT_wake,
                        c=nw_vals, cmap="viridis",
                        s=80, zorder=3,
                        edgecolors="#444", linewidths=0.5)
        for xi, yi, nwi in zip(t_hat_wake, err_CT_wake, nw_vals):
            if not np.isnan(yi):
                ax.annotate(str(int(nwi)), (xi, yi),
                            textcoords="offset points", xytext=(5, 4),
                            fontsize=8, color="#333")
        # Highlight baseline
        if N_WAKE in nw_vals:
            idx_base_nw = nw_vals.index(N_WAKE)
            ax.scatter([t_hat_wake[idx_base_nw]], [err_CT_wake[idx_base_nw]],
                       s=180, color="#d55e00", zorder=5,
                       label=rf"Baseline $N_{{wake}}={N_WAKE}$")
            ax.legend(fontsize=9)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(r"$N_{wake}$ [rotations]", fontsize=10)
        ax.set_xlabel(r"Normalised compute time $\hat{t}$ [-]")
        ax.set_ylabel(r"$\epsilon_{C_T}$ relative to longest wake [%]")
        ax.set_yscale("log"); ax.grid(True, which="both")
        fig.tight_layout(); save_fig("Sens_Wake_e1_accuracy_vs_cost_CT")

    # ── e2: Combined dual-axis: error + time vs N_wake ────────────────────
    if has_timing_wake:
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax2 = ax1.twinx()
        l1, = ax1.semilogy(nw_vals, err_CT_wake, "o-", color="#0173b2",
                           lw=2, label=r"$\epsilon_{C_T}$ [%]")
        l2, = ax1.semilogy(nw_vals, err_CP_wake, "s-", color="#029e73",
                           lw=2, label=r"$\epsilon_{C_P}$ [%]")
        l3, = ax2.plot(nw_vals, t_hat_wake, "^--", color="#de8f05",
                       lw=2, label=r"Normalised time $\hat{t}$")
        vl  = ax1.axvline(N_WAKE, color="#949494", ls=":", lw=1.5)
        ax1.set_xlabel(r"Wake length $N_{wake}$ [rotations]")
        ax1.set_ylabel(r"Relative error [%]")
        ax2.set_ylabel(r"Normalised compute time $\hat{t}$ [-]")
        ax2.set_ylim(0, 1.15)
        lines  = [l1, l2, l3, vl]
        labels = [r"$\epsilon_{C_T}$ [%]", r"$\epsilon_{C_P}$ [%]",
                  r"Norm. time $\hat{t}$",
                  rf"Baseline $N_{{wake}}={N_WAKE}$"]
        ax1.legend(lines, labels, fontsize=9, loc="upper right")
        ax1.grid(True, which="both")
        fig.tight_layout(); save_fig("Sens_Wake_e2_combined_error_time_vs_Nwake")

elif PLOT_SENS_WAKE:
    _skip("PLOT_SENS_WAKE", "wake length sensitivity data missing")

# =============================================================================
# 22.  SAVE RESULTS TO NPZ
# =============================================================================

def save_ll_results(path=LL_RESULTS_PATH):
    kw = dict(
        polar_alpha=polar_alpha, polar_cl=polar_cl, polar_cd=polar_cd,
        cfg_Radius=Radius, cfg_NBlades=NBlades, cfg_U0=U0, cfg_rho=rho,
        cfg_RootLocation_R=RootLocation_R, cfg_TipLocation_R=TipLocation_R,
        cfg_Pitch=Pitch, cfg_N_PANELS=N_PANELS, cfg_N_WAKE=N_WAKE,
        cfg_DPSI_DEG=DPSI_DEG, cfg_A_WAKE=A_WAKE,
        cfg_USE_ITERATED_AW=np.array(USE_ITERATED_AW),
        sweep_tsrs=np.array(TSR_SWEEP_SPAN, dtype=float),
        tsr_CT=tsr_CT_span if len(tsr_CT_span) > 0 else np.array([]),
        tsr_CP=tsr_CP_span if len(tsr_CP_span) > 0 else np.array([]),
        perf_tsrs=np.array(TSR_SWEEP_PERF, dtype=float),
        perf_CT=tsr_CT_perf if len(tsr_CT_perf) > 0 else np.array([]),
        perf_CP=tsr_CP_perf if len(tsr_CP_perf) > 0 else np.array([]),
        sens_aw_vals   =np.array(SENS_AW_LIST,    dtype=float),
        sens_aw_CT     =np.array([sens_aw_data[aw][1]  for aw in SENS_AW_LIST   if aw  in sens_aw_data],   dtype=float),
        sens_aw_CP     =np.array([sens_aw_data[aw][2]  for aw in SENS_AW_LIST   if aw  in sens_aw_data],   dtype=float),
        sens_dpsi_vals =np.array(SENS_DPSI_LIST,  dtype=float),
        sens_dpsi_CT   =np.array([sens_dpsi_data[d][1] for d  in SENS_DPSI_LIST if d   in sens_dpsi_data], dtype=float),
        sens_dpsi_CP   =np.array([sens_dpsi_data[d][2] for d  in SENS_DPSI_LIST if d   in sens_dpsi_data], dtype=float),
        sens_nwake_vals=np.array(SENS_NWAKE_LIST, dtype=float),
        sens_nwake_CT  =np.array([sens_wake_data[nw][1] for nw in SENS_NWAKE_LIST if nw in sens_wake_data], dtype=float),
        sens_nwake_CP  =np.array([sens_wake_data[nw][2] for nw in SENS_NWAKE_LIST if nw in sens_wake_data], dtype=float),
        sens_N_vals    =np.array(SENS_N_LIST,     dtype=float),
        sens_N_CT_cosine=np.array([sens_disc_data[(N, "cosine")][1] for N in SENS_N_LIST if (N, "cosine") in sens_disc_data], dtype=float),
        sens_N_CP_cosine=np.array([sens_disc_data[(N, "cosine")][2] for N in SENS_N_LIST if (N, "cosine") in sens_disc_data], dtype=float),
    )
    for TSR in TSR_SWEEP_SPAN:
        if TSR in sweep_data_span:
            kw[f"sweep_res_{int(TSR)}"] = sweep_data_span[TSR]
    for aw in SENS_AW_LIST:
        if aw in sens_aw_data:
            kw[f"sens_aw_res_{aw}"] = sens_aw_data[aw][0]
    for d in SENS_DPSI_LIST:
        if d in sens_dpsi_data:
            kw[f"sens_dpsi_res_{d}"] = sens_dpsi_data[d][0]
    for nw in SENS_NWAKE_LIST:
        if nw in sens_wake_data:
            kw[f"sens_nwake_res_{nw}"] = sens_wake_data[nw][0]
    for N in SENS_N_LIST:
        for dist in ["cosine", "constant"]:
            if (N, dist) in sens_disc_data:
                kw[f"sens_disc_res_{N}_{dist}"] = sens_disc_data[(N, dist)][0]

    np.savez(path, **kw)
    sz = sum(v.nbytes for v in kw.values() if hasattr(v, "nbytes"))
    print(f"\nLL results saved  ->  {path}  ({len(kw)} arrays, {sz//1024} KB)")


if SAVE_LL_RESULTS:
    save_ll_results()
else:
    print("\nSAVE_LL_RESULTS=False — ll_results.npz not written.")

# =============================================================================
# 23.  SAVE PDF TABLES
# =============================================================================

if SAVE_TABLES_PDF:
    save_all_tables()
else:
    print("\nSAVE_TABLES_PDF=False — no tables written.")

print("\nDone!")
print("  Plots  →", save_folder)
print("  Tables →", tables_folder)