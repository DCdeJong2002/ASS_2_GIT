"""
LiftingLine_FINAL.py  —  AE4135 Rotor/Wake Aerodynamics, Assignment 2
Frozen Vortex Wake / Lifting Line model for the DU95W180 wind turbine rotor.

Authors: Douwe de Jong (5313899), Martijn van Leeuwen (5614422)
================================================================
Self-contained script producing all required plots and saving
results to ll_results.npz for use with PLOTTING_LL_FINAL.py.

Run:  python LiftingLine_FINAL.py
"""

import os, sys
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

# ── Rotor geometry (identical to BEM Assignment) ─────────────────────────────
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

# ── TSR sweeps ────────────────────────────────────────────────────────────────
TSR_SWEEP_SPAN = [6, 8, 10]   # used for all spanwise plots

# ── Sensitivity sweep parameters (all run at TSR=8) ──────────────────────────
SENS_TSR          = 8
SENS_N_LIST       = [10, 20, 30, 50]          # panel count sensitivity
SENS_AW_LIST      = [0.0, 0.1, 0.25, 0.4]    # convection speed sensitivity
SENS_DPSI_LIST    = [5.0, 10.0, 15.0, 20.0]  # azimuthal step sensitivity
SENS_NWAKE_LIST   = [1, 2, 3, 5, 8]          # wake-length sensitivity (rotations)

# ── Display / save ─────────────────────────────────────────────────────────────
SHOW_PLOTS = False  # False -> save & close immediately; True -> plt.show() after save

# ── Computations (toggle off to skip expensive runs) ─────────────────────────
RUN_TSR_SWEEP_SPAN   = True
RUN_SENS_CONV_SPEED  = True   # sensitivity: convection speed (a_w)
RUN_SENS_DISC        = True   # sensitivity: blade discretization (N, cosine vs constant)
RUN_SENS_AZIMUTHAL   = True   # sensitivity: azimuthal step (dpsi)
RUN_SENS_WAKE_LENGTH = True   # sensitivity: wake length (N_wake)

# ── Plots ──────────────────────────────────────────────────────────────────────
PLOT_LL_1  = True   # inflow angle and AoA vs r/R           (requires RUN_TSR_SWEEP_SPAN)
PLOT_LL_2  = True   # axial and tangential induction vs r/R (requires RUN_TSR_SWEEP_SPAN)
PLOT_LL_3  = True   # axial and azimuthal loading vs r/R    (requires RUN_TSR_SWEEP_SPAN)
PLOT_LL_4  = True   # circulation vs r/R                    (requires RUN_TSR_SWEEP_SPAN)
PLOT_LL_5  = True   # CT and CP scalars per TSR             (requires RUN_TSR_SWEEP_SPAN)
PLOT_SENS_AW   = True   # sensitivity: convection speed     (requires RUN_SENS_CONV_SPEED)
PLOT_SENS_DISC = True   # sensitivity: discretization       (requires RUN_SENS_DISC)
PLOT_SENS_DPSI = True   # sensitivity: azimuthal step       (requires RUN_SENS_AZIMUTHAL)
PLOT_SENS_WAKE = True   # sensitivity: wake length          (requires RUN_SENS_WAKE_LENGTH)

# ── Save ──────────────────────────────────────────────────────────────────────
SAVE_LL_RESULTS = True  # write ll_results.npz
SAVE_TABLES_PDF = True  # write summary tables to LLM_tables/

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

# ── Matplotlib / font settings ────────────────────────────────────────────────
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

# ── Seaborn colorblind palette ────────────────────────────────────────────────
_CB_PALETTE = sns.color_palette("colorblind").as_hex()

def _tsr_color(idx, n):
    """Color for TSR sweep lines — cycles through colorblind palette."""
    return _CB_PALETTE[idx % len(_CB_PALETTE)]

def _sens_color(idx, n):
    """Color for sensitivity study lines — cycles through colorblind palette."""
    return _CB_PALETTE[idx % len(_CB_PALETTE)]

# =============================================================================
# 3.  BLADE GEOMETRY
# =============================================================================

def blade_chord(r_R):
    """Chord [m] as a function of r/R."""
    return 3.0 * (1.0 - r_R) + 1.0

def blade_twist(r_R):
    """Local pitch angle [deg] = twist + global pitch."""
    return 14.0 * (1.0 - r_R) + Pitch

def make_panels(N=N_PANELS, distribution=DISTRIBUTION):
    """
    Return (r_edges, r_centers, dr) for N spanwise panels.

    Parameters
    ----------
    N            : number of panels
    distribution : "cosine" or "constant"
    """
    r_root = RootLocation_R * Radius
    r_tip  = TipLocation_R  * Radius
    if distribution == "cosine":
        theta = np.linspace(0.0, np.pi, N + 1)
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
    """
    Build the full list of control points and vortex filament rings
    for all blades.

    Convention (x, y, z):
      x  — axial (downstream)
      y  — in-plane horizontal
      z  — in-plane vertical
    Blade 0 starts at angle 0 in the y-z plane.

    Each ring = one spanwise panel on one blade.
    A ring contains:
      [0]         bound filament from r_inner to r_outer (at c/4 line)
      [1..Nw]     trailing filament at r_inner (inward, shed backward)
      [Nw+1..2Nw] trailing filament at r_outer (outward, shed backward)

    Returns
    -------
    controlpoints : list of dicts, one per panel x blade
    rings         : list of filament lists, one per panel x blade
    """
    r_edges, r_centers, dr = make_panels(N, distribution)
    U_wake  = U0 * (1.0 - a_w)
    dpsi    = np.radians(dpsi_deg)
    # azimuthal angles of wake nodes (0 … N_wake × 2π)
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

            # ── Control point at the lifting line (rotor plane, no chord offset) ──
            cp_y = r * cosR
            cp_z = r * sinR
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

            # ── Bound vortex filament (from inner to outer edge at x=0) ──────
            y_in  = r_edges[i]   * cosR;  z_in  = r_edges[i]   * sinR
            y_out = r_edges[i+1] * cosR;  z_out = r_edges[i+1] * sinR
            filaments.append({
                'x1': 0.0, 'y1': y_in,  'z1': z_in,
                'x2': 0.0, 'y2': y_out, 'z2': z_out,
            })

            # ── Trailing vortex at inner edge (r_edges[i]) ───────────────────
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

            # ── Trailing vortex at outer edge (r_edges[i+1]) — opposite sign ─
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
    """
    Induced velocity at point xp due to a unit-strength vortex filament
    from x1 to x2.
    """
    R1    = xp - x1
    R2    = xp - x2
    cross = np.cross(R1, R2)
    cross_sq = np.einsum('ij,ij->i', cross, cross)

    R1_mag = np.linalg.norm(R1, axis=1)
    R2_mag = np.linalg.norm(R2, axis=1)

    inside = (cross_sq < r_core**2) | (R1_mag < r_core) | (R2_mag < r_core)

    R0R1 = np.einsum('j,ij->i', x2 - x1, R1)
    R0R2 = np.einsum('j,ij->i', x2 - x1, R2)

    R1_mag  = np.where(R1_mag  < 1e-12, 1e-12, R1_mag)
    R2_mag  = np.where(R2_mag  < 1e-12, 1e-12, R2_mag)
    cross_sq = np.where(cross_sq < 1e-24, 1e-24, cross_sq)

    K = (1.0 / (4.0 * np.pi * cross_sq)) * (R0R1 / R1_mag - R0R2 / R2_mag)
    K = np.where(inside, 0.0, K)

    return K[:, None] * cross

# =============================================================================
# 6.  INFLUENCE MATRIX ASSEMBLY
# =============================================================================

def assemble_influence_matrix(controlpoints, rings):
    """
    Assemble the 3 x (N_total x N_total) influence matrices.
    """
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
# 7.  CIRCULATION SOLVER  (Aitken dynamic relaxation)
# =============================================================================

def solve_circulation(A_u, A_v, A_w, controlpoints, Omega,
                      tol=1e-4, max_iter=500):
    """
    Iteratively solve for the bound circulation Gamma on each panel.
    Uses Kutta-Joukowski closure with Aitken dynamic relaxation.

    Returns
    -------
    Gamma   : (N_total,) — converged bound circulation per panel
    u_ind   : (N_total,) — induced axial velocity  at CPs
    v_ind   : (N_total,) — induced y-velocity       at CPs
    w_ind   : (N_total,) — induced z-velocity       at CPs
    n_iter  : int        — number of iterations taken
    converged : bool     — whether convergence criterion was met
    """
    N_total = len(controlpoints)
    Gamma   = np.zeros(N_total)
    R_prev  = np.zeros(N_total)
    omega   = 0.1
    err     = np.inf

    u_ind = v_ind = w_ind = np.zeros(N_total)

    for k in range(max_iter):
        u_ind = A_u @ Gamma
        v_ind = A_v @ Gamma
        w_ind = A_w @ Gamma

        Gamma_new = np.zeros(N_total)

        for i, cp in enumerate(controlpoints):
            r_cp = cp['coords']
            v_rot = np.cross(np.array([-Omega, 0.0, 0.0]), r_cp)

            V_ax  = U0 + u_ind[i] + v_rot[0]
            V_y   = v_ind[i] + v_rot[1]
            V_z   = w_ind[i] + v_rot[2]

            r_mag   = max(cp['r'], 1e-12)
            azim_dir = np.cross(np.array([-1.0 / r_mag, 0.0, 0.0]), r_cp)
            V_tan    = float(np.dot(azim_dir, [V_ax, V_y, V_z]))

            V_eff = np.sqrt(V_ax**2 + V_tan**2)
            phi   = np.arctan2(V_ax, V_tan)
            alpha = phi - cp['twist_rad']

            cl = float(np.interp(np.degrees(alpha), polar_alpha, polar_cl))
            Gamma_new[i] = 0.5 * cp['chord'] * V_eff * cl

        # ── Aitken relaxation ─────────────────────────────────────────────────
        R_k  = Gamma_new - Gamma
        err  = np.max(np.abs(R_k)) / max(np.max(np.abs(Gamma_new)), 1e-6)

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
    """
    Compute aerodynamic quantities for blade 0 panels only.

    Result array columns:
      0  a          axial induction factor
      1  a'         tangential induction factor
      2  r/R        non-dimensional span location
      3  F_n        normal (axial) force per unit span [N/m]
      4  F_t        tangential (azimuthal) force per unit span [N/m]
      5  Gamma      bound circulation [m²/s]
      6  alpha      angle of attack [deg]
      7  phi        inflow angle [deg]
      8  Cl         lift coefficient
      9  Cd         drag coefficient
    """
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

        lift  = 0.5 * rho * V_eff**2 * cp['chord'] * cl
        drag  = 0.5 * rho * V_eff**2 * cp['chord'] * cd
        F_n   = lift * np.cos(phi) + drag * np.sin(phi)
        F_t   = lift * np.sin(phi) - drag * np.cos(phi)

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

    return res, CT, CP

# =============================================================================
# 9.  ROTOR EVALUATOR  (one-stop function)
# =============================================================================

# Store solver metadata (iterations, convergence) keyed by run label
_solver_meta = {}   # label -> (n_iter, converged)

def run_case(TSR, N=N_PANELS, N_wake=N_WAKE, dpsi_deg=DPSI_DEG,
             a_w=A_WAKE, distribution=DISTRIBUTION, verbose=True,
             _meta_key=None):
    """
    Run a full Lifting Line solve for a single operating point.

    Parameters
    ----------
    TSR          : tip-speed ratio
    N            : number of spanwise panels per blade
    N_wake       : number of wake rotations
    dpsi_deg     : azimuthal step [deg]
    a_w          : frozen-wake axial induction (convection factor)
    distribution : "cosine" or "constant"
    verbose      : print progress
    _meta_key    : key under which to store (n_iter, converged) in _solver_meta

    Returns
    -------
    res : (N, 10) result array
    CT  : float
    CP  : float
    """
    Omega = U0 * TSR / Radius
    if verbose:
        print(f"  Building vortex system  (N={N}, N_wake={N_wake}, "
              f"dpsi={dpsi_deg}°, a_w={a_w}, dist={distribution}) ...")
    cps, rings = build_vortex_system(
        Omega, N=N, N_wake=N_wake, dpsi_deg=dpsi_deg,
        a_w=a_w, distribution=distribution)

    if verbose:
        print(f"  Assembling influence matrix ({len(cps)}×{len(cps)}) ...")
    A_u, A_v, A_w = assemble_influence_matrix(cps, rings)

    if verbose:
        print(f"  Solving circulation ...")
    Gamma, u_ind, v_ind, w_ind, n_iter, converged = solve_circulation(
        A_u, A_v, A_w, cps, Omega)

    if _meta_key is not None:
        _solver_meta[_meta_key] = (n_iter, converged)

    res, CT, CP = post_process(Gamma, u_ind, v_ind, w_ind, cps, Omega)
    if verbose:
        print(f"  TSR={TSR}  CT={CT:.4f}  CP={CP:.4f}")
    return res, CT, CP

# =============================================================================
# 10.  SAVE / SHOW HELPER  (identical structure to BEM)
# =============================================================================

LL_RESULTS_PATH = "LLM_results.npz"

save_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "LLM_plots")
os.makedirs(save_folder, exist_ok=True)

tables_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "LLM_tables")
os.makedirs(tables_folder, exist_ok=True)

def save_fig(name):
    stem = name.rsplit(".", 1)[0] if "." in name else name
    fpath = os.path.join(save_folder, stem + ".pdf")
    plt.savefig(fpath)
    print(f"  Saved: {stem}.pdf")
    if SHOW_PLOTS: plt.show()
    else:          plt.close()

def _skip(name, reason):
    print(f"  [SKIP] {name} — {reason}")
    plt.close("all")

norm_val = 0.5 * rho * U0**2 * Radius   # non-dimensionalisation for loading

# =============================================================================
# 10b.  PDF TABLE HELPER
# =============================================================================

def _make_table_pdf(filename, title, headers, rows, col_widths=None,
                    footnote=None):
    """
    Save a single table as a PDF page in LLM_tables/.

    Parameters
    ----------
    filename   : output filename (without folder, without extension)
    title      : table title string (shown above the table)
    headers    : list of column header strings
    rows       : list of lists — each inner list is one table row (strings)
    col_widths : optional list of column widths in points; auto-sized if None
    footnote   : optional string printed below the table
    """
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle,
                                     Paragraph, Spacer)
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER

    fpath = os.path.join(tables_folder, filename + ".pdf")

    # Choose orientation based on number of columns
    pagesize = landscape(A4) if len(headers) > 6 else A4
    doc = SimpleDocTemplate(
        fpath, pagesize=pagesize,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TableTitle", parent=styles["Heading2"],
        alignment=TA_CENTER, spaceAfter=6,
    )
    foot_style = ParagraphStyle(
        "Footnote", parent=styles["Normal"],
        fontSize=8, textColor=colors.grey, spaceBefore=4,
    )

    story = []
    story.append(Paragraph(title, title_style))
    story.append(Spacer(1, 0.3*cm))

    # Build table data: header row + data rows
    table_data = [headers] + rows

    # Auto column widths if not provided
    page_w = pagesize[0] - 4*cm
    if col_widths is None:
        col_widths = [page_w / len(headers)] * len(headers)

    tbl = Table(table_data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle([
        # Header row
        ("BACKGROUND",   (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, 0), 10),
        ("ALIGN",        (0, 0), (-1, 0), "CENTER"),
        ("BOTTOMPADDING",(0, 0), (-1, 0), 8),
        ("TOPPADDING",   (0, 0), (-1, 0), 8),
        # Data rows
        ("FONTNAME",     (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",     (0, 1), (-1, -1), 9),
        ("ALIGN",        (0, 1), (-1, -1), "CENTER"),
        ("TOPPADDING",   (0, 1), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 1), (-1, -1), 5),
        # Alternating row shading
        *[("BACKGROUND", (0, i), (-1, i), colors.HexColor("#ecf0f1"))
          for i in range(2, len(table_data), 2)],
        # Grid
        ("GRID",         (0, 0), (-1, -1), 0.5, colors.HexColor("#bdc3c7")),
        ("BOX",          (0, 0), (-1, -1), 1.0, colors.HexColor("#2c3e50")),
        # Vertical alignment
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
    ]))

    story.append(tbl)

    if footnote:
        story.append(Spacer(1, 0.3*cm))
        story.append(Paragraph(footnote, foot_style))

    doc.build(story)
    print(f"  Saved table: {filename}.pdf")


def save_all_tables():
    """
    Build and save PDF summary tables for all completed sensitivity runs.
    Tables are written to LLM_tables/.
    """
    print("\n" + "="*60)
    print("Saving summary tables to LLM_tables/ ...")

    # ── Table 1 : TSR sweep summary ──────────────────────────────────────────
    # Columns: lambda | CT | CP | iters | conv.
    if sweep_data_span:
        headers = [u"\u03bb", "CT", "CP", "iters", "conv."]
        rows = []
        for TSR in TSR_SWEEP_SPAN:
            if TSR not in sweep_data_span:
                continue
            mk  = f"tsr_{TSR}"
            ni, cv = _solver_meta.get(mk, ("—", True))
            CT  = tsr_CT_span[TSR_SWEEP_SPAN.index(TSR)]
            CP  = tsr_CP_span[TSR_SWEEP_SPAN.index(TSR)]
            rows.append([
                str(TSR),
                f"{CT:.4f}",
                f"{CP:.4f}",
                str(ni),
                u"\u2713" if cv else u"\u2717",
            ])
        _make_table_pdf(
            "Table_1_TSR_sweep",
            "Table 1 — TSR Sweep Results (Lifting Line)",
            headers, rows,
            col_widths=[60, 80, 80, 60, 60],
            footnote=(
                f"Baseline: N={N_PANELS} panels (cosine), N_wake={N_WAKE} rotations, "
                f"dpsi={DPSI_DEG} deg, a_w={A_WAKE}, R={Radius} m, "
                f"U0={U0} m/s, B={NBlades} blades."
            ),
        )

    # ── Table 2 : Sensitivity — convection speed (a_w) ──────────────────────
    # Columns: a_w | CT | CP | delta_CT | delta_CP | iters | conv.
    if sens_aw_data:
        headers = ["a_w", "CT", "CP", "dCT", "dCP", "iters", "conv."]
        ref_CT = sens_aw_data.get(A_WAKE, (None, None, None))[1]
        ref_CP = sens_aw_data.get(A_WAKE, (None, None, None))[2]
        rows = []
        for aw in SENS_AW_LIST:
            if aw not in sens_aw_data:
                continue
            _, CT, CP = sens_aw_data[aw]
            mk = f"aw_{aw}"
            ni, cv = _solver_meta.get(mk, ("—", True))
            dCT = f"{CT - ref_CT:+.4f}" if ref_CT is not None else "—"
            dCP = f"{CP - ref_CP:+.4f}" if ref_CP is not None else "—"
            rows.append([
                f"{aw:.2f}",
                f"{CT:.4f}",
                f"{CP:.4f}",
                dCT, dCP,
                str(ni),
                u"\u2713" if cv else u"\u2717",
            ])
        _make_table_pdf(
            "Table_2_sensitivity_aw",
            f"Table 2 — Sensitivity: Wake Convection Speed a_w  (TSR={SENS_TSR})",
            headers, rows,
            col_widths=[55, 70, 70, 65, 65, 55, 55],
            footnote=(
                f"Reference: a_w={A_WAKE}. "
                f"dCT = CT - CT_ref,  dCP = CP - CP_ref. "
                f"Other params: N={N_PANELS}, N_wake={N_WAKE}, dpsi={DPSI_DEG} deg."
            ),
        )

    # ── Table 3 : Sensitivity — blade discretisation (N + distribution) ──────
    # Columns: N | distribution | CT | CP | delta_CT | delta_CP | iters | conv.
    if sens_disc_data:
        headers = ["N", "dist.", "CT", "CP", "dCT", "dCP", "iters", "conv."]
        ref_key = (N_PANELS, "cosine")
        ref_CT  = sens_disc_data[ref_key][1] if ref_key in sens_disc_data else None
        ref_CP  = sens_disc_data[ref_key][2] if ref_key in sens_disc_data else None
        rows = []
        for N in SENS_N_LIST:
            for dist in ["cosine", "constant"]:
                if (N, dist) not in sens_disc_data:
                    continue
                _, CT, CP = sens_disc_data[(N, dist)]
                mk = f"disc_{N}_{dist}"
                ni, cv = _solver_meta.get(mk, ("—", True))
                dCT = f"{CT - ref_CT:+.4f}" if ref_CT is not None else "—"
                dCP = f"{CP - ref_CP:+.4f}" if ref_CP is not None else "—"
                rows.append([
                    str(N),
                    dist.capitalize(),
                    f"{CT:.4f}",
                    f"{CP:.4f}",
                    dCT, dCP,
                    str(ni),
                    u"\u2713" if cv else u"\u2717",
                ])
        _make_table_pdf(
            "Table_3_sensitivity_disc",
            f"Table 3 — Sensitivity: Blade Discretisation  (TSR={SENS_TSR})",
            headers, rows,
            col_widths=[40, 65, 65, 65, 60, 60, 50, 50],
            footnote=(
                f"Reference: N={N_PANELS}, cosine. "
                f"dCT = CT - CT_ref,  dCP = CP - CP_ref. "
                f"Other params: N_wake={N_WAKE}, dpsi={DPSI_DEG} deg, a_w={A_WAKE}."
            ),
        )

    # ── Table 4 : Sensitivity — azimuthal step (dpsi) ────────────────────────
    # Columns: dpsi [deg] | N_steps | CT | CP | delta_CT | delta_CP | iters | conv.
    if sens_dpsi_data:
        headers = ["dpsi [deg]", "N_steps", "CT", "CP",
                   "dCT", "dCP", "iters", "conv."]
        ref_CT = sens_dpsi_data.get(DPSI_DEG, (None, None, None))[1]
        ref_CP = sens_dpsi_data.get(DPSI_DEG, (None, None, None))[2]
        rows = []
        for dpsi in SENS_DPSI_LIST:
            if dpsi not in sens_dpsi_data:
                continue
            _, CT, CP = sens_dpsi_data[dpsi]
            mk = f"dpsi_{dpsi}"
            ni, cv = _solver_meta.get(mk, ("—", True))
            # number of azimuthal steps per wake rotation
            n_steps = int(round(360.0 / dpsi))
            dCT = f"{CT - ref_CT:+.4f}" if ref_CT is not None else "—"
            dCP = f"{CP - ref_CP:+.4f}" if ref_CP is not None else "—"
            rows.append([
                f"{dpsi:.1f}",
                str(n_steps),
                f"{CT:.4f}",
                f"{CP:.4f}",
                dCT, dCP,
                str(ni),
                u"\u2713" if cv else u"\u2717",
            ])
        _make_table_pdf(
            "Table_4_sensitivity_dpsi",
            f"Table 4 — Sensitivity: Azimuthal Step  (TSR={SENS_TSR})",
            headers, rows,
            col_widths=[65, 60, 65, 65, 60, 60, 50, 50],
            footnote=(
                f"Reference: dpsi={DPSI_DEG} deg. "
                f"N_steps = 360 / dpsi (per wake rotation). "
                f"Other params: N={N_PANELS}, N_wake={N_WAKE}, a_w={A_WAKE}."
            ),
        )

    # ── Table 5 : Sensitivity — wake length (N_wake) ─────────────────────────
    # Columns: N_wake | N_filaments | CT | CP | delta_CT | delta_CP | iters | conv.
    if sens_wake_data:
        headers = ["N_wake", "N_filaments", "CT", "CP",
                   "dCT", "dCP", "iters", "conv."]
        ref_CT = sens_wake_data.get(N_WAKE, (None, None, None))[1]
        ref_CP = sens_wake_data.get(N_WAKE, (None, None, None))[2]
        rows = []
        for nw in SENS_NWAKE_LIST:
            if nw not in sens_wake_data:
                continue
            _, CT, CP = sens_wake_data[nw]
            mk = f"nwake_{nw}"
            ni, cv = _solver_meta.get(mk, ("—", True))
            # total trailing filaments per panel per blade
            n_fil = int(round(nw * 360.0 / DPSI_DEG))
            dCT = f"{CT - ref_CT:+.4f}" if ref_CT is not None else "—"
            dCP = f"{CP - ref_CP:+.4f}" if ref_CP is not None else "—"
            rows.append([
                str(nw),
                str(n_fil),
                f"{CT:.4f}",
                f"{CP:.4f}",
                dCT, dCP,
                str(ni),
                u"\u2713" if cv else u"\u2717",
            ])
        _make_table_pdf(
            "Table_5_sensitivity_wake",
            f"Table 5 — Sensitivity: Wake Length  (TSR={SENS_TSR})",
            headers, rows,
            col_widths=[60, 75, 65, 65, 60, 60, 50, 50],
            footnote=(
                f"Reference: N_wake={N_WAKE}. "
                f"N_filaments = N_wake * 360 / dpsi (trailing segments per panel). "
                f"Other params: N={N_PANELS}, dpsi={DPSI_DEG} deg, a_w={A_WAKE}."
            ),
        )

    print("All tables saved.")


# =============================================================================
# 11.  MAIN COMPUTATIONS
# =============================================================================

# ── Containers ────────────────────────────────────────────────────────────────
sweep_data_span = {}
tsr_CT_span     = []
tsr_CP_span     = []

sens_aw_data    = {}
sens_disc_data  = {}
sens_dpsi_data  = {}
sens_wake_data  = {}

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

# ── Sensitivity: convection speed (a_w) ──────────────────────────────────────
if RUN_SENS_CONV_SPEED:
    print("\n" + "="*60)
    print(f"Running sensitivity: convection speed (TSR={SENS_TSR}) ...")
    for a_w in SENS_AW_LIST:
        print(f"\n  a_w = {a_w}:")
        res, CT, CP = run_case(SENS_TSR, a_w=a_w, _meta_key=f"aw_{a_w}")
        sens_aw_data[a_w] = (res, CT, CP)

# ── Sensitivity: blade discretisation (N + constant vs cosine) ───────────────
if RUN_SENS_DISC:
    print("\n" + "="*60)
    print(f"Running sensitivity: discretisation (TSR={SENS_TSR}) ...")
    for N in SENS_N_LIST:
        for dist in ["cosine", "constant"]:
            print(f"\n  N={N}, distribution={dist}:")
            res, CT, CP = run_case(SENS_TSR, N=N, distribution=dist,
                                   _meta_key=f"disc_{N}_{dist}")
            sens_disc_data[(N, dist)] = (res, CT, CP)

# ── Sensitivity: azimuthal discretisation (dpsi) ─────────────────────────────
if RUN_SENS_AZIMUTHAL:
    print("\n" + "="*60)
    print(f"Running sensitivity: azimuthal step (TSR={SENS_TSR}) ...")
    for dpsi in SENS_DPSI_LIST:
        print(f"\n  dpsi = {dpsi} deg:")
        res, CT, CP = run_case(SENS_TSR, dpsi_deg=dpsi, _meta_key=f"dpsi_{dpsi}")
        sens_dpsi_data[dpsi] = (res, CT, CP)

# ── Sensitivity: wake length (N_wake) ────────────────────────────────────────
if RUN_SENS_WAKE_LENGTH:
    print("\n" + "="*60)
    print(f"Running sensitivity: wake length (TSR={SENS_TSR}) ...")
    for nw in SENS_NWAKE_LIST:
        print(f"\n  N_wake = {nw} rotations:")
        res, CT, CP = run_case(SENS_TSR, N_wake=nw, _meta_key=f"nwake_{nw}")
        sens_wake_data[nw] = (res, CT, CP)

# =============================================================================
# 12.  PLOTS — SECTION LL.1  (inflow angle and AoA)
# =============================================================================

n_span = len(TSR_SWEEP_SPAN)

if PLOT_LL_1 and sweep_data_span:
    for qty_col, ylabel, fname in [
            (7, r"$\phi$ [deg]",   "LL_1a_inflow_angle_vs_rR.png"),
            (6, r"$\alpha$ [deg]", "LL_1b_angle_of_attack_vs_rR.png")]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for k, TSR in enumerate(TSR_SWEEP_SPAN):
            res = sweep_data_span[TSR]
            ax.plot(res[:, 2], res[:, qty_col], color=_tsr_color(k, n_span), lw=2,
                    label=rf"$\lambda={TSR}$")
        ax.set_xlabel("r/R")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        save_fig(fname)
elif PLOT_LL_1:
    _skip("PLOT_LL_1", "span sweep data missing")

# =============================================================================
# 13.  PLOTS — SECTION LL.2  (induction factors)
# =============================================================================

if PLOT_LL_2 and sweep_data_span:
    for qty_col, ylabel, fname in [
            (0, r"$a$ [-]",  "LL_2a_axial_induction_vs_rR.png"),
            (1, r"$a'$ [-]", "LL_2b_tangential_induction_vs_rR.png")]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for k, TSR in enumerate(TSR_SWEEP_SPAN):
            res = sweep_data_span[TSR]
            ax.plot(res[:, 2], res[:, qty_col], color=_tsr_color(k, n_span), lw=2,
                    label=rf"$\lambda={TSR}$")
        ax.set_xlabel("r/R")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        save_fig(fname)
elif PLOT_LL_2:
    _skip("PLOT_LL_2", "span sweep data missing")

# =============================================================================
# 14.  PLOTS — SECTION LL.3  (loading distributions)
# =============================================================================

if PLOT_LL_3 and sweep_data_span:
    for qty_col, ylabel, fname in [
            (3, r"$C_n = F_n\,/\,(½\rho U_\infty^2 R)$", "LL_3a_normal_loading_Cn_vs_rR.png"),
            (4, r"$C_t = F_t\,/\,(½\rho U_\infty^2 R)$", "LL_3b_azimuthal_loading_Ct_vs_rR.png")]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for k, TSR in enumerate(TSR_SWEEP_SPAN):
            res = sweep_data_span[TSR]
            ax.plot(res[:, 2], res[:, qty_col] / norm_val, color=_tsr_color(k, n_span), lw=2,
                    label=rf"$\lambda={TSR}$")
        ax.set_xlabel("r/R")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        save_fig(fname)
elif PLOT_LL_3:
    _skip("PLOT_LL_3", "span sweep data missing")

# =============================================================================
# 15.  PLOTS — SECTION LL.4  (circulation distribution)
# =============================================================================

if PLOT_LL_4 and sweep_data_span:
    fig, ax = plt.subplots(figsize=(8, 5))
    for k, TSR in enumerate(TSR_SWEEP_SPAN):
        res   = sweep_data_span[TSR]
        Omega = U0 * TSR / Radius
        norm_G = np.pi * U0**2 / (NBlades * Omega)
        ax.plot(res[:, 2], res[:, 5] / norm_G, color=_tsr_color(k, n_span), lw=2,
                label=rf"$\lambda={TSR}$")
    ax.set_xlabel("r/R")
    ax.set_ylabel(r"$\Gamma\,/\,(\pi U_\infty^2 / (B\,\Omega))$ [-]")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    save_fig("LL_4_circulation_vs_rR.png")
elif PLOT_LL_4:
    _skip("PLOT_LL_4", "span sweep data missing")

# =============================================================================
# 16.  PLOTS — SECTION LL.5  (CT and CP scalars per TSR)
# =============================================================================

if PLOT_LL_5 and sweep_data_span:
    for vals, ylabel, fname, col in [
            (tsr_CT_span, r"$C_T$ [-]", "LL_5a_CT_vs_TSR.png", "#0000FF"),
            (tsr_CP_span, r"$C_P$ [-]", "LL_5b_CP_vs_TSR.png", "#2ca02c")]:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(TSR_SWEEP_SPAN, vals, "o-", color=col, lw=2)
        ax.set_xlabel(r"Tip-speed ratio $\lambda$ [-]")
        ax.set_ylabel(ylabel)
        ax.grid(True)
        fig.tight_layout()
        save_fig(fname)
elif PLOT_LL_5:
    _skip("PLOT_LL_5", "span sweep data missing")

# =============================================================================
# 17.  PLOTS — SENSITIVITY: convection speed (a_w)
# =============================================================================

if PLOT_SENS_AW and sens_aw_data:
    n_aw = len(SENS_AW_LIST)

    for qty_col, ylabel, fname in [
            (0, r"$a$ [-]",                               "Sens_AW_a_axial_induction.png"),
            (3, r"$C_n = F_n\,/\,(½\rho U_\infty^2 R)$", "Sens_AW_b_normal_loading.png"),
            (5, r"$\Gamma$ [m²/s]",                       "Sens_AW_c_circulation.png")]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for idx, a_w in enumerate(SENS_AW_LIST):
            res, CT, CP = sens_aw_data[a_w]
            y = res[:, qty_col] / norm_val if qty_col == 3 else res[:, qty_col]
            ax.plot(res[:, 2], y, color=_sens_color(idx, n_aw), lw=2,
                    label=rf"$a_w={a_w}$  $C_T={CT:.3f}$")
        ax.set_xlabel("r/R")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True)
        fig.tight_layout()
        save_fig(fname)

    aw_vals  = SENS_AW_LIST
    ct_vals  = [sens_aw_data[aw][1] for aw in aw_vals]
    cp_vals  = [sens_aw_data[aw][2] for aw in aw_vals]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(aw_vals, ct_vals, "o-", color="#0000FF", lw=2)
    axes[1].plot(aw_vals, cp_vals, "o-", color="#2ca02c", lw=2)
    axes[0].set_xlabel(r"Wake induction $a_w$ [-]"); axes[0].set_ylabel(r"$C_T$ [-]")
    axes[1].set_xlabel(r"Wake induction $a_w$ [-]"); axes[1].set_ylabel(r"$C_P$ [-]")
    axes[0].grid(True); axes[1].grid(True)
    fig.tight_layout()
    save_fig("Sens_AW_d_CT_CP_vs_aw.png")

elif PLOT_SENS_AW:
    _skip("PLOT_SENS_AW", "convection speed sensitivity data missing")

# =============================================================================
# 18.  PLOTS — SENSITIVITY: blade discretisation
# =============================================================================

if PLOT_SENS_DISC and sens_disc_data:
    n_N    = len(SENS_N_LIST)

    # A) Panel count — cosine distribution
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for idx, N in enumerate(SENS_N_LIST):
        if (N, "cosine") not in sens_disc_data: continue
        res, CT, CP = sens_disc_data[(N, "cosine")]
        col = _sens_color(idx, n_N)
        axes[0].plot(res[:, 2], res[:, 3] / norm_val, color=col, lw=2,
                     label=rf"N={N}  $C_T={CT:.3f}$")
        axes[1].plot(res[:, 2], res[:, 0], color=col, lw=2,
                     label=rf"N={N}  $C_P={CP:.3f}$")
    axes[0].set_xlabel("r/R"); axes[0].set_ylabel(r"$C_n$ [-]")
    axes[1].set_xlabel("r/R"); axes[1].set_ylabel(r"$a$ [-]")
    axes[0].set_title("Panel count sensitivity (cosine)")
    axes[1].set_title("Axial induction (cosine)")
    axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
    axes[0].grid(True); axes[1].grid(True)
    fig.tight_layout()
    save_fig("Sens_Disc_a_panel_count_cosine.png")

    # B) Constant vs cosine at baseline N
    N_ref = N_PANELS
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for dist, col, ls in [("cosine", "#0000FF", "-"), ("constant", "#d62728", "--")]:
        if (N_ref, dist) not in sens_disc_data: continue
        res, CT, CP = sens_disc_data[(N_ref, dist)]
        axes[0].plot(res[:, 2], res[:, 3] / norm_val, color=col, lw=2, ls=ls,
                     label=rf"{dist.capitalize()}  $C_T={CT:.3f}$")
        axes[1].plot(res[:, 2], res[:, 0], color=col, lw=2, ls=ls,
                     label=rf"{dist.capitalize()}  $C_P={CP:.3f}$")
    axes[0].set_xlabel("r/R"); axes[0].set_ylabel(r"$C_n$ [-]")
    axes[1].set_xlabel("r/R"); axes[1].set_ylabel(r"$a$ [-]")
    axes[0].set_title(f"Cosine vs constant spacing (N={N_ref})")
    axes[1].set_title(f"Axial induction (N={N_ref})")
    axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
    axes[0].grid(True); axes[1].grid(True)
    fig.tight_layout()
    save_fig("Sens_Disc_b_cosine_vs_constant.png")

    # C) CT and CP convergence vs N (cosine only)
    N_vals  = [N for N in SENS_N_LIST if (N, "cosine") in sens_disc_data]
    ct_N    = [sens_disc_data[(N, "cosine")][1] for N in N_vals]
    cp_N    = [sens_disc_data[(N, "cosine")][2] for N in N_vals]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(N_vals, ct_N, "o-", color="#0000FF", lw=2); axes[0].set_xlabel("N panels"); axes[0].set_ylabel(r"$C_T$ [-]"); axes[0].grid(True)
    axes[1].plot(N_vals, cp_N, "o-", color="#2ca02c", lw=2); axes[1].set_xlabel("N panels"); axes[1].set_ylabel(r"$C_P$ [-]"); axes[1].grid(True)
    fig.tight_layout()
    save_fig("Sens_Disc_c_CT_CP_convergence_vs_N.png")

elif PLOT_SENS_DISC:
    _skip("PLOT_SENS_DISC", "discretisation sensitivity data missing")

# =============================================================================
# 19.  PLOTS — SENSITIVITY: azimuthal step (dpsi)
# =============================================================================

if PLOT_SENS_DPSI and sens_dpsi_data:
    n_dpsi = len(SENS_DPSI_LIST)

    for qty_col, ylabel, fname in [
            (0, r"$a$ [-]",                               "Sens_DPSI_a_axial_induction.png"),
            (3, r"$C_n = F_n\,/\,(½\rho U_\infty^2 R)$", "Sens_DPSI_b_normal_loading.png")]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for idx, dpsi in enumerate(SENS_DPSI_LIST):
            res, CT, CP = sens_dpsi_data[dpsi]
            y = res[:, qty_col] / norm_val if qty_col == 3 else res[:, qty_col]
            ax.plot(res[:, 2], y, color=_sens_color(idx, n_dpsi), lw=2,
                    label=rf"$\Delta\psi={dpsi}°$  $C_T={CT:.3f}$")
        ax.set_xlabel("r/R"); ax.set_ylabel(ylabel)
        ax.legend(fontsize=8); ax.grid(True)
        fig.tight_layout()
        save_fig(fname)

    dpsi_vals = SENS_DPSI_LIST
    ct_vals   = [sens_dpsi_data[d][1] for d in dpsi_vals]
    cp_vals   = [sens_dpsi_data[d][2] for d in dpsi_vals]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(dpsi_vals, ct_vals, "o-", color="#0000FF", lw=2)
    axes[1].plot(dpsi_vals, cp_vals, "o-", color="#2ca02c", lw=2)
    axes[0].set_xlabel(r"$\Delta\psi$ [deg]"); axes[0].set_ylabel(r"$C_T$ [-]")
    axes[1].set_xlabel(r"$\Delta\psi$ [deg]"); axes[1].set_ylabel(r"$C_P$ [-]")
    axes[0].grid(True); axes[1].grid(True)
    fig.tight_layout()
    save_fig("Sens_DPSI_c_CT_CP_vs_dpsi.png")

elif PLOT_SENS_DPSI:
    _skip("PLOT_SENS_DPSI", "azimuthal step sensitivity data missing")

# =============================================================================
# 20.  PLOTS — SENSITIVITY: wake length (N_wake)
# =============================================================================

if PLOT_SENS_WAKE and sens_wake_data:
    n_nw = len(SENS_NWAKE_LIST)

    for qty_col, ylabel, fname in [
            (0, r"$a$ [-]",                               "Sens_Wake_a_axial_induction.png"),
            (3, r"$C_n = F_n\,/\,(½\rho U_\infty^2 R)$", "Sens_Wake_b_normal_loading.png")]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for idx, nw in enumerate(SENS_NWAKE_LIST):
            res, CT, CP = sens_wake_data[nw]
            y = res[:, qty_col] / norm_val if qty_col == 3 else res[:, qty_col]
            ax.plot(res[:, 2], y, color=_sens_color(idx, n_nw), lw=2,
                    label=rf"$N_{{wake}}={nw}$  $C_T={CT:.3f}$")
        ax.set_xlabel("r/R"); ax.set_ylabel(ylabel)
        ax.legend(fontsize=8); ax.grid(True)
        fig.tight_layout()
        save_fig(fname)

    nw_vals = SENS_NWAKE_LIST
    ct_vals = [sens_wake_data[nw][1] for nw in nw_vals]
    cp_vals = [sens_wake_data[nw][2] for nw in nw_vals]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(nw_vals, ct_vals, "o-", color="#0000FF", lw=2)
    axes[1].plot(nw_vals, cp_vals, "o-", color="#2ca02c", lw=2)
    axes[0].set_xlabel(r"Wake length $N_{wake}$ [rotations]"); axes[0].set_ylabel(r"$C_T$ [-]")
    axes[1].set_xlabel(r"Wake length $N_{wake}$ [rotations]"); axes[1].set_ylabel(r"$C_P$ [-]")
    axes[0].grid(True); axes[1].grid(True)
    fig.tight_layout()
    save_fig("Sens_Wake_c_CT_CP_convergence_vs_Nwake.png")

elif PLOT_SENS_WAKE:
    _skip("PLOT_SENS_WAKE", "wake length sensitivity data missing")

# =============================================================================
# 21.  SAVE RESULTS TO NPZ  (for PLOTTING_LL_FINAL.py)
# =============================================================================

def save_ll_results(path=LL_RESULTS_PATH):
    """
    Save all computed results to a .npz file.
    """
    kw = dict(
        polar_alpha=polar_alpha, polar_cl=polar_cl, polar_cd=polar_cd,
        cfg_Radius=Radius, cfg_NBlades=NBlades, cfg_U0=U0, cfg_rho=rho,
        cfg_RootLocation_R=RootLocation_R, cfg_TipLocation_R=TipLocation_R,
        cfg_Pitch=Pitch, cfg_N_PANELS=N_PANELS, cfg_N_WAKE=N_WAKE,
        cfg_DPSI_DEG=DPSI_DEG, cfg_A_WAKE=A_WAKE,
        sweep_tsrs=np.array(TSR_SWEEP_SPAN, dtype=float),
        tsr_CT=tsr_CT_span if len(tsr_CT_span) > 0 else np.array([]),
        tsr_CP=tsr_CP_span if len(tsr_CP_span) > 0 else np.array([]),
        sens_aw_vals  =np.array(SENS_AW_LIST,    dtype=float),
        sens_aw_CT    =np.array([sens_aw_data[aw][1]   for aw in SENS_AW_LIST   if aw   in sens_aw_data],   dtype=float),
        sens_aw_CP    =np.array([sens_aw_data[aw][2]   for aw in SENS_AW_LIST   if aw   in sens_aw_data],   dtype=float),
        sens_dpsi_vals=np.array(SENS_DPSI_LIST,  dtype=float),
        sens_dpsi_CT  =np.array([sens_dpsi_data[d][1]  for d  in SENS_DPSI_LIST if d    in sens_dpsi_data], dtype=float),
        sens_dpsi_CP  =np.array([sens_dpsi_data[d][2]  for d  in SENS_DPSI_LIST if d    in sens_dpsi_data], dtype=float),
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
# 22.  SAVE PDF TABLES
# =============================================================================

if SAVE_TABLES_PDF:
    save_all_tables()
else:
    print("\nSAVE_TABLES_PDF=False — no tables written.")

print("\nDone!")
print("  Plots  →", save_folder)
print("  Tables →", tables_folder)