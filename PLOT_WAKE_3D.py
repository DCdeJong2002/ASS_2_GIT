"""
VortexGeometry.py  —  AE4135 Rotor/Wake Aerodynamics, Assignment 2
Interactive 3-D visualisation of the frozen vortex wake geometry.

Run:  python VortexGeometry.py
      Outputs (all written to the same directory as this script):

      vortex_geometry_report.pdf          single-panel publication figure
      vortex_geometry_comp_dpsi.pdf       3-panel Δψ comparison
      vortex_geometry_comp_aw.pdf         3-panel a_w comparison
      vortex_geometry_comp_nwake.pdf      3-panel N_wake comparison
      *.html variants                     interactive Plotly (if toggle = True)

Requires:  numpy  plotly  matplotlib  seaborn
"""

import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

# =============================================================================
# CONFIGURATION  —  edit here
# =============================================================================

# ── Rotor (must match LiftingLine_FINAL.py) ───────────────────────────────────
Radius         = 50.0
NBlades        = 3
U0             = 10.0
RootLocation_R = 0.2
TipLocation_R  = 1.0
Pitch          = -2.0          # blade pitch [deg]

# ── Default single-panel parameters ──────────────────────────────────────────
TSR      = 8       # tip-speed ratio
N        = 20      # spanwise panels per blade
N_wake   = 5       # number of full wake rotations shown
dpsi_deg = 10.0    # azimuthal step [deg]
a_w      = 0.25    # frozen-wake axial induction (convection factor)

# ── Comparison lists (exactly 3 values each) ──────────────────────────────────
COMPARE_DPSI   = [2.0, 10.0, 90.0]   # Δψ [deg]          — other params at defaults
COMPARE_AW     = [0.05,  0.25, 0.50]   # axial induction    — other params at defaults
COMPARE_N_WAKE = [1,    5,    15]   # wake rotations     — other params at defaults

# ── Output toggles ────────────────────────────────────────────────────────────
SAVE_REPORT_PDF        = True    # single-panel matplotlib PDF
SAVE_PLOTLY_HTML       = False   # single-panel interactive Plotly HTML

SAVE_COMP_DPSI_PDF     = True    # Δψ   comparison PDF
SAVE_COMP_AW_PDF       = True    # a_w  comparison PDF
SAVE_COMP_NWAKE_PDF    = True    # N_wake comparison PDF

SAVE_COMP_DPSI_HTML    = False   # Δψ   comparison interactive HTML
SAVE_COMP_AW_HTML      = False   # a_w  comparison interactive HTML
SAVE_COMP_NWAKE_HTML   = False   # N_wake comparison interactive HTML

# ── Seaborn colorblind palette ────────────────────────────────────────────────
_CB = sns.color_palette("colorblind")
BLADE_COLORS_MPL = [_CB[0], _CB[1], _CB[2]]
def _to255(c):
    return tuple(int(round(v * 255)) for v in c)
BLADE_COLORS_PLY = [_to255(c) for c in BLADE_COLORS_MPL]

# =============================================================================
# BLADE GEOMETRY
# =============================================================================

def blade_chord(r_R):
    return 3.0 * (1.0 - r_R) + 1.0

def blade_twist(r_R):
    return 14.0 * (1.0 - r_R) + Pitch

def make_panels(n_panels, distribution="cosine"):
    r_root  = RootLocation_R * Radius
    r_tip   = TipLocation_R  * Radius
    if distribution == "cosine":
        theta   = np.linspace(0.0, np.pi, n_panels + 1)
        r_edges = r_root + 0.5 * (r_tip - r_root) * (1.0 - np.cos(theta))
    else:
        r_edges = np.linspace(r_root, r_tip, n_panels + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    dr        = np.diff(r_edges)
    return r_edges, r_centers, dr

# =============================================================================
# VORTEX SYSTEM BUILDER
# =============================================================================

def build_vortex_system(Omega, n_panels, n_wake, dpsi, aw):
    """
    Parameters
    ----------
    Omega    : rotor angular velocity [rad/s]
    n_panels : spanwise panels per blade
    n_wake   : number of full wake rotations
    dpsi     : azimuthal step [deg]
    aw       : frozen-wake axial induction factor
    """
    r_edges, r_centers, dr = make_panels(n_panels)
    U_wake  = U0 * (1.0 - aw)
    dpsi_r  = np.radians(dpsi)
    psi_arr = np.arange(0.0, n_wake * 2.0 * np.pi + dpsi_r * 0.5, dpsi_r)

    controlpoints, rings = [], []

    for k_blade in range(NBlades):
        angle_rot  = 2.0 * np.pi / NBlades * k_blade
        cosR, sinR = np.cos(angle_rot), np.sin(angle_rot)

        for i in range(n_panels):
            r   = r_centers[i]
            r_R = r / Radius
            controlpoints.append({
                'r': r, 'r_R': r_R,
                'chord': blade_chord(r_R),
                'twist_rad': np.radians(blade_twist(r_R)),
                'coords': np.array([0.0, r * cosR, r * sinR]),
                'dr': dr[i],
            })

            filaments = []

            # bound vortex
            y_in  = r_edges[i]   * cosR;  z_in  = r_edges[i]   * sinR
            y_out = r_edges[i+1] * cosR;  z_out = r_edges[i+1] * sinR
            filaments.append({'x1':0., 'y1':y_in,  'z1':z_in,
                               'x2':0., 'y2':y_out, 'z2':z_out})

            # inner trailing (reversed: wake → rotor)
            for j in range(len(psi_arr) - 1):
                t1 = psi_arr[j]   / Omega;  t2 = psi_arr[j+1] / Omega
                ri = r_edges[i]
                x1 = U_wake*t1; y1 = ri*np.cos(-Omega*t1); z1 = ri*np.sin(-Omega*t1)
                x2 = U_wake*t2; y2 = ri*np.cos(-Omega*t2); z2 = ri*np.sin(-Omega*t2)
                y1r = y1*cosR - z1*sinR;  z1r = y1*sinR + z1*cosR
                y2r = y2*cosR - z2*sinR;  z2r = y2*sinR + z2*cosR
                filaments.append({'x1':x2,'y1':y2r,'z1':z2r, 'x2':x1,'y2':y1r,'z2':z1r})

            # outer trailing (forward: rotor → wake)
            for j in range(len(psi_arr) - 1):
                t1 = psi_arr[j]   / Omega;  t2 = psi_arr[j+1] / Omega
                ro = r_edges[i+1]
                x1 = U_wake*t1; y1 = ro*np.cos(-Omega*t1); z1 = ro*np.sin(-Omega*t1)
                x2 = U_wake*t2; y2 = ro*np.cos(-Omega*t2); z2 = ro*np.sin(-Omega*t2)
                y1r = y1*cosR - z1*sinR;  z1r = y1*sinR + z1*cosR
                y2r = y2*cosR - z2*sinR;  z2r = y2*sinR + z2*cosR
                filaments.append({'x1':x1,'y1':y1r,'z1':z1r, 'x2':x2,'y2':y2r,'z2':z2r})

            rings.append(filaments)

    return controlpoints, rings

# =============================================================================
# SHARED MATPLOTLIB STYLE
# =============================================================================

def _apply_rcparams(base_fontsize=11):
    mpl.rcParams.update({
        "text.usetex"       : False,
        "font.family"       : "serif",
        "font.serif"        : ["CMU Serif", "Computer Modern Roman",
                               "Latin Modern Roman", "DejaVu Serif"],
        "mathtext.fontset"  : "cm",
        "axes.labelsize"    : base_fontsize,
        "legend.fontsize"   : base_fontsize - 2,
        "xtick.labelsize"   : base_fontsize - 2,
        "ytick.labelsize"   : base_fontsize - 2,
        "axes.titlesize"    : base_fontsize,
        "savefig.bbox"      : "tight",
        "savefig.pad_inches": 0.02,
        "legend.frameon"    : True,
    })

# =============================================================================
# HELPER: populate a single Axes3D with the full wake geometry
# =============================================================================

def _draw_wake_on_ax(ax, rings, cps, n_panels_per_blade, elev=22, azim=118,
                     lw_scale=1.0):
    """
    Draw bound + trailing vortex filaments, blade outlines, disc and axis
    into *ax* (a matplotlib Axes3D).  lw_scale lets comparison panels use
    slightly thinner lines than the full single-panel figure.
    """
    n_ws = (len(rings[0]) - 1) // 2

    pane_col = (0.93, 0.93, 0.93, 0.50)
    ax.set_facecolor('white')
    ax.xaxis.set_pane_color(pane_col)
    ax.yaxis.set_pane_color(pane_col)
    ax.zaxis.set_pane_color(pane_col)
    ax.xaxis._axinfo['grid']['color'] = (0.80, 0.80, 0.80, 0.50)
    ax.yaxis._axinfo['grid']['color'] = (0.80, 0.80, 0.80, 0.50)
    ax.zaxis._axinfo['grid']['color'] = (0.80, 0.80, 0.80, 0.50)

    for b in range(NBlades):
        col   = BLADE_COLORS_MPL[b]
        angle = 2.0 * np.pi / NBlades * b
        cosR, sinR = np.cos(angle), np.sin(angle)

        # blade outline
        _, r_arr, _ = make_panels(80)
        le_pts, te_pts = [], []
        for r in r_arr:
            r_R   = r / Radius
            chord = blade_chord(r_R)
            twist = np.radians(blade_twist(r_R))
            ry = r * cosR;  rz = r * sinR
            le_pts.append([(chord/2)*np.sin(twist),  ry, rz-(chord/2)*np.cos(twist)])
            te_pts.append([-(chord/2)*np.sin(twist), ry, rz+(chord/2)*np.cos(twist)])
        le_pts = np.array(le_pts);  te_pts = np.array(te_pts)

        ax.plot(le_pts[:,0], le_pts[:,1], le_pts[:,2],
                color=col, lw=1.8*lw_scale, alpha=1.0, zorder=8)
        ax.plot(te_pts[:,0], te_pts[:,1], te_pts[:,2],
                color=col, lw=1.0*lw_scale, alpha=0.60, zorder=8)
        ax.plot([le_pts[-1,0],te_pts[-1,0]], [le_pts[-1,1],te_pts[-1,1]],
                [le_pts[-1,2],te_pts[-1,2]], color=col, lw=0.8*lw_scale, alpha=0.60)
        ax.plot([le_pts[0,0], te_pts[0,0]], [le_pts[0,1], te_pts[0,1]],
                [le_pts[0,2], te_pts[0,2]], color=col, lw=0.8*lw_scale, alpha=0.60)
        step = max(1, len(r_arr) // 8)
        for idx in range(0, len(r_arr), step):
            ax.plot([le_pts[idx,0],te_pts[idx,0]],
                    [le_pts[idx,1],te_pts[idx,1]],
                    [le_pts[idx,2],te_pts[idx,2]],
                    color=col, lw=0.4*lw_scale, alpha=0.18)

        # bound vortex
        bv_xs, bv_ys, bv_zs = [], [], []
        for p in range(n_panels_per_blade):
            f = rings[b*n_panels_per_blade+p][0]
            bv_xs += [f['x1'],f['x2']]; bv_ys += [f['y1'],f['y2']]; bv_zs += [f['z1'],f['z2']]
        ax.plot(bv_xs, bv_ys, bv_zs, color=col, lw=3.0*lw_scale,
                solid_capstyle='round', zorder=10)

        # inner trailing
        for p in range(n_panels_per_blade):
            for f in rings[b*n_panels_per_blade+p][1 : n_ws+1]:
                ax.plot([f['x1'],f['x2']], [f['y1'],f['y2']], [f['z1'],f['z2']],
                        color=col, lw=0.8*lw_scale, alpha=0.65, zorder=4)

        # outer trailing
        light = tuple(min(1.0, c*0.60+0.40) for c in col[:3])
        for p in range(n_panels_per_blade):
            for f in rings[b*n_panels_per_blade+p][n_ws+1 : 2*n_ws+1]:
                ax.plot([f['x1'],f['x2']], [f['y1'],f['y2']], [f['z1'],f['z2']],
                        color=light, lw=0.55*lw_scale, alpha=0.50,
                        linestyle='--', dashes=(5, 4), zorder=3)

    # rotor disc
    theta    = np.linspace(0, 2*np.pi, 360)
    r_root_m = RootLocation_R * Radius
    ax.plot(np.zeros(360), Radius*np.cos(theta), Radius*np.sin(theta),
            color='#999999', lw=1.0, alpha=0.55)
    ax.plot(np.zeros(360), r_root_m*np.cos(theta), r_root_m*np.sin(theta),
            color='#999999', lw=0.6, alpha=0.40)
    for ang in np.linspace(0, 2*np.pi, 7)[:-1]:
        ax.plot([0,0], [r_root_m*np.cos(ang), Radius*np.cos(ang)],
                       [r_root_m*np.sin(ang), Radius*np.sin(ang)],
                color='#cccccc', lw=0.4, alpha=0.35)

    # rotor axis
    x_max = max(f['x2'] for ring in rings for f in ring)
    ax.plot([-8, x_max*1.02], [0,0], [0,0],
            color='#888888', lw=0.8, linestyle=':', alpha=0.65)

    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('x — axial [m]', labelpad=8, fontsize=9)
    ax.set_ylabel('y [m]',          labelpad=8, fontsize=9)
    ax.set_zlabel('z [m]',          labelpad=6, fontsize=9)

# =============================================================================
# SHARED LEGEND HANDLES
# =============================================================================

def _legend_handles():
    handles = []
    for b in range(NBlades):
        handles.append(Line2D([0],[0], color=BLADE_COLORS_MPL[b],
                               lw=2.2, label=f'Blade {b+1}'))
    handles += [
        Line2D([0],[0], color='white',   lw=0,   label=' '),
        Line2D([0],[0], color='#333333', lw=0,   label='Line types:'),
        Line2D([0],[0], color='#444444', lw=3.2, label='Bound vortex'),
        Line2D([0],[0], color='#555555', lw=0.9, alpha=0.8,
               label='Inner trailing'),
        Line2D([0],[0], color='#aaaaaa', lw=0.8, linestyle='--',
               dashes=(5,4), label='Outer trailing'),
        Line2D([0],[0], color='#999999', lw=1.0, label='Rotor disc / axis'),
    ]
    return handles

# =============================================================================
# GENERIC 3-PANEL COMPARISON  (matplotlib PDF + optional Plotly HTML)
# =============================================================================

def _make_comparison_pdf(param_values, subtitle_fn, build_kwargs_fn,
                          out_pdf, out_html=None):
    """
    Parameters
    ----------
    param_values    : list of 3 values to compare
    subtitle_fn     : callable(value) → str  — subplot title for each panel
    build_kwargs_fn : callable(value) → dict with keys
                      {n_panels, n_wake, dpsi, aw}
                      passed to build_vortex_system()
    out_pdf         : output PDF path
    out_html        : output HTML path, or None to skip
    """
    _apply_rcparams(base_fontsize=10)
    Omega = U0 * TSR / Radius

    n_panels_fig = len(param_values)
    panel_w      = 0.80 / n_panels_fig
    leg_x0       = 0.82

    fig = plt.figure(figsize=(5.5 * n_panels_fig + 2.5, 6.5))
    fig.patch.set_facecolor('white')

    for idx, val in enumerate(param_values):
        kw          = build_kwargs_fn(val)
        cps, rings  = build_vortex_system(Omega, **kw)
        n_pb        = len(cps) // NBlades

        left = 0.01 + idx * panel_w
        ax   = fig.add_axes([left, 0.04, panel_w - 0.01, 0.88], projection='3d')
        _draw_wake_on_ax(ax, rings, cps, n_pb, elev=22, azim=118, lw_scale=0.85)
        ax.set_title(subtitle_fn(val), fontsize=15, fontweight='semibold', pad=0)

    # legend panel
    ax_leg = fig.add_axes([leg_x0, 0.12, 1.0 - leg_x0 - 0.01, 0.76])
    ax_leg.axis('off')
    leg = ax_leg.legend(
        handles=_legend_handles(), loc='center',
        framealpha=0.97, edgecolor='#cccccc',
        handlelength=2.6, labelspacing=0.55,
        title='Vortex wake geometry', title_fontsize=10,
    )
    leg.get_frame().set_linewidth(0.7)
    leg.get_title().set_fontweight('semibold')

    fig.savefig(out_pdf, dpi=300, facecolor='white', edgecolor='none')
    print(f"Saved  →  {out_pdf}")
    plt.close(fig)

    if out_html is not None:
        _make_comparison_html(param_values, subtitle_fn, build_kwargs_fn, out_html)


def _make_comparison_html(param_values, subtitle_fn, build_kwargs_fn, out_html):
    """Interactive Plotly version of the 3-panel comparison."""
    Omega = U0 * TSR / Radius
    n     = len(param_values)

    fig = make_subplots(
        rows=1, cols=n,
        specs=[[{'type': 'scatter3d'}] * n],
        subplot_titles=[subtitle_fn(v) for v in param_values],
    )

    def rgba(rgb, a):
        return f'rgba({rgb[0]},{rgb[1]},{rgb[2]},{a})'
    def lighten(rgb, f):
        return tuple(int(c + (255-c)*f) for c in rgb)

    for ci, val in enumerate(param_values):
        kw          = build_kwargs_fn(val)
        cps, rings  = build_vortex_system(Omega, **kw)
        n_pb        = len(cps) // NBlades
        n_ws        = (len(rings[0]) - 1) // 2
        show_leg    = (ci == 0)

        for b in range(NBlades):
            rgb   = BLADE_COLORS_PLY[b]
            bname = f'Blade {b+1}'

            bound_s, inner_s, outer_s = [], [], []
            for p in range(n_pb):
                ring = rings[b*n_pb+p]
                bound_s.append(ring[0])
                inner_s.extend(ring[1      : n_ws+1])
                outer_s.extend(ring[n_ws+1 : 2*n_ws+1])

            def seg_xyz(segs):
                xs, ys, zs = [], [], []
                for f in segs:
                    xs += [f['x1'],f['x2'],None]; ys += [f['y1'],f['y2'],None]
                    zs += [f['z1'],f['z2'],None]
                return xs, ys, zs

            bx,by,bz = seg_xyz(bound_s)
            fig.add_trace(go.Scatter3d(x=bx,y=by,z=bz,mode='lines',
                line=dict(color=rgba(rgb,1.0),width=4),
                name=f'{bname} bound',legendgroup=f'bv{b}',
                showlegend=show_leg,hoverinfo='skip'), row=1,col=ci+1)

            ix,iy,iz = seg_xyz(inner_s)
            fig.add_trace(go.Scatter3d(x=ix,y=iy,z=iz,mode='lines',
                line=dict(color=rgba(rgb,0.55),width=1.2),
                name=f'{bname} inner',legendgroup=f'it{b}',
                showlegend=show_leg,hoverinfo='skip'), row=1,col=ci+1)

            ox,oy,oz = seg_xyz(outer_s)
            fig.add_trace(go.Scatter3d(x=ox,y=oy,z=oz,mode='lines',
                line=dict(color=rgba(lighten(rgb,0.40),0.35),width=0.8,dash='dot'),
                name=f'{bname} outer',legendgroup=f'ot{b}',
                showlegend=show_leg,hoverinfo='skip'), row=1,col=ci+1)

        theta = np.linspace(0,2*np.pi,180)
        fig.add_trace(go.Scatter3d(
            x=np.zeros(180),y=Radius*np.cos(theta),z=Radius*np.sin(theta),
            mode='lines',line=dict(color='rgba(150,150,150,0.25)',width=1),
            name='Rotor disc',legendgroup='disc',
            showlegend=show_leg,hoverinfo='skip'), row=1,col=ci+1)

        x_max = max(f['x2'] for ring in rings for f in ring)
        fig.add_trace(go.Scatter3d(
            x=[-8,x_max*1.02],y=[0,0],z=[0,0],mode='lines',
            line=dict(color='rgba(150,150,150,0.3)',width=1,dash='dot'),
            name='Rotor axis',legendgroup='axis',
            showlegend=show_leg,hoverinfo='skip'), row=1,col=ci+1)

        sk = 'scene' if ci == 0 else f'scene{ci+1}'
        fig.layout[sk].update(
            xaxis=dict(title='x — axial [m]',color='#888',
                       gridcolor='rgba(255,255,255,0.07)',
                       showbackground=True,backgroundcolor='rgba(0,0,0,0)'),
            yaxis=dict(title='y [m]',color='#888',
                       gridcolor='rgba(255,255,255,0.07)',
                       showbackground=True,backgroundcolor='rgba(0,0,0,0)'),
            zaxis=dict(title='z [m]',color='#888',
                       gridcolor='rgba(255,255,255,0.07)',
                       showbackground=True,backgroundcolor='rgba(0,0,0,0)'),
            bgcolor='#0f0f0f', aspectmode='data',
            camera=dict(eye=dict(x=1.3,y=0.75,z=0.55)),
        )

    fig.update_layout(
        paper_bgcolor='#0f0f0f',
        font=dict(color='#cccccc',size=10,family='Arial'),
        legend=dict(bgcolor='rgba(20,20,20,0.88)',
                    bordercolor='rgba(255,255,255,0.12)',
                    borderwidth=1,font=dict(size=9)),
        height=700,
    )
    fig.write_html(out_html)
    print(f"Saved  →  {out_html}")

# =============================================================================
# THREE COMPARISON WRAPPERS
# =============================================================================

def _here(fname):
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'LLM_3D_plots')
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, fname)


# ── Δψ comparison ─────────────────────────────────────────────────────────────
def plot_comparison_dpsi():
    if not SAVE_COMP_DPSI_PDF:
        return
    _make_comparison_pdf(
        param_values    = COMPARE_DPSI,
        subtitle_fn     = lambda d: (f'$\\Delta\\psi = {d:.0f}^\\circ$  '
                                     f'({int(round(360/d))} steps/rev)'),
        build_kwargs_fn = lambda d: dict(n_panels=N, n_wake=N_wake,
                                         dpsi=d, aw=a_w),
        out_pdf         = _here('vortex_geometry_comp_dpsi.pdf'),
        out_html        = _here('vortex_geometry_comp_dpsi.html')
                          if SAVE_COMP_DPSI_HTML else None,
    )


# ── a_w comparison ────────────────────────────────────────────────────────────
def plot_comparison_aw():
    if not SAVE_COMP_AW_PDF:
        return
    _make_comparison_pdf(
        param_values    = COMPARE_AW,
        subtitle_fn     = lambda a: f'$a_w = {a:.2f}$',
        build_kwargs_fn = lambda a: dict(n_panels=N, n_wake=N_wake,
                                         dpsi=dpsi_deg, aw=a),
        out_pdf         = _here('vortex_geometry_comp_aw.pdf'),
        out_html        = _here('vortex_geometry_comp_aw.html')
                          if SAVE_COMP_AW_HTML else None,
    )


# ── N_wake comparison ─────────────────────────────────────────────────────────
def plot_comparison_nwake():
    if not SAVE_COMP_NWAKE_PDF:
        return
    _make_comparison_pdf(
        param_values    = COMPARE_N_WAKE,
        subtitle_fn     = lambda nw: (f'$N_{{wake}} = {nw}$  '
                                      f'({nw} rotation{"s" if nw > 1 else ""})'),
        build_kwargs_fn = lambda nw: dict(n_panels=N, n_wake=nw,
                                          dpsi=dpsi_deg, aw=a_w),
        out_pdf         = _here('vortex_geometry_comp_nwake.pdf'),
        out_html        = _here('vortex_geometry_comp_nwake.html')
                          if SAVE_COMP_NWAKE_HTML else None,
    )

# =============================================================================
# SINGLE-PANEL FIGURES  (unchanged from original)
# =============================================================================

def plot_report_figure():
    """Publication-quality single-panel matplotlib PDF."""
    if not SAVE_REPORT_PDF:
        return
    _apply_rcparams(base_fontsize=12)
    Omega       = U0 * TSR / Radius
    cps, rings  = build_vortex_system(Omega, N, N_wake, dpsi_deg, a_w)
    n_pb        = len(cps) // NBlades

    fig     = plt.figure(figsize=(12, 5.0))
    ax      = fig.add_axes([-0.03, -0.10, 0.80, 1.20], projection='3d')
    ax_leg  = fig.add_axes([0.76, 0.10, 0.24, 0.80])
    ax_leg.axis('off')
    fig.patch.set_facecolor('white')

    _draw_wake_on_ax(ax, rings, cps, n_pb, elev=22, azim=118, lw_scale=1.0)

    handles = _legend_handles()
    # replace generic 'Blade X' labels with larger line width for single panel
    leg = ax_leg.legend(handles=handles, loc='center',
                        framealpha=0.97, edgecolor='#cccccc',
                        handlelength=2.6, labelspacing=0.55,
                        title='Vortex wake geometry', title_fontsize=11)
    leg.get_frame().set_linewidth(0.7)
    leg.get_title().set_fontweight('semibold')

    out = _here('vortex_geometry_report.pdf')
    fig.savefig(out, dpi=300, facecolor='white', edgecolor='none', bbox_inches='tight')
    print(f"Saved  →  {out}")
    plt.close(fig)
    return out


def plot_vortex_geometry():
    """Interactive single-panel Plotly HTML."""
    if not SAVE_PLOTLY_HTML:
        return
    Omega       = U0 * TSR / Radius
    cps, rings  = build_vortex_system(Omega, N, N_wake, dpsi_deg, a_w)
    n_pb        = len(cps) // NBlades
    n_ws        = (len(rings[0]) - 1) // 2

    def rgba(rgb, a): return f'rgba({rgb[0]},{rgb[1]},{rgb[2]},{a})'
    def lighten(rgb, f): return tuple(int(c+(255-c)*f) for c in rgb)

    def seg_trace(segs, color, width, name, lg, show_legend, dash=None):
        xs, ys, zs = [], [], []
        for f in segs:
            xs += [f['x1'],f['x2'],None]; ys += [f['y1'],f['y2'],None]
            zs += [f['z1'],f['z2'],None]
        ld = dict(color=color, width=width)
        if dash: ld['dash'] = dash
        return go.Scatter3d(x=xs,y=ys,z=zs,mode='lines',line=ld,
                            name=name,legendgroup=lg,
                            showlegend=show_legend,hoverinfo='skip')

    traces = []
    for b in range(NBlades):
        rgb   = BLADE_COLORS_PLY[b]
        bname = f'Blade {b+1}'
        angle = 2.0*np.pi/NBlades*b
        cosR, sinR = np.cos(angle), np.sin(angle)

        bound_s, inner_s, outer_s = [], [], []
        for p in range(n_pb):
            ring = rings[b*n_pb+p]
            bound_s.append(ring[0])
            inner_s.extend(ring[1      : n_ws+1])
            outer_s.extend(ring[n_ws+1 : 2*n_ws+1])

        _, r_arr, _ = make_panels(40)
        le_x,le_y,le_z,te_x,te_y,te_z = [],[],[],[],[],[]
        cxs,cys,czs = [],[],[]
        step = max(1, len(r_arr)//12)
        for idx, r in enumerate(r_arr):
            r_R=r/Radius; chord=blade_chord(r_R); twist=np.radians(blade_twist(r_R))
            ry=r*cosR; rz=r*sinR
            le_x.append( (chord/2)*np.sin(twist)); le_y.append(ry); le_z.append(rz-(chord/2)*np.cos(twist))
            te_x.append(-(chord/2)*np.sin(twist)); te_y.append(ry); te_z.append(rz+(chord/2)*np.cos(twist))
            if idx%step==0:
                cxs+=[le_x[-1],te_x[-1],None]; cys+=[le_y[-1],te_y[-1],None]; czs+=[le_z[-1],te_z[-1],None]

        traces.append(go.Scatter3d(x=le_x,y=le_y,z=le_z,mode='lines',
            line=dict(color=rgba(rgb,0.85),width=3),
            name=f'{bname} LE',legendgroup=f'bl{b}',showlegend=True,hoverinfo='skip'))
        traces.append(go.Scatter3d(x=te_x,y=te_y,z=te_z,mode='lines',
            line=dict(color=rgba(rgb,0.30),width=1.5),
            name=f'{bname} TE',legendgroup=f'bl{b}',showlegend=False,hoverinfo='skip'))
        traces.append(seg_trace(bound_s,rgba(rgb,1.0),5,
                                f'{bname} bound vortex',f'bv{b}',True))
        traces.append(seg_trace(inner_s,rgba(rgb,0.55),1.5,
                                f'{bname} inner trailing',f'it{b}',True))
        traces.append(seg_trace(outer_s,rgba(lighten(rgb,0.40),0.35),1.0,
                                f'{bname} outer trailing',f'ot{b}',True,dash='dot'))

    theta = np.linspace(0,2*np.pi,180)
    traces.append(go.Scatter3d(
        x=np.zeros(180),y=Radius*np.cos(theta),z=Radius*np.sin(theta),
        mode='lines',line=dict(color='rgba(200,200,200,0.18)',width=1),
        name='Rotor disc',legendgroup='disc',showlegend=True,hoverinfo='skip'))

    x_max = max(f['x2'] for ring in rings for f in ring)
    traces.append(go.Scatter3d(
        x=[-8,x_max*1.05],y=[0,0],z=[0,0],mode='lines',
        line=dict(color='rgba(200,200,200,0.15)',width=1,dash='dot'),
        name='Rotor axis',legendgroup='axis',showlegend=True,hoverinfo='skip'))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=(f'Frozen vortex wake  ·  {NBlades} blades  ·  '
                         f'N = {N}  ·  {N_wake} wake rot.  ·  TSR = {TSR}  ·  '
                         f'Δψ = {dpsi_deg}°'),
                   font=dict(size=12,color='#cccccc'),x=0.5,xanchor='center'),
        scene=dict(
            xaxis=dict(title='x — axial [m]',showgrid=True,
                       gridcolor='rgba(255,255,255,0.07)',color='#888',
                       backgroundcolor='rgba(0,0,0,0)',showbackground=True),
            yaxis=dict(title='y [m]',showgrid=True,
                       gridcolor='rgba(255,255,255,0.07)',color='#888',
                       backgroundcolor='rgba(0,0,0,0)',showbackground=True),
            zaxis=dict(title='z [m]',showgrid=True,
                       gridcolor='rgba(255,255,255,0.07)',color='#888',
                       backgroundcolor='rgba(0,0,0,0)',showbackground=True),
            bgcolor='#0f0f0f',aspectmode='data',
            camera=dict(eye=dict(x=1.3,y=0.75,z=0.55),up=dict(x=0,y=0,z=1)),
        ),
        paper_bgcolor='#0f0f0f',
        font=dict(color='#cccccc',size=11,family='Arial'),
        legend=dict(bgcolor='rgba(20,20,20,0.88)',
                    bordercolor='rgba(255,255,255,0.12)',
                    borderwidth=1,tracegroupgap=2,
                    font=dict(size=10),itemsizing='constant'),
        margin=dict(l=0,r=0,t=50,b=0), height=750,
    )
    out = _here('vortex_geometry.html')
    fig.write_html(out)
    print(f"Saved  →  {out}")
    fig.show()
    return fig

# =============================================================================
if __name__ == "__main__":
    plot_report_figure()        # single-panel PDF  (if SAVE_REPORT_PDF)
    plot_comparison_dpsi()      # Δψ comparison     (if SAVE_COMP_DPSI_PDF)
    plot_comparison_aw()        # a_w comparison    (if SAVE_COMP_AW_PDF)
    plot_comparison_nwake()     # N_wake comparison  (if SAVE_COMP_NWAKE_PDF)
    plot_vortex_geometry()      # interactive HTML   (if SAVE_PLOTLY_HTML)