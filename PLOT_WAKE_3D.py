"""
VortexGeometry.py  —  AE4135 Rotor/Wake Aerodynamics, Assignment 2
Interactive 3-D visualisation of the frozen vortex wake geometry.

Run:  python VortexGeometry.py
      → opens vortex_geometry.html in your browser
      → saves vortex_geometry_report.pdf  (publication-quality static figure)

Requires:  numpy  plotly  matplotlib  seaborn
           pip install plotly matplotlib seaborn
"""

import os
import numpy as np
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

# ── Visualisation parameters ──────────────────────────────────────────────────
TSR      = 8       # tip-speed ratio
N        = 20      # spanwise panels per blade
N_wake   = 2       # number of full wake rotations shown
dpsi_deg = 90.0    # azimuthal step [deg]
a_w      = 0.25    # frozen-wake axial induction (convection factor)

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_HTML   = "vortex_geometry.html"
OUTPUT_REPORT = "vortex_geometry_report.pdf"

# ── Seaborn colorblind palette (first 3 entries for 3 blades) ────────────────
# Blue, Orange, Green — all clearly distinguishable
CB_PALETTE = sns.color_palette("colorblind")
BLADE_COLORS_MPL = [CB_PALETTE[0], CB_PALETTE[1], CB_PALETTE[2]]
# Convert to 0-255 integers for Plotly rgba strings
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

def make_panels(N, distribution="cosine"):
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
# VORTEX SYSTEM BUILDER
# =============================================================================

def build_vortex_system(Omega, N, N_wake, dpsi_deg, a_w):
    r_edges, r_centers, dr = make_panels(N)
    U_wake  = U0 * (1.0 - a_w)
    dpsi    = np.radians(dpsi_deg)
    psi_arr = np.arange(0.0, N_wake * 2.0 * np.pi + dpsi * 0.5, dpsi)

    controlpoints = []
    rings         = []

    for k_blade in range(NBlades):
        angle_rot  = 2.0 * np.pi / NBlades * k_blade
        cosR, sinR = np.cos(angle_rot), np.sin(angle_rot)

        for i in range(N):
            r   = r_centers[i]
            r_R = r / Radius

            controlpoints.append({
                'r'        : r,
                'r_R'      : r_R,
                'chord'    : blade_chord(r_R),
                'twist_rad': np.radians(blade_twist(r_R)),
                'coords'   : np.array([0.0, r * cosR, r * sinR]),
                'dr'       : dr[i],
            })

            filaments = []

            y_in  = r_edges[i]   * cosR;  z_in  = r_edges[i]   * sinR
            y_out = r_edges[i+1] * cosR;  z_out = r_edges[i+1] * sinR
            filaments.append({
                'x1': 0.0, 'y1': y_in,  'z1': z_in,
                'x2': 0.0, 'y2': y_out, 'z2': z_out,
            })

            for j in range(len(psi_arr) - 1):
                t1 = psi_arr[j]   / Omega;  t2 = psi_arr[j+1] / Omega
                ri = r_edges[i]
                x1 = U_wake * t1;  y1 = ri * np.cos(-Omega*t1);  z1 = ri * np.sin(-Omega*t1)
                x2 = U_wake * t2;  y2 = ri * np.cos(-Omega*t2);  z2 = ri * np.sin(-Omega*t2)
                y1r = y1*cosR - z1*sinR;  z1r = y1*sinR + z1*cosR
                y2r = y2*cosR - z2*sinR;  z2r = y2*sinR + z2*cosR
                filaments.append({'x1':x2,'y1':y2r,'z1':z2r, 'x2':x1,'y2':y1r,'z2':z1r})

            for j in range(len(psi_arr) - 1):
                t1 = psi_arr[j]   / Omega;  t2 = psi_arr[j+1] / Omega
                ro = r_edges[i+1]
                x1 = U_wake * t1;  y1 = ro * np.cos(-Omega*t1);  z1 = ro * np.sin(-Omega*t1)
                x2 = U_wake * t2;  y2 = ro * np.cos(-Omega*t2);  z2 = ro * np.sin(-Omega*t2)
                y1r = y1*cosR - z1*sinR;  z1r = y1*sinR + z1*cosR
                y2r = y2*cosR - z2*sinR;  z2r = y2*sinR + z2*cosR
                filaments.append({'x1':x1,'y1':y1r,'z1':z1r, 'x2':x2,'y2':y2r,'z2':z2r})

            rings.append(filaments)

    return controlpoints, rings

# =============================================================================
# INTERACTIVE PLOTLY FIGURE  (original + colorblind colours)
# =============================================================================

def plot_vortex_geometry():

    Omega       = U0 * TSR / Radius
    cps, rings  = build_vortex_system(Omega, N, N_wake, dpsi_deg, a_w)
    N_per_blade = len(cps) // NBlades
    n_ws        = (len(rings[0]) - 1) // 2

    def rgba(rgb, a):
        return f'rgba({rgb[0]},{rgb[1]},{rgb[2]},{a})'

    def lighten(rgb, f):
        return tuple(int(c + (255 - c) * f) for c in rgb)

    def seg_trace(segs, color, width, name, lg, show_legend, dash=None):
        xs, ys, zs = [], [], []
        for f in segs:
            xs += [f['x1'], f['x2'], None]
            ys += [f['y1'], f['y2'], None]
            zs += [f['z1'], f['z2'], None]
        ld = dict(color=color, width=width)
        if dash:
            ld['dash'] = dash
        return go.Scatter3d(x=xs, y=ys, z=zs, mode='lines', line=ld,
                            name=name, legendgroup=lg,
                            showlegend=show_legend, hoverinfo='skip')

    traces = []

    for b in range(NBlades):
        rgb   = BLADE_COLORS_PLY[b]
        bname = f'Blade {b + 1}'
        angle = 2.0 * np.pi / NBlades * b
        cosR, sinR = np.cos(angle), np.sin(angle)

        bound_s, inner_s, outer_s = [], [], []
        for p in range(N_per_blade):
            ring = rings[b * N_per_blade + p]
            bound_s.append(ring[0])
            inner_s.extend(ring[1        : n_ws + 1])
            outer_s.extend(ring[n_ws + 1 : 2*n_ws + 1])

        _, r_arr, _ = make_panels(N=40)
        le_x, le_y, le_z = [], [], []
        te_x, te_y, te_z = [], [], []
        chord_xs, chord_ys, chord_zs = [], [], []
        step = max(1, len(r_arr) // 12)

        for idx, r in enumerate(r_arr):
            r_R   = r / Radius
            chord = blade_chord(r_R)
            twist = np.radians(blade_twist(r_R))
            ry = r * cosR;  rz = r * sinR
            le_x.append( (chord/2) * np.sin(twist));  le_y.append(ry);  le_z.append(rz - (chord/2)*np.cos(twist))
            te_x.append(-(chord/2) * np.sin(twist));  te_y.append(ry);  te_z.append(rz + (chord/2)*np.cos(twist))
            if idx % step == 0:
                chord_xs += [le_x[-1], te_x[-1], None]
                chord_ys += [le_y[-1], te_y[-1], None]
                chord_zs += [le_z[-1], te_z[-1], None]

        traces.append(go.Scatter3d(x=le_x, y=le_y, z=le_z, mode='lines',
            line=dict(color=rgba(rgb, 0.85), width=3),
            name=f'{bname} LE', legendgroup=f'bl{b}', showlegend=True, hoverinfo='skip'))
        traces.append(go.Scatter3d(x=te_x, y=te_y, z=te_z, mode='lines',
            line=dict(color=rgba(rgb, 0.30), width=1.5),
            name=f'{bname} TE', legendgroup=f'bl{b}', showlegend=False, hoverinfo='skip'))
        traces.append(go.Scatter3d(x=chord_xs, y=chord_ys, z=chord_zs, mode='lines',
            line=dict(color=rgba(rgb, 0.20), width=1),
            name=f'{bname} chord', legendgroup=f'bl{b}', showlegend=False, hoverinfo='skip'))

        traces.append(seg_trace(bound_s, rgba(rgb, 1.0), 5,
                                f'{bname} bound vortex', f'bv{b}', True))
        traces.append(seg_trace(inner_s, rgba(rgb, 0.55), 1.5,
                                f'{bname} inner trailing', f'it{b}', True))
        traces.append(seg_trace(outer_s, rgba(lighten(rgb, 0.40), 0.35), 1.0,
                                f'{bname} outer trailing', f'ot{b}', True, dash='dot'))

        cpx   = [cps[b*N_per_blade+p]['coords'][0] for p in range(N_per_blade)]
        cpy   = [cps[b*N_per_blade+p]['coords'][1] for p in range(N_per_blade)]
        cpz   = [cps[b*N_per_blade+p]['coords'][2] for p in range(N_per_blade)]
        htxt  = [f"r/R = {cps[b*N_per_blade+p]['r_R']:.3f}<br>"
                 f"chord = {blade_chord(cps[b*N_per_blade+p]['r_R']):.2f} m"
                 for p in range(N_per_blade)]
        traces.append(go.Scatter3d(x=cpx, y=cpy, z=cpz, mode='markers',
            marker=dict(size=6, color=rgba(rgb, 1.0), symbol='circle',
                        line=dict(color='white', width=1.5)),
            name=f'{bname} control pts', legendgroup=f'cp{b}', showlegend=True,
            text=htxt, hovertemplate='<b>%{text}</b><extra></extra>'))

    theta = np.linspace(0.0, 2.0*np.pi, 180)
    traces.append(go.Scatter3d(
        x=np.zeros(180), y=Radius*np.cos(theta), z=Radius*np.sin(theta),
        mode='lines', line=dict(color='rgba(200,200,200,0.18)', width=1),
        name='Rotor disc', legendgroup='disc', showlegend=True, hoverinfo='skip'))

    r_root_m = RootLocation_R * Radius
    traces.append(go.Scatter3d(
        x=np.zeros(180), y=r_root_m*np.cos(theta), z=r_root_m*np.sin(theta),
        mode='lines', line=dict(color='rgba(200,200,200,0.10)', width=1),
        name='Root circle', legendgroup='disc', showlegend=False, hoverinfo='skip'))

    for ang in np.linspace(0, 2*np.pi, 7)[:-1]:
        traces.append(go.Scatter3d(
            x=[0, 0],
            y=[r_root_m*np.cos(ang), Radius*np.cos(ang)],
            z=[r_root_m*np.sin(ang), Radius*np.sin(ang)],
            mode='lines', line=dict(color='rgba(200,200,200,0.08)', width=0.5),
            showlegend=False, hoverinfo='skip'))

    x_max = max(f['x2'] for ring in rings for f in ring)
    traces.append(go.Scatter3d(
        x=[-8, x_max*1.05], y=[0, 0], z=[0, 0], mode='lines',
        line=dict(color='rgba(200,200,200,0.15)', width=1, dash='dot'),
        name='Rotor axis', legendgroup='axis', showlegend=True, hoverinfo='skip'))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text=(f'Frozen vortex wake  \u00b7  {NBlades} blades  \u00b7  '
                  f'N\u2009=\u2009{N} panels/blade  \u00b7  '
                  f'{N_wake} wake rotations  \u00b7  '
                  f'TSR\u2009=\u2009{TSR}  \u00b7  '
                  f'\u0394\u03c8\u2009=\u2009{dpsi_deg}\u00b0'),
            font=dict(size=12, color='#cccccc'),
            x=0.5, xanchor='center',
        ),
        scene=dict(
            xaxis=dict(title='x \u2014 axial [m]', showgrid=True,
                       gridcolor='rgba(255,255,255,0.07)', color='#888',
                       backgroundcolor='rgba(0,0,0,0)', showbackground=True),
            yaxis=dict(title='y [m]', showgrid=True,
                       gridcolor='rgba(255,255,255,0.07)', color='#888',
                       backgroundcolor='rgba(0,0,0,0)', showbackground=True),
            zaxis=dict(title='z [m]', showgrid=True,
                       gridcolor='rgba(255,255,255,0.07)', color='#888',
                       backgroundcolor='rgba(0,0,0,0)', showbackground=True),
            bgcolor='#0f0f0f',
            aspectmode='data',
            camera=dict(eye=dict(x=1.3, y=0.75, z=0.55),
                        up=dict(x=0, y=0, z=1)),
        ),
        paper_bgcolor='#0f0f0f',
        font=dict(color='#cccccc', size=11, family='Arial'),
        legend=dict(
            bgcolor='rgba(20,20,20,0.88)',
            bordercolor='rgba(255,255,255,0.12)',
            borderwidth=1,
            tracegroupgap=2,
            font=dict(size=10),
            itemsizing='constant',
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        height=750,
    )

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_HTML)
    fig.write_html(out_path)
    print(f"Saved  \u2192  {out_path}")
    fig.show()
    return fig


# =============================================================================
# STATIC REPORT FIGURE  (matplotlib, white background, colorblind palette)
# =============================================================================

def plot_report_figure():
    """
    Publication-quality 3-D view using matplotlib's Axes3D.
    White background, seaborn colorblind palette, legend panel beside the plot.
    Front-of-rotor view: rotor disc visible on the right, wake trailing to the left.
    Saved as high-res PDF.
    """
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

    Omega       = U0 * TSR / Radius
    cps, rings  = build_vortex_system(Omega, N, N_wake, dpsi_deg, a_w)
    N_per_blade = len(cps) // NBlades
    n_ws        = (len(rings[0]) - 1) // 2

    fig = plt.figure(figsize=(12, 7))
    ax        = fig.add_axes([0.0, 0.0, 0.77, 1.0], projection='3d')
    ax_legend = fig.add_axes([0.76, 0.12, 0.24, 0.76])
    ax_legend.axis('off')
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    pane_col = (0.93, 0.93, 0.93, 0.50)
    ax.xaxis.set_pane_color(pane_col)
    ax.yaxis.set_pane_color(pane_col)
    ax.zaxis.set_pane_color(pane_col)
    ax.xaxis._axinfo['grid']['color'] = (0.80, 0.80, 0.80, 0.50)
    ax.yaxis._axinfo['grid']['color'] = (0.80, 0.80, 0.80, 0.50)
    ax.zaxis._axinfo['grid']['color'] = (0.80, 0.80, 0.80, 0.50)

    legend_handles = []

    for b in range(NBlades):
        col   = BLADE_COLORS_MPL[b]
        angle = 2.0 * np.pi / NBlades * b
        cosR, sinR = np.cos(angle), np.sin(angle)

        # ── blade outline (LE + TE + caps + chord ticks) ──────────────────
        _, r_arr, _ = make_panels(N=80)
        le_pts, te_pts = [], []
        for r in r_arr:
            r_R   = r / Radius
            chord = blade_chord(r_R)
            twist = np.radians(blade_twist(r_R))
            ry = r * cosR;  rz = r * sinR
            le_pts.append([(chord/2)*np.sin(twist), ry, rz-(chord/2)*np.cos(twist)])
            te_pts.append([-(chord/2)*np.sin(twist), ry, rz+(chord/2)*np.cos(twist)])
        le_pts = np.array(le_pts);  te_pts = np.array(te_pts)

        ax.plot(le_pts[:,0], le_pts[:,1], le_pts[:,2], color=col, lw=2.2, alpha=1.0,  zorder=8)
        ax.plot(te_pts[:,0], te_pts[:,1], te_pts[:,2], color=col, lw=1.2, alpha=0.60, zorder=8)
        ax.plot([le_pts[-1,0],te_pts[-1,0]], [le_pts[-1,1],te_pts[-1,1]],
                [le_pts[-1,2],te_pts[-1,2]], color=col, lw=1.0, alpha=0.60)
        ax.plot([le_pts[0,0], te_pts[0,0]], [le_pts[0,1], te_pts[0,1]],
                [le_pts[0,2], te_pts[0,2]], color=col, lw=1.0, alpha=0.60)
        step = max(1, len(r_arr) // 8)
        for idx in range(0, len(r_arr), step):
            ax.plot([le_pts[idx,0],te_pts[idx,0]],
                    [le_pts[idx,1],te_pts[idx,1]],
                    [le_pts[idx,2],te_pts[idx,2]],
                    color=col, lw=0.5, alpha=0.18)

        # ── bound vortex (thick solid) ────────────────────────────────────
        bv_xs, bv_ys, bv_zs = [], [], []
        for p in range(N_per_blade):
            f = rings[b*N_per_blade+p][0]
            bv_xs += [f['x1'],f['x2']];  bv_ys += [f['y1'],f['y2']];  bv_zs += [f['z1'],f['z2']]
        ax.plot(bv_xs, bv_ys, bv_zs, color=col, lw=4.0, solid_capstyle='round', zorder=10)

        # ── inner trailing vortex (medium solid) ──────────────────────────
        for p in range(N_per_blade):
            for f in rings[b*N_per_blade+p][1 : n_ws+1]:
                ax.plot([f['x1'],f['x2']], [f['y1'],f['y2']], [f['z1'],f['z2']],
                        color=col, lw=0.9, alpha=0.65, zorder=4)

        # ── outer trailing vortex (light dashed) ──────────────────────────
        light_col = tuple(min(1.0, c*0.60+0.40) for c in col[:3])
        for p in range(N_per_blade):
            for f in rings[b*N_per_blade+p][n_ws+1 : 2*n_ws+1]:
                ax.plot([f['x1'],f['x2']], [f['y1'],f['y2']], [f['z1'],f['z2']],
                        color=light_col, lw=0.65, alpha=0.50,
                        linestyle='--', dashes=(5, 4), zorder=3)

        legend_handles.append(Line2D([0],[0], color=col, lw=2.5, label=f'Blade {b+1}'))

    # ── rotor disc reference ───────────────────────────────────────────────
    theta    = np.linspace(0, 2*np.pi, 360)
    r_root_m = RootLocation_R * Radius
    ax.plot(np.zeros(360), Radius*np.cos(theta), Radius*np.sin(theta),
            color='#999999', lw=1.2, alpha=0.55)
    ax.plot(np.zeros(360), r_root_m*np.cos(theta), r_root_m*np.sin(theta),
            color='#999999', lw=0.7, alpha=0.40)
    for ang in np.linspace(0, 2*np.pi, 7)[:-1]:
        ax.plot([0, 0],
                [r_root_m*np.cos(ang), Radius*np.cos(ang)],
                [r_root_m*np.sin(ang), Radius*np.sin(ang)],
                color='#cccccc', lw=0.4, alpha=0.35)

    # ── rotor axis ────────────────────────────────────────────────────────
    x_max = max(f['x2'] for ring in rings for f in ring)
    ax.plot([-8, x_max*1.02], [0, 0], [0, 0],
            color='#888888', lw=0.9, linestyle=':', alpha=0.65)

    # ── view: front of rotor facing viewer, wake trailing to the left ─────
    ax.view_init(elev=22, azim=118)

    # ── axis labels (fontsize from rcParams) ──────────────────────────────
    ax.set_xlabel('x \u2014 axial [m]', labelpad=10)
    ax.set_ylabel('y [m]',              labelpad=10)
    ax.set_zlabel('z [m]',              labelpad=8)

    # ── legend panel ──────────────────────────────────────────────────────
    legend_handles += [
        Line2D([0],[0], color='white',   lw=0,   label=' '),
        Line2D([0],[0], color='#333333', lw=0,   label='Line types:'),
        Line2D([0],[0], color='#444444', lw=3.8, label='Bound vortex'),
        Line2D([0],[0], color='#666666', lw=1.0, alpha=0.8,
               label='Inner trailing vortex'),
        Line2D([0],[0], color='#aaaaaa', lw=0.9, linestyle='--',
               dashes=(5, 4), label='Outer trailing vortex'),
        Line2D([0],[0], color='#999999', lw=1.0, label='Rotor disc / axis'),
    ]
    leg = ax_legend.legend(handles=legend_handles, loc='center',
                           framealpha=0.97, edgecolor='#cccccc',
                           handlelength=2.6, labelspacing=0.55,
                           title='Vortex wake geometry', title_fontsize=11)
    leg.get_frame().set_linewidth(0.7)
    leg.get_title().set_fontweight('semibold')

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_REPORT)
    fig.savefig(out_path, dpi=300, facecolor='white', edgecolor='none')
    print(f"Saved  \u2192  {out_path}")
    plt.close(fig)
    return out_path


# =============================================================================
if __name__ == "__main__":
    # 1) static report figure (always runs, no display needed)
    plot_report_figure()
    # 2) interactive Plotly figure
    plot_vortex_geometry()