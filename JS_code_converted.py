"""
bem_liftingline.py
==================
AE4135 Lifting Line Group Assignment — Wind Turbine (DU 95-W-180 airfoil)

Solves:
  • BEM model
  • Frozen-vortex Lifting Line model
for the assignment rotor at TSR = 6, 8, 10  (U0=10 m/s, R=50 m, 3 blades)

Produces all required plots:
  1. Radial distribution of inflow angle phi and angle of attack alpha
  2. Radial distribution of axial (a) and tangential (a') induction
  3. Radial distribution of axial and tangential loading (Fnorm, Ftan)
  4. Radial distribution of circulation Gamma
  5. CT and CP comparison table / bar chart (BEM vs LL)
  6. Sensitivity: convection speed (wake pitch factor)
  7. Sensitivity: blade discretisation (constant vs cosine)
  8. Sensitivity: azimuthal discretisation (segments per rotation)
  9. Sensitivity: wake length (number of rotations) + convergence

Polar loaded directly from:  polar_DU95W180__3_.xlsx
Place the xlsx file in the same directory as this script.

Dependencies: numpy, pandas, matplotlib, openpyxl
"""

import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── locate the polar file ────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_POLAR_CANDIDATES = [
    os.path.join(_SCRIPT_DIR, "polar_DU95W180__3_.xlsx"),
    "/mnt/user-data/uploads/polar_DU95W180__3_.xlsx",
    "polar_DU95W180__3_.xlsx",
]

# ============================================================
#  ROTOR / ASSIGNMENT PARAMETERS
# ============================================================
RADIUS   = 50.0   # m
NBLADES  = 3
U0       = 10.0   # m/s  free-stream wind speed
TSR_LIST = [6.0, 8.0, 10.0]
ROOT_R   = 0.2    # blade root starts at r/R = 0.2

# ============================================================
#  POLAR LOADING
# ============================================================

def load_polar(path):
    """
    Read alpha, Cl, Cd from the DU95W180 Excel polar (sheet 'Blad1').
    Returns (alpha_list, cl_list, cd_list).
    """
    df = pd.read_excel(path, sheet_name="Blad1", header=None)
    header_row = None
    for i, row in df.iterrows():
        if str(row[0]).strip() == "Alfa":
            header_row = i
            break
    if header_row is None:
        raise ValueError("Could not find 'Alfa' header in polar file.")
    data = df.iloc[header_row + 1:].reset_index(drop=True)
    data = data.apply(pd.to_numeric, errors="coerce").dropna()
    return data[0].tolist(), data[1].tolist(), data[2].tolist()


_POLAR_PATH = None
for _p in _POLAR_CANDIDATES:
    if os.path.exists(_p):
        _POLAR_PATH = _p
        break
if _POLAR_PATH is None:
    raise FileNotFoundError(
        "polar_DU95W180__3_.xlsx not found. "
        "Place it in the same directory as this script."
    )

_POLAR_ALPHA, _POLAR_CL, _POLAR_CD = load_polar(_POLAR_PATH)
print(f"Polar loaded: {_POLAR_PATH}  "
      f"({len(_POLAR_ALPHA)} pts, "
      f"alpha = {_POLAR_ALPHA[0]:.1f} to {_POLAR_ALPHA[-1]:.1f} deg)")

# ============================================================
#  UTILITIES
# ============================================================

def create_array_sequence(start, delta, end):
    data, v = [], start
    while v <= end - delta:
        data.append(v)
        v += delta
    data.append(end)
    return data


def interp1d(xarr, yarr, xnew):
    if xnew <= xarr[0]:  return yarr[0]
    if xnew >= xarr[-1]: return yarr[-1]
    for i in range(len(xarr) - 1):
        if xarr[i] <= xnew <= xarr[i + 1]:
            t = (xnew - xarr[i]) / (xarr[i + 1] - xarr[i])
            return yarr[i] + t * (yarr[i + 1] - yarr[i])
    return yarr[-1]


def polar_airfoil(alpha_deg):
    """Return (Cl, Cd) interpolated from DU 95-W-180 polar."""
    cl = interp1d(_POLAR_ALPHA, _POLAR_CL, alpha_deg)
    cd = interp1d(_POLAR_ALPHA, _POLAR_CD, alpha_deg)
    return cl, cd

# ============================================================
#  BLADE GEOMETRY  (assignment spec)
# ============================================================

def geo_blade(r_R):
    """
    Assignment geometry (for r/R > 0.2):
      chord(r/R) = 3*(1-r/R) + 1  [m]
      twist(r/R) = 14*(1-r/R)     [deg, nose-up]
      pitch      = -2              [deg, collective]
    Returns (chord [m], local_pitch [deg]).
    """
    chord = 3.0 * (1.0 - r_R) + 1.0
    twist = 14.0 * (1.0 - r_R)
    pitch = -2.0
    return chord, twist + pitch

# ============================================================
#  BLADE ELEMENT LOADS
# ============================================================

def load_blade_element(Vnorm, Vtan, r_R):
    """
    2-D blade element aerodynamics.
    Returns (Fnorm, Ftan, Gamma, alpha_deg, phi_deg).
    """
    # clamp velocities to avoid overflow on degenerate wake panels
    Vnorm = max(min(Vnorm,  1e6), -1e6)
    Vtan  = max(min(Vtan,   1e6), -1e6)
    Vmag2        = Vnorm**2 + Vtan**2
    phi          = math.atan2(Vnorm, Vtan)          # inflow angle [rad]
    chord, beta  = geo_blade(r_R)
    alpha_deg    = math.degrees(phi) - beta          # AoA = phi - pitch

    cl, cd = polar_airfoil(alpha_deg)

    Lift  = 0.5 * Vmag2 * cl * chord
    Drag  = 0.5 * Vmag2 * cd * chord
    Fnorm = Lift * math.cos(phi) + Drag * math.sin(phi)
    Ftan  = Lift * math.sin(phi) - Drag * math.cos(phi)
    Gamma = 0.5 * math.sqrt(Vmag2) * cl * chord
    return Fnorm, Ftan, Gamma, alpha_deg, math.degrees(phi)

# ============================================================
#  BEM — Glauert + Prandtl corrections
# ============================================================

def _glauert_a_from_CT(CT):
    CT1 = 1.816
    CT2 = 2.0 * math.sqrt(CT1) - CT1
    if CT < CT2:
        return 0.5 - 0.5 * math.sqrt(max(1.0 - CT, 0.0))
    return 1.0 + (CT - CT1) / (4.0 * (math.sqrt(CT1) - 1.0))


def _prandtl_correction(r_R, root_R, tip_R, TSR_rotor, NB, a):
    sqt = math.sqrt(1.0 + (TSR_rotor * r_R)**2 / max((1.0 - a)**2, 1e-12))
    Ft  = 2/math.pi * math.acos(
              min(math.exp(-NB/2 * (tip_R - r_R) / max(r_R, 1e-9) * sqt), 1.0))
    Fr  = 2/math.pi * math.acos(
              min(math.exp( NB/2 * (root_R - r_R) / max(r_R, 1e-9) * sqt), 1.0))
    return max(Ft * Fr, 1e-4)


def solve_streamtube(Uinf, r1_R, r2_R, root_R, tip_R, Omega, Radius, NB):
    r_R   = (r1_R + r2_R) / 2.0
    Area  = math.pi * ((r2_R * Radius)**2 - (r1_R * Radius)**2)
    a     = 0.3
    aline = 0.0
    for _ in range(200):
        Urotor = Uinf * (1.0 - a)
        Utan   = (1.0 + aline) * Omega * r_R * Radius
        Fn, Ft, Gam, alpha, phi = load_blade_element(Urotor, Utan, r_R)
        CT   = Fn * Radius * (r2_R - r1_R) * NB / (0.5 * Area * Uinf**2)
        anew = _glauert_a_from_CT(CT)
        F    = _prandtl_correction(r_R, root_R, tip_R,
                                   Omega * Radius / Uinf, NB, anew)
        anew  /= F
        aline  = (Ft * NB /
                  (2*math.pi * Uinf*(1-a) * Omega * 2*(r_R*Radius)**2)) / F
        if abs(a - anew) < 1e-6:
            a = anew
            break
        a = 0.75*a + 0.25*anew
    return a, aline, r_R, Fn, Ft, Gam, alpha, phi


def solve_BEM(Uinf, r_R_array, Omega, Radius, NB):
    keys = ('a','aline','r_R','Fnorm','Ftan','Gamma','alpha','phi')
    out  = {k: [] for k in keys}
    for i in range(len(r_R_array) - 1):
        vals = solve_streamtube(Uinf, r_R_array[i], r_R_array[i+1],
                                r_R_array[0], r_R_array[-1],
                                Omega, Radius, NB)
        for k, v in zip(keys, vals):
            out[k].append(v)
    return out


def CT_CP_BEM(res, r_R_array, Uinf, Omega, Radius, NB):
    CT = CP = 0.0
    for i in range(len(r_R_array) - 1):
        dr = r_R_array[i+1] - r_R_array[i]
        rr = (r_R_array[i] + r_R_array[i+1]) / 2.0
        CT += dr * res['Fnorm'][i] * NB / (0.5 * Uinf**2 * math.pi * Radius)
        CP += dr * res['Ftan'][i]  * rr * Omega * NB / (0.5 * Uinf**3 * math.pi)
    return CT, CP

# ============================================================
#  BIOT-SAVART  (Katz & Plotkin)
# ============================================================

def vel_vortex_filament(GAMMA, XV1, XV2, XVP, CORE=1e-5):
    X1,Y1,Z1 = XV1;  X2,Y2,Z2 = XV2;  XP,YP,ZP = XVP
    R1 = math.sqrt((XP-X1)**2+(YP-Y1)**2+(ZP-Z1)**2)
    R2 = math.sqrt((XP-X2)**2+(YP-Y2)**2+(ZP-Z2)**2)
    RX = (YP-Y1)*(ZP-Z2)-(ZP-Z1)*(YP-Y2)
    RY =-(XP-X1)*(ZP-Z2)+(ZP-Z1)*(XP-X2)
    RZ = (XP-X1)*(YP-Y2)-(YP-Y1)*(XP-X2)
    RSQ = RX**2+RY**2+RZ**2
    R0R1=(X2-X1)*(XP-X1)+(Y2-Y1)*(YP-Y1)+(Z2-Z1)*(ZP-Z1)
    R0R2=(X2-X1)*(XP-X2)+(Y2-Y1)*(YP-Y2)+(Z2-Z1)*(ZP-Z2)
    if RSQ < CORE**2: RSQ = CORE**2
    if R1  < CORE:    R1  = CORE
    if R2  < CORE:    R2  = CORE
    K = GAMMA / (4*math.pi*RSQ) * (R0R1/R1 - R0R2/R2)
    return [K*RX, K*RY, K*RZ]


def vel_ring(ring, cp):
    v = [0.0, 0.0, 0.0]
    for f in ring['filaments']:
        dv = vel_vortex_filament(
            f['Gamma'],
            [f['x1'],f['y1'],f['z1']],
            [f['x2'],f['y2'],f['z2']], cp)
        v[0]+=dv[0]; v[1]+=dv[1]; v[2]+=dv[2]
    return v


def set_ring_gamma(ring, g):
    for f in ring['filaments']:
        f['Gamma'] = g
    return ring

# ============================================================
#  ROTOR GEOMETRY BUILDER
# ============================================================

def create_rotor_geometry(span_array, radius, tsr_wake, theta_array, nblades):
    """
    Build frozen helical wake + blade control points.
    tsr_wake controls the axial spacing (helical pitch) of the wake:
      dx = dtheta / tsr_wake * radius
    """
    controlpoints, rings, bladepanels = [], [], []

    for krot in range(nblades):
        ang_rot = 2*math.pi / nblades * krot
        cr, sr  = math.cos(ang_rot), math.sin(ang_rot)

        for i in range(len(span_array)-1):
            r = (span_array[i]+span_array[i+1]) / 2.0
            chord, tp = geo_blade(r / radius)
            ang = math.radians(tp)

            # control point at mid-panel, rotated to blade position
            cp = {
                'coordinates': [0.0, r*cr, r*sr],
                'chord': chord,
                'normal':     [ math.cos(ang), 0.0, -math.sin(ang)],
                'tangential': [-math.sin(ang), 0.0, -math.cos(ang)],
            }
            controlpoints.append(cp)

            fils = []

            # --- bound vortex ---
            y1b = span_array[i]*cr;   z1b = span_array[i]*sr
            y2b = span_array[i+1]*cr; z2b = span_array[i+1]*sr
            fils.append({'x1':0,'y1':y1b,'z1':z1b,
                         'x2':0,'y2':y2b,'z2':z2b, 'Gamma':0.0})

            # --- inner trailing vortex ---
            c_i, tp_i = geo_blade(span_array[i] / radius)
            ai = math.radians(tp_i)
            x0i = c_i*math.sin(-ai)
            fils.append({'x1':x0i, 'y1':y1b, 'z1':z1b,
                         'x2':0.0, 'y2':y1b, 'z2':z1b, 'Gamma':0.0})
            for j in range(len(theta_array)-1):
                xt=fils[-1]['x1']; yt=fils[-1]['y1']; zt=fils[-1]['z1']
                dy_w=(math.cos(-theta_array[j+1])-math.cos(-theta_array[j]))*span_array[i]
                dz_w=(math.sin(-theta_array[j+1])-math.sin(-theta_array[j]))*span_array[i]
                dx  =(theta_array[j+1]-theta_array[j])/tsr_wake*radius
                dy_r= dy_w*cr - dz_w*sr
                dz_r= dy_w*sr + dz_w*cr
                fils.append({'x1':xt+dx,'y1':yt+dy_r,'z1':zt+dz_r,
                             'x2':xt,   'y2':yt,       'z2':zt, 'Gamma':0.0})

            # --- outer trailing vortex ---
            c_o, tp_o = geo_blade(span_array[i+1] / radius)
            ao = math.radians(tp_o)
            x0o = c_o*math.sin(-ao)
            fils.append({'x1':0.0, 'y1':y2b, 'z1':z2b,
                         'x2':x0o, 'y2':y2b, 'z2':z2b, 'Gamma':0.0})
            for j in range(len(theta_array)-1):
                xt=fils[-1]['x2']; yt=fils[-1]['y2']; zt=fils[-1]['z2']
                dy_w=(math.cos(-theta_array[j+1])-math.cos(-theta_array[j]))*span_array[i+1]
                dz_w=(math.sin(-theta_array[j+1])-math.sin(-theta_array[j]))*span_array[i+1]
                dx  =(theta_array[j+1]-theta_array[j])/tsr_wake*radius
                dy_r= dy_w*cr - dz_w*sr
                dz_r= dy_w*sr + dz_w*cr
                fils.append({'x1':xt,    'y1':yt,       'z1':zt,
                             'x2':xt+dx, 'y2':yt+dy_r, 'z2':zt+dz_r,
                             'Gamma':0.0})

            rings.append({'filaments': fils})
            bladepanels.append({'r_in': span_array[i], 'r_out': span_array[i+1]})

    return {'controlpoints': controlpoints, 'rings': rings,
            'bladepanels': bladepanels}

# ============================================================
#  LIFTING LINE SOLVER
# ============================================================

def solve_lifting_line(rotor_wake, wind, Omega, R,
                       Niter=1200, tol=0.01, conv_w=0.3):
    """
    Iterative matrix lifting-line solver for a HAWT rotor.
    Returns dict: a, aline, r_R, Fnorm, Ftan, Gamma, alpha, phi
    """
    cps   = rotor_wake['controlpoints']
    rings = rotor_wake['rings']
    N     = len(cps)

    # set unit Gamma in all rings for matrix assembly
    for j in range(N):
        rings[j] = set_ring_gamma(rings[j], 1.0)

    # pre-compute induction matrices
    MU = [[0.0]*N for _ in range(N)]
    MV = [[0.0]*N for _ in range(N)]
    MW = [[0.0]*N for _ in range(N)]
    for icp in range(N):
        for jr in range(N):
            vi = vel_ring(rings[jr], cps[icp]['coordinates'])
            MU[icp][jr]=vi[0]; MV[icp][jr]=vi[1]; MW[icp][jr]=vi[2]

    # output arrays
    GammaNew = [0.0]*N
    Gamma    = [0.0]*N
    a_o=[0.]*N; al_o=[0.]*N; rR_o=[0.]*N
    Fn_o=[0.]*N; Ft_o=[0.]*N; G_o=[0.]*N
    alp_o=[0.]*N; phi_o=[0.]*N

    for _ in range(Niter):
        Gamma = GammaNew[:]

        for icp in range(N):
            coords = cps[icp]['coordinates']
            r_pos  = math.sqrt(sum(c**2 for c in coords))

            u = sum(MU[icp][j]*Gamma[j] for j in range(N))
            v = sum(MV[icp][j]*Gamma[j] for j in range(N))
            w = sum(MW[icp][j]*Gamma[j] for j in range(N))

            vrot   = np.cross([-Omega, 0, 0], coords).tolist()
            vel1   = [wind[0]+u+vrot[0],
                      wind[1]+v+vrot[1],
                      wind[2]+w+vrot[2]]
            azim   = np.cross([-1/r_pos, 0, 0], coords).tolist()
            vazim  = float(np.dot(azim, vel1))
            vaxial = float(np.dot([1, 0, 0], vel1))

            Fn,Ft,Gn,alp,phi = load_blade_element(vaxial, vazim, r_pos/R)
            GammaNew[icp] = Gn

            a_o[icp]   = -(u+vrot[0]) / wind[0]
            al_o[icp]  = vazim / (r_pos*Omega) - 1.0
            rR_o[icp]  = r_pos / R
            Fn_o[icp]  = Fn;  Ft_o[icp]  = Ft
            G_o[icp]   = Gn
            alp_o[icp] = alp; phi_o[icp] = phi

        ref  = max(max(abs(g) for g in GammaNew), 1e-3)
        err  = max(abs(GammaNew[i]-Gamma[i]) for i in range(N)) / ref
        if err < tol:
            break
        GammaNew = [(1-conv_w)*Gamma[i]+conv_w*GammaNew[i] for i in range(N)]

    return {'a':a_o,'aline':al_o,'r_R':rR_o,
            'Fnorm':Fn_o,'Ftan':Ft_o,'Gamma':G_o,
            'alpha':alp_o,'phi':phi_o}


def CT_CP_LL(res, r_R_array, Uinf, Omega, Radius, NB):
    """Integrate LL results over first blade, multiply by NB."""
    n  = len(r_R_array) - 1   # panels per blade (first blade = indices 0..n-1)
    CT = CP = 0.0
    for i in range(n):
        dr = r_R_array[i+1] - r_R_array[i]
        rr = (r_R_array[i]+r_R_array[i+1]) / 2.0
        CT += dr * res['Fnorm'][i] * NB / (0.5*Uinf**2*math.pi*Radius)
        CP += dr * res['Ftan'][i]  * rr * Omega * NB / (0.5*Uinf**3*math.pi)
    return CT, CP

# ============================================================
#  DISCRETISATION HELPERS
# ============================================================

def cosine_r_array(N, r_root=ROOT_R, r_tip=1.0):
    theta = create_array_sequence(0.0, math.pi/N, math.pi)
    return [r_root + (r_tip-r_root)*(1-math.cos(t))/2.0 for t in theta]


def uniform_r_array(N, r_root=ROOT_R, r_tip=1.0):
    return [r_root + (r_tip-r_root)*i/N for i in range(N+1)]


def build_wake_theta(segs_per_rot, Nrotations):
    return create_array_sequence(0.0,
                                 2*math.pi/segs_per_rot,
                                 Nrotations*2*math.pi)

# ============================================================
#  STANDARD RUN WRAPPERS
# ============================================================
# Default parameters chosen for good accuracy / reasonable runtime
_N_DEFAULT    = 20
_NROT_DEFAULT = 2.0
_SEGS_DEFAULT = 36


def standard_BEM_run(TSR, N=_N_DEFAULT):
    Omega   = TSR * U0 / RADIUS
    r_R_arr = cosine_r_array(N)
    res     = solve_BEM(U0, r_R_arr, Omega, RADIUS, NBLADES)
    CT, CP  = CT_CP_BEM(res, r_R_arr, U0, Omega, RADIUS, NBLADES)
    return res, r_R_arr, CT, CP


def standard_LL_run(TSR, N=_N_DEFAULT, Nrot=_NROT_DEFAULT, segs=_SEGS_DEFAULT):
    Omega    = TSR * U0 / RADIUS
    r_R_arr  = cosine_r_array(N)
    span_arr = [r*RADIUS for r in r_R_arr]
    maxR     = max(span_arr)
    theta    = build_wake_theta(segs, Nrot)
    tsr_wake = TSR / (1.0 - ROOT_R)   # convection: mean axial velocity = (1-a)*U0

    wake = create_rotor_geometry(span_arr, maxR, tsr_wake, theta, NBLADES)
    res  = solve_lifting_line(wake, [U0, 0, 0], Omega, maxR)
    CT, CP = CT_CP_LL(res, r_R_arr, U0, Omega, maxR, NBLADES)
    return res, r_R_arr, CT, CP

# ============================================================
#  PLOT SETUP
# ============================================================
COLORS   = ['#1f77b4', '#ff7f0e', '#2ca02c']   # blue / orange / green
_LS_BEM  = '-'
_LS_LL   = '--'


def _savefig(fig, name):
    fig.savefig(f"{name}.pdf", bbox_inches='tight')
    fig.savefig(f"{name}.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved  {name}.pdf / .png")

# ============================================================
#  PLOT 1 — inflow angle phi and angle of attack alpha
# ============================================================

def plot_inflow_alpha():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for k, TSR in enumerate(TSR_LIST):
        r_bem, res_bem = standard_BEM_run(TSR)[1], standard_BEM_run(TSR)[0]
        r_ll,  res_ll  = standard_LL_run(TSR)[1],  standard_LL_run(TSR)[0]
        n = len(r_bem)-1
        r_mid_bem = [(r_bem[i]+r_bem[i+1])/2 for i in range(n)]
        r_mid_ll  = res_ll['r_R'][:n]
        for ax, key in zip(axes, ['phi', 'alpha']):
            ax.plot(r_mid_bem, res_bem[key],    color=COLORS[k],
                    ls=_LS_BEM, label=f'BEM  λ={TSR:.0f}')
            ax.plot(r_mid_ll,  res_ll[key][:n], color=COLORS[k],
                    ls=_LS_LL,  label=f'LL   λ={TSR:.0f}')
    axes[0].set_title('Inflow angle  φ')
    axes[0].set_ylabel('φ  [deg]')
    axes[1].set_title('Angle of attack  α')
    axes[1].set_ylabel('α  [deg]')
    for ax in axes:
        ax.set_xlabel('r/R  [-]'); ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle('Radial distribution — φ and α  '
                 '(solid = BEM, dashed = LL)', fontsize=11)
    fig.tight_layout()
    _savefig(fig, 'plot1_phi_alpha')

# ============================================================
#  PLOT 2 — axial and tangential induction
# ============================================================

def plot_induction():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for k, TSR in enumerate(TSR_LIST):
        r_bem, res_bem = standard_BEM_run(TSR)[1], standard_BEM_run(TSR)[0]
        r_ll,  res_ll  = standard_LL_run(TSR)[1],  standard_LL_run(TSR)[0]
        n = len(r_bem)-1
        r_mid_bem = [(r_bem[i]+r_bem[i+1])/2 for i in range(n)]
        r_mid_ll  = res_ll['r_R'][:n]
        for ax, key in zip(axes, ['a', 'aline']):
            ax.plot(r_mid_bem, res_bem[key],    color=COLORS[k],
                    ls=_LS_BEM, label=f'BEM  λ={TSR:.0f}')
            ax.plot(r_mid_ll,  res_ll[key][:n], color=COLORS[k],
                    ls=_LS_LL,  label=f'LL   λ={TSR:.0f}')
    axes[0].set_title("Axial induction  a");        axes[0].set_ylabel("a  [-]")
    axes[1].set_title("Tangential induction  a'");  axes[1].set_ylabel("a'  [-]")
    for ax in axes:
        ax.set_xlabel('r/R  [-]'); ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle('Radial distribution — induction factors', fontsize=11)
    fig.tight_layout()
    _savefig(fig, 'plot2_induction')

# ============================================================
#  PLOT 3 — axial and tangential loading (non-dimensionalised)
# ============================================================

def plot_loading():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    nd = 0.5 * U0**2 * RADIUS        # reference: 1/2 * U0^2 * R
    for k, TSR in enumerate(TSR_LIST):
        r_bem, res_bem = standard_BEM_run(TSR)[1], standard_BEM_run(TSR)[0]
        r_ll,  res_ll  = standard_LL_run(TSR)[1],  standard_LL_run(TSR)[0]
        n = len(r_bem)-1
        r_mid_bem = [(r_bem[i]+r_bem[i+1])/2 for i in range(n)]
        r_mid_ll  = res_ll['r_R'][:n]
        for ax, key in zip(axes, ['Fnorm', 'Ftan']):
            ax.plot(r_mid_bem, [f/nd for f in res_bem[key]],
                    color=COLORS[k], ls=_LS_BEM, label=f'BEM  λ={TSR:.0f}')
            ax.plot(r_mid_ll,  [f/nd for f in res_ll[key][:n]],
                    color=COLORS[k], ls=_LS_LL,  label=f'LL   λ={TSR:.0f}')
    axes[0].set_title('Axial loading  Fnorm')
    axes[0].set_ylabel('Fnorm / (½ U₀² R)  [-]')
    axes[1].set_title('Tangential loading  Ftan')
    axes[1].set_ylabel('Ftan / (½ U₀² R)  [-]')
    for ax in axes:
        ax.set_xlabel('r/R  [-]'); ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle('Radial distribution — non-dimensional blade loading', fontsize=11)
    fig.tight_layout()
    _savefig(fig, 'plot3_loading')

# ============================================================
#  PLOT 4 — circulation (non-dimensionalised)
# ============================================================

def plot_circulation():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k, TSR in enumerate(TSR_LIST):
        Omega = TSR * U0 / RADIUS
        adim  = math.pi / (NBLADES * TSR / RADIUS)   # non-dim scale
        r_bem, res_bem = standard_BEM_run(TSR)[1], standard_BEM_run(TSR)[0]
        r_ll,  res_ll  = standard_LL_run(TSR)[1],  standard_LL_run(TSR)[0]
        n = len(r_bem)-1
        r_mid_bem = [(r_bem[i]+r_bem[i+1])/2 for i in range(n)]
        r_mid_ll  = res_ll['r_R'][:n]
        ax.plot(r_mid_bem, [g/adim for g in res_bem['Gamma']],
                color=COLORS[k], ls=_LS_BEM, label=f'BEM  λ={TSR:.0f}')
        ax.plot(r_mid_ll,  [g/adim for g in res_ll['Gamma'][:n]],
                color=COLORS[k], ls=_LS_LL,  label=f'LL   λ={TSR:.0f}')
    ax.set_xlabel('r/R  [-]')
    ax.set_ylabel('Γ / (π U₀ R / (B·λ))  [-]')
    ax.set_title('Radial distribution — non-dimensional circulation')
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _savefig(fig, 'plot4_circulation')

# ============================================================
#  PLOT 5 — CT and CP comparison
# ============================================================

def plot_CT_CP():
    rows = []
    for TSR in TSR_LIST:
        _, _, CTb, CPb = standard_BEM_run(TSR)
        _, _, CTl, CPl = standard_LL_run(TSR)
        rows.append(dict(TSR=TSR, CT_BEM=CTb, CP_BEM=CPb,
                                   CT_LL=CTl,  CP_LL=CPl))
    df = pd.DataFrame(rows)
    print("\n=== CT / CP Comparison ===")
    print(df.to_string(index=False, float_format='{:.4f}'.format))

    x = np.arange(len(TSR_LIST))
    w = 0.2
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, col, title in zip(axes,
                               ['CT', 'CP'],
                               ['Thrust coeff. CT', 'Power coeff. CP']):
        ax.bar(x-w, df[f'{col}_BEM'], 2*w, label='BEM', color='steelblue')
        ax.bar(x+w, df[f'{col}_LL'],  2*w, label='LL',  color='darkorange')
        ax.set_xticks(x)
        ax.set_xticklabels([f'λ={t:.0f}' for t in TSR_LIST])
        ax.set_title(title); ax.set_ylabel(col)
        ax.legend(); ax.grid(axis='y', alpha=0.3)
    fig.suptitle('CT and CP — BEM vs Lifting Line', fontsize=11)
    fig.tight_layout()
    _savefig(fig, 'plot5_CT_CP')
    return df

# ============================================================
#  PLOT 6 — sensitivity: convection speed
# ============================================================

def plot_sens_convection():
    TSR   = 8.0
    Omega = TSR * U0 / RADIUS
    r_R_arr  = cosine_r_array(_N_DEFAULT)
    span_arr = [r*RADIUS for r in r_R_arr]
    maxR     = max(span_arr)
    theta    = build_wake_theta(_SEGS_DEFAULT, _NROT_DEFAULT)
    n        = len(r_R_arr) - 1

    # vary the assumed mean axial induction for the wake pitch
    a_conv_vals = [0.0, 0.1, 1.0/3.0, 0.4]
    labels      = [f'a_conv={a:.2f}  (Uc={(1-a)*U0:.1f} m/s)' for a in a_conv_vals]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for a_c, lbl in zip(a_conv_vals, labels):
        tsr_w = TSR / max(1.0 - a_c, 0.05)
        wake  = create_rotor_geometry(span_arr, maxR, tsr_w, theta, NBLADES)
        res   = solve_lifting_line(wake, [U0,0,0], Omega, maxR)
        rr    = res['r_R'][:n]
        axes[0].plot(rr, res['a'][:n],     label=lbl)
        axes[1].plot(rr, res['Gamma'][:n], label=lbl)

    for ax, ylabel, title in zip(
            axes, ['a  [-]', 'Γ  [m²/s]'],
            ['Axial induction', 'Circulation']):
        ax.set_xlabel('r/R'); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle(f'Sensitivity — convection speed  (λ={TSR:.0f})', fontsize=11)
    fig.tight_layout()
    _savefig(fig, 'plot6_sens_convection')

# ============================================================
#  PLOT 7 — sensitivity: blade discretisation
# ============================================================

def plot_sens_discretisation():
    TSR   = 8.0
    Omega = TSR * U0 / RADIUS
    tsr_w = TSR / (1.0 - ROOT_R)
    theta = build_wake_theta(_SEGS_DEFAULT, _NROT_DEFAULT)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for N in [10, 20, 40]:
        for disc, ls in [('cosine', '-'), ('uniform', '--')]:
            r_R_arr = cosine_r_array(N) if disc=='cosine' else uniform_r_array(N)
            span_arr = [r*RADIUS for r in r_R_arr]
            maxR     = max(span_arr)
            wake = create_rotor_geometry(span_arr, maxR, tsr_w, theta, NBLADES)
            res  = solve_lifting_line(wake, [U0,0,0], Omega, maxR)
            n    = len(r_R_arr)-1
            rr   = res['r_R'][:n]
            lbl  = f'N={N} {disc}'
            axes[0].plot(rr, res['a'][:n],     ls=ls, label=lbl)
            axes[1].plot(rr, res['Gamma'][:n], ls=ls, label=lbl)

    for ax, ylabel, title in zip(
            axes, ['a  [-]', 'Γ  [m²/s]'],
            ['Axial induction', 'Circulation']):
        ax.set_xlabel('r/R'); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=6, ncol=2)
    fig.suptitle(f'Sensitivity — blade discretisation  (λ={TSR:.0f})', fontsize=11)
    fig.tight_layout()
    _savefig(fig, 'plot7_sens_discretisation')

# ============================================================
#  PLOT 8 — sensitivity: azimuthal discretisation
# ============================================================

def plot_sens_azimuthal():
    TSR   = 8.0
    Omega = TSR * U0 / RADIUS
    tsr_w = TSR / (1.0 - ROOT_R)
    r_R_arr  = cosine_r_array(_N_DEFAULT)
    span_arr = [r*RADIUS for r in r_R_arr]
    maxR     = max(span_arr)
    n        = len(r_R_arr)-1

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for segs in [12, 24, 36, 72]:
        theta = build_wake_theta(segs, _NROT_DEFAULT)
        wake  = create_rotor_geometry(span_arr, maxR, tsr_w, theta, NBLADES)
        res   = solve_lifting_line(wake, [U0,0,0], Omega, maxR)
        rr    = res['r_R'][:n]
        axes[0].plot(rr, res['a'][:n],     label=f'{segs} seg/rot')
        axes[1].plot(rr, res['Gamma'][:n], label=f'{segs} seg/rot')

    for ax, ylabel, title in zip(
            axes, ['a  [-]', 'Γ  [m²/s]'],
            ['Axial induction', 'Circulation']):
        ax.set_xlabel('r/R'); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(f'Sensitivity — azimuthal discretisation  (λ={TSR:.0f})', fontsize=11)
    fig.tight_layout()
    _savefig(fig, 'plot8_sens_azimuthal')

# ============================================================
#  PLOT 9 — sensitivity: wake length + convergence
# ============================================================

def plot_sens_wake_length():
    TSR   = 8.0
    Omega = TSR * U0 / RADIUS
    tsr_w = TSR / (1.0 - ROOT_R)
    r_R_arr  = cosine_r_array(_N_DEFAULT)
    span_arr = [r*RADIUS for r in r_R_arr]
    maxR     = max(span_arr)
    n        = len(r_R_arr)-1

    nrot_list = [0.5, 1.0, 2.0, 3.0, 5.0]
    CT_list   = []; CP_list = []

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for nrot in nrot_list:
        theta = build_wake_theta(_SEGS_DEFAULT, nrot)
        wake  = create_rotor_geometry(span_arr, maxR, tsr_w, theta, NBLADES)
        res   = solve_lifting_line(wake, [U0,0,0], Omega, maxR)
        rr    = res['r_R'][:n]
        axes[0].plot(rr, res['a'][:n], label=f'{nrot:.1f} rot')
        CT, CP = CT_CP_LL(res, r_R_arr, U0, Omega, maxR, NBLADES)
        CT_list.append(CT); CP_list.append(CP)

    axes[0].set_xlabel('r/R'); axes[0].set_ylabel('a  [-]')
    axes[0].set_title('Axial induction'); axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(nrot_list, CT_list, 'o-', label='CT', color='steelblue')
    axes[1].plot(nrot_list, CP_list, 's-', label='CP', color='darkorange')
    axes[1].set_xlabel('Wake length  [rotations]')
    axes[1].set_ylabel('Coefficient  [-]')
    axes[1].set_title('CT and CP convergence with wake length')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    fig.suptitle(f'Sensitivity — wake length  (λ={TSR:.0f})', fontsize=11)
    fig.tight_layout()
    _savefig(fig, 'plot9_sens_wake_length')

# ============================================================
#  MAIN
# ============================================================

if __name__ == '__main__':
    print("\n" + "="*55)
    print(" AE4135 Lifting Line Assignment — DU 95-W-180 rotor")
    print("="*55)
    print("Running all plots (may take several minutes) ...\n")

    print("[1/9] Inflow angle and AoA ...")
    plot_inflow_alpha()

    print("[2/9] Induction factors ...")
    plot_induction()

    print("[3/9] Blade loading ...")
    plot_loading()

    print("[4/9] Circulation ...")
    plot_circulation()

    print("[5/9] CT / CP comparison ...")
    df = plot_CT_CP()

    print("[6/9] Sensitivity — convection speed ...")
    plot_sens_convection()

    print("[7/9] Sensitivity — discretisation ...")
    plot_sens_discretisation()

    print("[8/9] Sensitivity — azimuthal resolution ...")
    plot_sens_azimuthal()

    print("[9/9] Sensitivity — wake length ...")
    plot_sens_wake_length()

    print("\nAll done.  PDF and PNG files saved in the working directory.")