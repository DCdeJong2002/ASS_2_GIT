"""
LLM_Spike_Analysis.py
Generates specific plots of the inflow angle (phi) at TSR=6 for different
panel resolutions (N) and spanwise distributions (cosine vs constant) to analyze the root spike.
Maximum of 2 lines per plot for clarity.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# 1. Geometry and Core Parameters
Radius = 50.0
NBlades = 3
U0 = 10.0
rho = 1.0
RootLocation_R = 0.2
TipLocation_R = 1.0
Pitch = -2.0
DPSI_DEG = 10.0
A_WAKE = 0.25
R_CORE = 1e-3
TSR = 6.0  # We evaluate at TSR=6 where the spike is most prominent

# Set up output directory
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Spike_Analysis_Plots")
os.makedirs(OUT_DIR, exist_ok=True)

# Load Polar
_df = pd.read_excel("polar DU95W180 (3).xlsx", skiprows=3)
polar_alpha = _df["Alfa"].to_numpy()
polar_cl = _df["Cl"].to_numpy()

# -----------------------------------------------------------------------------
# CORE FUNCTIONS (Robust Physics Engine)
# -----------------------------------------------------------------------------
def blade_chord(r_R): return 3.0 * (1.0 - r_R) + 1.0
def blade_twist(r_R): return 14.0 * (1.0 - r_R) + Pitch

def build_geometry(N, N_wake, distribution="cosine"):
    if distribution == "cosine":
        theta = np.linspace(0.0, np.pi, N + 1)
        r_edges = (RootLocation_R * Radius) + 0.5 * ((TipLocation_R - RootLocation_R) * Radius) * (1.0 - np.cos(theta))
    else: # constant/uniform
        r_edges = np.linspace(RootLocation_R * Radius, TipLocation_R * Radius, N + 1)
        
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    
    Omega = U0 * TSR / Radius
    U_wake = U0 * (1.0 - A_WAKE)
    dpsi = np.radians(DPSI_DEG)
    psi_arr = np.arange(0.0, N_wake * 2.0 * np.pi + dpsi * 0.5, dpsi)

    controlpoints = []
    rings = []

    for k_blade in range(NBlades):
        angle_rot = 2.0 * np.pi / NBlades * k_blade
        cosR, sinR = np.cos(angle_rot), np.sin(angle_rot)

        for i in range(N):
            r = r_centers[i]
            chord = blade_chord(r / Radius)
            twist_rad = np.radians(blade_twist(r / Radius))
            cp_rot = np.array([0.0, r * cosR, r * sinR])

            controlpoints.append({'r': r, 'r_R': r / Radius, 'chord': chord, 'twist_rad': twist_rad, 'coords': cp_rot})

            filaments = []
            # Bound
            filaments.append({'x1': 0.0, 'y1': r_edges[i]*cosR, 'z1': r_edges[i]*sinR, 
                              'x2': 0.0, 'y2': r_edges[i+1]*cosR, 'z2': r_edges[i+1]*sinR})
            
            # Inner
            for j in range(len(psi_arr) - 1):
                t1, t2 = psi_arr[j] / Omega, psi_arr[j+1] / Omega
                y1, z1 = r_edges[i] * np.cos(-Omega * t1), r_edges[i] * np.sin(-Omega * t1)
                y2, z2 = r_edges[i] * np.cos(-Omega * t2), r_edges[i] * np.sin(-Omega * t2)
                filaments.append({'x1': U_wake * t2, 'y1': y2 * cosR - z2 * sinR, 'z1': y2 * sinR + z2 * cosR,
                                  'x2': U_wake * t1, 'y2': y1 * cosR - z1 * sinR, 'z2': y1 * sinR + z1 * cosR})
            
            # Outer
            for j in range(len(psi_arr) - 1):
                t1, t2 = psi_arr[j] / Omega, psi_arr[j+1] / Omega
                y1, z1 = r_edges[i+1] * np.cos(-Omega * t1), r_edges[i+1] * np.sin(-Omega * t1)
                y2, z2 = r_edges[i+1] * np.cos(-Omega * t2), r_edges[i+1] * np.sin(-Omega * t2)
                filaments.append({'x1': U_wake * t1, 'y1': y1 * cosR - z1 * sinR, 'z1': y1 * sinR + z1 * cosR,
                                  'x2': U_wake * t2, 'y2': y2 * cosR - z2 * sinR, 'z2': y2 * sinR + z2 * cosR})
            rings.append(filaments)
    return controlpoints, rings

def biot_savart_segment(x1, x2, xp, r_core=1e-3):
    """Includes robust Van Garrel algebraic core regularization"""
    R1 = xp - x1
    R2 = xp - x2
    cross = np.cross(R1, R2)
    cross_sq = np.einsum('ij,ij->i', cross, cross)
    
    R1_mag = np.linalg.norm(R1, axis=1)
    R2_mag = np.linalg.norm(R2, axis=1)
    
    d = x2 - x1
    d_norm = np.linalg.norm(d)
    
    # Geometric delta regularization to prevent 1/0 singularities
    delta_sq = (1e-4 * d_norm)**2
    cross_sq_reg = cross_sq + delta_sq
    
    R1_mag_reg = np.sqrt(R1_mag**2 + delta_sq)
    R2_mag_reg = np.sqrt(R2_mag**2 + delta_sq)
    
    R0R1 = np.einsum('j,ij->i', d, R1)
    R0R2 = np.einsum('j,ij->i', d, R2)
    
    K = (1.0 / (4.0 * np.pi * cross_sq_reg)) * (R0R1 / R1_mag_reg - R0R2 / R2_mag_reg)
    return K[:, None] * cross

def get_inflow_angle(N, N_wake, distribution="cosine"):
    Omega = U0 * TSR / Radius
    cps, rings = build_geometry(N, N_wake, distribution)
    
    A_u = np.zeros((len(cps), len(cps)))
    A_v = np.zeros((len(cps), len(cps)))
    A_w = np.zeros((len(cps), len(cps)))
    XP = np.array([cp['coords'] for cp in cps])

    for j, ring in enumerate(rings):
        vel_j = np.zeros((len(cps), 3))
        for fil in ring:
            x1, x2 = np.array([fil['x1'], fil['y1'], fil['z1']]), np.array([fil['x2'], fil['y2'], fil['z2']])
            vel_j += biot_savart_segment(x1, x2, XP)
        A_u[:, j], A_v[:, j], A_w[:, j] = vel_j[:, 0], vel_j[:, 1], vel_j[:, 2]

    Gamma = np.zeros(len(cps))
    R_prev = np.zeros(len(cps))
    omega = 0.1
    
    u_ind = v_ind = w_ind = np.zeros(len(cps))
    
    for k in range(500):
        u_ind = A_u @ Gamma
        v_ind = A_v @ Gamma
        w_ind = A_w @ Gamma
        Gamma_new = np.zeros(len(cps))
        
        for i, cp in enumerate(cps):
            v_rot = np.cross([-Omega, 0, 0], cp['coords'])
            V_ax = U0 + u_ind[i] + v_rot[0]
            V_y = v_ind[i] + v_rot[1]
            V_z = w_ind[i] + v_rot[2]
            
            azim_dir = np.cross([-1.0 / max(cp['r'], 1e-12), 0.0, 0.0], cp['coords'])
            V_tan = float(np.dot(azim_dir, [V_ax, V_y, V_z]))
            
            V_eff = np.sqrt(V_ax**2 + V_tan**2)
            phi = np.arctan2(V_ax, V_tan)
            alpha = phi - cp['twist_rad']
            
            cl = float(np.interp(np.degrees(alpha), polar_alpha, polar_cl))
            Gamma_new[i] = 0.5 * cp['chord'] * V_eff * cl
            
        R_k = Gamma_new - Gamma
        err = np.max(np.abs(R_k)) / max(np.max(np.abs(Gamma_new)), 1e-6)
        if err < 1e-4: break
        
        if k > 0:
            delta_R = R_k - R_prev
            den = np.dot(delta_R, delta_R)
            if den > 1e-16:
                omega = -omega * np.dot(R_prev, delta_R) / den
            omega = float(np.clip(omega, 0.01, 0.5))
            
        Gamma = Gamma + omega * R_k
        R_prev = R_k.copy()
    
    # Calculate final phi array specifically for the first blade
    phi_arr = np.zeros(N)
    r_R_arr = np.zeros(N)
    for i in range(N):
        cp = cps[i]
        v_rot = np.cross([-Omega, 0, 0], cp['coords'])
        V_ax = U0 + u_ind[i] + v_rot[0]
        V_y = v_ind[i] + v_rot[1]
        V_z = w_ind[i] + v_rot[2]
        
        azim_dir = np.cross([-1.0 / max(cp['r'], 1e-12), 0.0, 0.0], cp['coords'])
        V_tan = float(np.dot(azim_dir, [V_ax, V_y, V_z]))
        
        phi_arr[i] = np.degrees(np.arctan2(V_ax, V_tan))
        r_R_arr[i] = cp['r_R']
        
    return r_R_arr, phi_arr

# -----------------------------------------------------------------------------
# EXECUTION & PLOTTING
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    N_list = [10, 12, 15, 18, 20, 25, 30, 35]
    
    # Split the N_list into chunks of 2 for plotting
    N_pairs = [N_list[i:i + 2] for i in range(0, len(N_list), 2)]

    # ---------------------------------------------------------
    # Part 1: N_wake = 5, Cosine Distribution
    # ---------------------------------------------------------
    print("Running Part 1: N_wake = 5, Cosine Distribution ...")
    results_cos = {}
    
    for n in N_list:
        print(f"  Evaluating N = {n} ...")
        results_cos[n] = get_inflow_angle(N=n, N_wake=5, distribution="cosine")
        
    for pair in N_pairs:
        plt.figure(figsize=(9, 6))
        for n in pair:
            r_R, phi = results_cos[n]
            plt.plot(r_R, phi, lw=2, label=f"N = {n}")
            
        plt.title(rf"Radial Inflow Angle ($\lambda=6, N_{{wake}}=5$, Cosine Spacing)")
        plt.xlabel("r/R")
        plt.ylabel(r"$\phi$ [deg]")
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        suffix = "_".join(str(x) for x in pair)
        plt.savefig(os.path.join(OUT_DIR, f"Inflow_Angle_Nwake5_Cosine_N_{suffix}.png"))
        plt.close()

    # ---------------------------------------------------------
    # Part 2: N_wake = 5, Constant Distribution
    # ---------------------------------------------------------
    print("\nRunning Part 2: N_wake = 5, Constant Distribution ...")
    results_con = {}
    
    for n in N_list:
        print(f"  Evaluating N = {n} ...")
        results_con[n] = get_inflow_angle(N=n, N_wake=5, distribution="constant")
        
    for pair in N_pairs:
        plt.figure(figsize=(9, 6))
        for n in pair:
            r_R, phi = results_con[n]
            plt.plot(r_R, phi, lw=2, label=f"N = {n}")
            
        plt.title(rf"Radial Inflow Angle ($\lambda=6, N_{{wake}}=5$, Constant Spacing)")
        plt.xlabel("r/R")
        plt.ylabel(r"$\phi$ [deg]")
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        suffix = "_".join(str(x) for x in pair)
        plt.savefig(os.path.join(OUT_DIR, f"Inflow_Angle_Nwake5_Constant_N_{suffix}.png"))
        plt.close()

    print(f"\nDone! Plots saved to {OUT_DIR}")