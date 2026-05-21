import os
import math
import numpy as np
import openpyxl
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# MODULE 1: GEOMETRY SETUP
# =============================================================================

def geoBlade(r_R):
    """Assignment definition of the DU95W180 baseline wind turbine blade."""
    pitch = -2.0
    chord = 3.0 * (1.0 - r_R) + 1.0
    twist = 14.0 * (1.0 - r_R)
    return chord, twist + pitch

def build_geometry(R=50.0, r_root=10.0, N=30, N_blades=3, N_wake=5, 
                   dpsi_deg=10.0, U_inf=10.0, Omega=1.6, a_w=0.25, distribution="cosine"):
    
    # 1. Spanwise Discretization (Ensuring exact symmetric midpoints)
    if distribution == "cosine":
        theta_dist = np.linspace(0, math.pi, N + 1)
        r_edges = r_root + 0.5 * (R - r_root) * (1 - np.cos(theta_dist))
    else:
        r_edges = np.linspace(r_root, R, N + 1)
        
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    
    # 2. Wake Azimuthal Discretization
    dpsi = math.radians(dpsi_deg)
    theta_arr = np.arange(0, N_wake * 2 * math.pi + dpsi * 0.5, dpsi)
    U_wake = U_inf * (1.0 - a_w)
    
    controlpoints = []
    rings = []
    
    for k in range(N_blades):
        psi_k = k * 2 * math.pi / N_blades
        
        for i in range(N):
            r = r_centers[i]
            chord, twist_deg = geoBlade(r / R)
            beta = math.radians(twist_deg)
            
            # --- Pistolesi Control Point Offset (c/4 to 3c/4) ---
            # Bound vortex at c/4. Chord vector points LE to TE.
            # Local tangential direction (motion): e_theta = (0, cos(psi), -sin(psi))
            # Offset c/2 downstream towards trailing edge.
            dx = (chord / 2.0) * math.sin(beta)
            dy = (chord / 2.0) * math.cos(beta) * math.cos(psi_k)
            dz = -(chord / 2.0) * math.cos(beta) * math.sin(psi_k)
            
            x_c4 = np.array([0.0, -r * math.sin(psi_k), r * math.cos(psi_k)])
            x_cp = x_c4 + np.array([dx, dy, dz])
            
            if k == 0:
                controlpoints.append({
                    'coords': x_cp, 'r': r, 'r_R': r / R, 'chord': chord, 
                    'twist': beta, 'dr': r_edges[i+1] - r_edges[i]
                })
            
            # --- Horseshoe Vortex Generation ---
            fils = []
            
            # Bound Vortex (c/4)
            x1 = np.array([0.0, -r_edges[i] * math.sin(psi_k), r_edges[i] * math.cos(psi_k)])
            x2 = np.array([0.0, -r_edges[i+1] * math.sin(psi_k), r_edges[i+1] * math.cos(psi_k)])
            fils.append((x1, x2))
            
            # Trailing Inner Wake (from Wake to Blade)
            for j in range(len(theta_arr) - 1):
                t1, t2 = theta_arr[j] / Omega, theta_arr[j+1] / Omega
                x1_w = np.array([U_wake * t1, -r_edges[i] * math.sin(psi_k - theta_arr[j]), r_edges[i] * math.cos(psi_k - theta_arr[j])])
                x2_w = np.array([U_wake * t2, -r_edges[i] * math.sin(psi_k - theta_arr[j+1]), r_edges[i] * math.cos(psi_k - theta_arr[j+1])])
                fils.append((x2_w, x1_w)) # Reversed so vector goes towards blade
                
            # Trailing Outer Wake (from Blade to Wake)
            for j in range(len(theta_arr) - 1):
                t1, t2 = theta_arr[j] / Omega, theta_arr[j+1] / Omega
                x1_w = np.array([U_wake * t1, -r_edges[i+1] * math.sin(psi_k - theta_arr[j]), r_edges[i+1] * math.cos(psi_k - theta_arr[j])])
                x2_w = np.array([U_wake * t2, -r_edges[i+1] * math.sin(psi_k - theta_arr[j+1]), r_edges[i+1] * math.cos(psi_k - theta_arr[j+1])])
                fils.append((x1_w, x2_w))
                
            rings.append(fils)
            
    return controlpoints, rings

# =============================================================================
# MODULE 2: BIOT-SAVART KERNEL (Vectorized)
# =============================================================================

def filament_velocity(X1, X2, XP, r_core=0.05):
    R1 = XP - X1
    R2 = XP - X2
    cross = np.cross(R1, R2)
    cross_sq = np.sum(cross**2, axis=1)
    
    R1_mag = np.linalg.norm(R1, axis=1)
    R2_mag = np.linalg.norm(R2, axis=1)
    
    inside_core = (cross_sq < r_core**2) | (R1_mag < r_core) | (R2_mag < r_core)
    
    R0R1 = np.sum((X2 - X1) * R1, axis=1)
    R0R2 = np.sum((X2 - X1) * R2, axis=1)
    
    R1_mag = np.where(R1_mag == 0, 1e-12, R1_mag)
    R2_mag = np.where(R2_mag == 0, 1e-12, R2_mag)
    cross_sq = np.where(cross_sq == 0, 1e-12, cross_sq)
    
    K = 1.0 / (4.0 * math.pi * cross_sq) * (R0R1 / R1_mag - R0R2 / R2_mag)
    K[inside_core] = 0.0
    
    return K[:, None] * cross

# =============================================================================
# MODULE 3: INFLUENCE MATRIX ASSEMBLY
# =============================================================================

def assemble_influence(controlpoints, rings, N_blades):
    N = len(controlpoints)
    XP = np.array([cp['coords'] for cp in controlpoints])
    
    U_mat, V_mat, W_mat = np.zeros((N, N)), np.zeros((N, N)), np.zeros((N, N))
    
    for j in range(N):
        vel = np.zeros((N, 3))
        for k in range(N_blades):
            ring_idx = k * N + j
            for fil in rings[ring_idx]:
                vel += filament_velocity(fil[0], fil[1], XP)
        U_mat[:, j] = vel[:, 0]
        V_mat[:, j] = vel[:, 1]
        W_mat[:, j] = vel[:, 2]
        
    return U_mat, V_mat, W_mat

# =============================================================================
# MODULE 4: AIRFOIL POLAR INTERPOLATION
# =============================================================================

def load_polar_excel(filepath):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Could not find {filepath}. Place it in the same folder.")
    wb = openpyxl.load_workbook(filepath, data_only=True)
    ws = wb.active
    data = []
    for row in ws.iter_rows(min_row=1, values_only=True):
        if row[0] is not None:
            try:
                data.append([float(row[0]), float(row[1]), float(row[2])])
            except (ValueError, TypeError):
                continue
    data = np.array(data)
    return {'alpha': data[:, 0], 'Cl': data[:, 1], 'Cd': data[:, 2]}

def airfoil_coefficients(alpha_deg, polar):
    cl = np.interp(alpha_deg, polar['alpha'], polar['Cl'])
    cd = np.interp(alpha_deg, polar['alpha'], polar['Cd'])
    return cl, cd

# =============================================================================
# MODULE 5: ITERATION LOOP (Heavy Stabilized Relaxation)
# =============================================================================

def solve_circulation(U_mat, V_mat, W_mat, cps, polar, U_inf, Omega, tol=1e-3):
    N = len(cps)
    Gamma = np.zeros(N)
    
    # Using a heavy fixed under-relaxation to guarantee complete stability
    # on highly dense cosine grids, completely preventing odd-even decoupling.
    w_relax = 0.03  
    max_iter = 800
    
    for k in range(max_iter):
        u_ind = U_mat @ Gamma
        v_ind = V_mat @ Gamma
        w_ind = W_mat @ Gamma
        Gamma_new = np.zeros(N)
        
        for i, cp in enumerate(cps):
            V_ax = U_inf + u_ind[i]
            
            # Local tangential direction (motion)
            azimdir = np.array([0.0, math.cos(0), -math.sin(0)]) # Reference blade at psi=0
            v_azim_fluid = np.dot(azimdir, [u_ind[i], v_ind[i], w_ind[i]])
            V_tan = Omega * cp['r'] - v_azim_fluid
            
            V_p = math.sqrt(V_ax**2 + V_tan**2)
            phi = math.atan2(V_ax, V_tan)
            alpha = phi - cp['twist']
            
            cl, _ = airfoil_coefficients(math.degrees(alpha), polar)
            Gamma_new[i] = 0.5 * cp['chord'] * V_p * cl
            
        err = np.max(np.abs(Gamma_new - Gamma)) / max(np.max(np.abs(Gamma_new)), 1e-3)
        if err < tol:
            print(f"    > Converged perfectly in {k} iterations.")
            break
            
        Gamma = (1 - w_relax) * Gamma + w_relax * Gamma_new
        
    return Gamma, u_ind, v_ind, w_ind

# =============================================================================
# MODULE 6: POST-PROCESSING & SENSITIVITY WRAPPER
# =============================================================================

def post_process(Gamma, u_ind, v_ind, w_ind, cps, polar, U_inf, Omega, N_blades, R):
    N = len(cps)
    r_R, phi_deg, alpha_deg, a_ind, aline_ind = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
    F_axial, F_azim, Gamma_out = np.zeros(N), np.zeros(N), np.zeros(N)
    
    T, P = 0.0, 0.0
    
    for i, cp in enumerate(cps):
        V_ax = U_inf + u_ind[i]
        azimdir = np.array([0.0, math.cos(0), -math.sin(0)]) 
        v_azim_fluid = np.dot(azimdir, [u_ind[i], v_ind[i], w_ind[i]])
        V_tan = Omega * cp['r'] - v_azim_fluid
        V_p = math.sqrt(V_ax**2 + V_tan**2)
        
        phi = math.atan2(V_ax, V_tan)
        alpha = phi - cp['twist']
        cl, cd = airfoil_coefficients(math.degrees(alpha), polar)
        
        Lift = 0.5 * V_p**2 * cp['chord'] * cl
        Drag = 0.5 * V_p**2 * cp['chord'] * cd
        
        F_ax = Lift * math.cos(phi) + Drag * math.sin(phi)
        F_az = Lift * math.sin(phi) - Drag * math.cos(phi)
        
        r_R[i] = cp['r_R']
        phi_deg[i] = math.degrees(phi)
        alpha_deg[i] = math.degrees(alpha)
        a_ind[i] = -u_ind[i] / U_inf
        aline_ind[i] = v_azim_fluid / (cp['r'] * Omega)
        F_axial[i] = F_ax
        F_azim[i] = F_az
        Gamma_out[i] = Gamma[i]
        
        T += N_blades * F_ax * cp['dr']
        P += N_blades * F_az * cp['r'] * Omega * cp['dr']
        
    CT = T / (0.5 * U_inf**2 * math.pi * R**2)
    CP = P / (0.5 * U_inf**3 * math.pi * R**2)
    
    return {
        'r_R': r_R, 'phi': phi_deg, 'alpha': alpha_deg, 'a': a_ind, 'aline': aline_ind,
        'F_ax': F_axial, 'F_az': F_azim, 'Gamma': Gamma_out, 'CT': CT, 'CP': CP
    }

def run_case(U_inf, TSR, R, N_blades, polar, kwargs):
    Omega = TSR * U_inf / R
    cps, rings = build_geometry(R=R, U_inf=U_inf, Omega=Omega, N_blades=N_blades, **kwargs)
    U_mat, V_mat, W_mat = assemble_influence(cps, rings, N_blades)
    Gamma, u_ind, v_ind, w_ind = solve_circulation(U_mat, V_mat, W_mat, cps, polar, U_inf, Omega)
    res = post_process(Gamma, u_ind, v_ind, w_ind, cps, polar, U_inf, Omega, N_blades, R)
    print(f"      CT = {res['CT']:.4f}  |  CP = {res['CP']:.4f}")
    return res

# =============================================================================
# MAIN EXECUTION AND SEPARATE PDF PLOTTING
# =============================================================================
if __name__ == "__main__":
    R, U_inf, N_blades = 50.0, 10.0, 3
    script_dir = os.path.dirname(os.path.abspath(__file__))
    polar_data = load_polar_excel(os.path.join(script_dir, "polar DU95W180 (3).xlsx"))
    
    out_dir = os.path.join(script_dir, "LLM_Plots")
    os.makedirs(out_dir, exist_ok=True)
    
    # Requested Color Sequence
    colors = ['blue', 'red', 'green']

    # ---------------------------------------------------------
    # 1. BASELINE RUNS
    # ---------------------------------------------------------
    print("\n=============================================")
    print("--- BASELINE EVALUATIONS ---")
    TSRs = [6, 8, 10]
    base_kwargs = {'N': 30, 'N_wake': 5, 'dpsi_deg': 10.0, 'a_w': 0.25, 'distribution': 'cosine'}
    baseline_res = {}
    
    for tsr in TSRs:
        print(f"\nTSR {tsr}:")
        baseline_res[tsr] = run_case(U_inf, tsr, R, N_blades, polar_data, base_kwargs)
        
    # --- Baseline Plots (Lines only, No markers) ---
    # 1. Angles
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, tsr in enumerate(TSRs):
        c = colors[i % len(colors)]
        axes[0].plot(baseline_res[tsr]['r_R'], baseline_res[tsr]['phi'], color=c, linestyle='-', marker='', label=f'TSR {tsr}')
        axes[1].plot(baseline_res[tsr]['r_R'], baseline_res[tsr]['alpha'], color=c, linestyle='-', marker='', label=f'TSR {tsr}')
    axes[0].set(xlabel='r/R', ylabel='Inflow Angle (deg)', title='Radial Inflow Angle')
    axes[1].set(xlabel='r/R', ylabel='Angle of Attack (deg)', title='Radial Angle of Attack')
    axes[0].grid(True); axes[0].legend(); axes[1].grid(True); axes[1].legend()
    fig.savefig(os.path.join(out_dir, "Baseline_1_Angles.pdf"), bbox_inches='tight'); plt.close()

    # 2. Induction
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, tsr in enumerate(TSRs):
        c = colors[i % len(colors)]
        axes[0].plot(baseline_res[tsr]['r_R'], baseline_res[tsr]['a'], color=c, linestyle='-', marker='', label=f'TSR {tsr}')
        axes[1].plot(baseline_res[tsr]['r_R'], baseline_res[tsr]['aline'], color=c, linestyle='-', marker='', label=f'TSR {tsr}')
    axes[0].set(xlabel='r/R', ylabel='Axial Induction a', title='Axial Induction')
    axes[1].set(xlabel='r/R', ylabel='Tangential Induction a\'', title='Tangential Induction')
    axes[0].grid(True); axes[0].legend(); axes[1].grid(True); axes[1].legend()
    fig.savefig(os.path.join(out_dir, "Baseline_2_Induction.pdf"), bbox_inches='tight'); plt.close()

    # 3. Loading
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    norm_F = 0.5 * 1.0 * U_inf**2 * R
    for i, tsr in enumerate(TSRs):
        c = colors[i % len(colors)]
        axes[0].plot(baseline_res[tsr]['r_R'], baseline_res[tsr]['F_ax'] / norm_F, color=c, linestyle='-', marker='', label=f'TSR {tsr}')
        axes[1].plot(baseline_res[tsr]['r_R'], baseline_res[tsr]['F_az'] / norm_F, color=c, linestyle='-', marker='', label=f'TSR {tsr}')
    axes[0].set(xlabel='r/R', ylabel='F_axial (non-dim)', title='Axial Loading')
    axes[1].set(xlabel='r/R', ylabel='F_azim (non-dim)', title='Tangential Loading')
    axes[0].grid(True); axes[0].legend(); axes[1].grid(True); axes[1].legend()
    fig.savefig(os.path.join(out_dir, "Baseline_3_Loading.pdf"), bbox_inches='tight'); plt.close()

    # 4. Circulation
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, tsr in enumerate(TSRs):
        c = colors[i % len(colors)]
        Omega = tsr * U_inf / R
        norm_Gamma = (N_blades * Omega) / (math.pi * U_inf**2)
        ax.plot(baseline_res[tsr]['r_R'], baseline_res[tsr]['Gamma'] * norm_Gamma, color=c, linestyle='-', marker='', label=f'TSR {tsr}')
    ax.set(xlabel='r/R', ylabel='Circulation (non-dim)', title='Radial Circulation')
    ax.grid(True); ax.legend()
    fig.savefig(os.path.join(out_dir, "Baseline_4_Circulation.pdf"), bbox_inches='tight'); plt.close()

    # 5. Performance
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    CTs = [baseline_res[tsr]['CT'] for tsr in TSRs]
    CPs = [baseline_res[tsr]['CP'] for tsr in TSRs]
    axes[0].plot(TSRs, CTs, color='blue', linestyle='-', marker='')
    axes[1].plot(TSRs, CPs, color='red', linestyle='-', marker='')
    axes[0].set(xlabel='TSR', ylabel='C_T', title='Thrust Coefficient')
    axes[1].set(xlabel='TSR', ylabel='C_P', title='Power Coefficient')
    axes[0].grid(True); axes[1].grid(True)
    fig.savefig(os.path.join(out_dir, "Baseline_5_Performance.pdf"), bbox_inches='tight'); plt.close()

    # ---------------------------------------------------------
    # 2. SENSITIVITY SWEEPS (TSR = 8)
    # ---------------------------------------------------------
    print("\n=============================================")
    print("--- RUNNING SENSITIVITY SWEEPS (TSR=8) ---")
    tsr_sens = 8
    
    # Sens 1: Number of Panels & Distribution Type
    print("\nEvaluating Sensitivity: Discretization (Constant vs Cosine)")
    N_list = [10, 20, 30]
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, n in enumerate(N_list):
        c = colors[i % len(colors)]
        kw = base_kwargs.copy(); kw['N'] = n
        res = run_case(U_inf, tsr_sens, R, N_blades, polar_data, kw)
        ax.plot(res['r_R'], res['F_ax'] / norm_F, color=c, linestyle='-', marker='', label=f'Cosine, N={n}')
    
    kw_uni = base_kwargs.copy(); kw_uni['distribution'] = 'uniform'
    res_uni = run_case(U_inf, tsr_sens, R, N_blades, polar_data, kw_uni)
    ax.plot(res_uni['r_R'], res_uni['F_ax'] / norm_F, color='black', linestyle='--', marker='', label='Uniform, N=30')
    ax.set(xlabel='r/R', ylabel='Axial Force', title='Sensitivity: Panels & Distribution')
    ax.legend(); ax.grid(True)
    fig.savefig(os.path.join(out_dir, "Sens_1_Panels.pdf"), bbox_inches='tight'); plt.close()

    # Sens 2: Convection Speed (a_w)
    print("\nEvaluating Sensitivity: Convection Speed (a_w)")
    aw_list = [0.1, 0.25, 0.4]
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, aw in enumerate(aw_list):
        c = colors[i % len(colors)]
        kw = base_kwargs.copy(); kw['a_w'] = aw
        res = run_case(U_inf, tsr_sens, R, N_blades, polar_data, kw)
        ax.plot(res['r_R'], res['F_ax'] / norm_F, color=c, linestyle='-', marker='', label=f'a_w={aw}')
    ax.set(xlabel='r/R', ylabel='Axial Force', title='Sensitivity: Convection Speed (a_w)')
    ax.legend(); ax.grid(True)
    fig.savefig(os.path.join(out_dir, "Sens_2_Convection.pdf"), bbox_inches='tight'); plt.close()

    # Sens 3: Azimuthal step size (dpsi)
    print("\nEvaluating Sensitivity: Azimuthal Discretization")
    dpsi_list = [5.0, 10.0, 20.0]
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, dpsi in enumerate(dpsi_list):
        c = colors[i % len(colors)]
        kw = base_kwargs.copy(); kw['dpsi_deg'] = dpsi
        res = run_case(U_inf, tsr_sens, R, N_blades, polar_data, kw)
        ax.plot(res['r_R'], res['F_ax'] / norm_F, color=c, linestyle='-', marker='', label=f'dpsi={dpsi} deg')
    ax.set(xlabel='r/R', ylabel='Axial Force', title='Sensitivity: Azimuthal Step (dpsi)')
    ax.legend(); ax.grid(True)
    fig.savefig(os.path.join(out_dir, "Sens_3_Azimuthal.pdf"), bbox_inches='tight'); plt.close()

    # Sens 4: Wake Length (Rotations)
    print("\nEvaluating Sensitivity: Length of Wake")
    nwake_list = [1, 3, 5]
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, nw in enumerate(nwake_list):
        c = colors[i % len(colors)]
        kw = base_kwargs.copy(); kw['N_wake'] = nw
        res = run_case(U_inf, tsr_sens, R, N_blades, polar_data, kw)
        ax.plot(res['r_R'], res['F_ax'] / norm_F, color=c, linestyle='-', marker='', label=f'N_wake={nw}')
    ax.set(xlabel='r/R', ylabel='Axial Force', title='Sensitivity: Wake Length')
    ax.legend(); ax.grid(True)
    fig.savefig(os.path.join(out_dir, "Sens_4_WakeLength.pdf"), bbox_inches='tight'); plt.close()
    
    print(f"\nExecution Complete! All plots safely stored in: {out_dir}")