import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# 1. AIRFOIL POLAR (MODULE 4)
# =============================================================================
class AirfoilPolar:
    def __init__(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Missing Polar File: {filepath}")
        # header=3 equivalent to skiprows=3 to skip text metadata
        df = pd.read_excel(filepath, sheet_name=0, skiprows=3)
        self.alpha_rad = np.radians(df.iloc[:, 0].to_numpy(dtype=float))
        self.Cl_data = df.iloc[:, 1].to_numpy(dtype=float)
        self.Cd_data = df.iloc[:, 2].to_numpy(dtype=float)

    def get_coefficients(self, alpha):
        cl = np.interp(alpha, self.alpha_rad, self.Cl_data)
        cd = np.interp(alpha, self.alpha_rad, self.Cd_data)
        return cl, cd

# =============================================================================
# 2. GEOMETRY AND WAKE (MODULE 1)
# =============================================================================
class RotorGeometry:
    def __init__(self, R=50.0, r_root=10.0, N=30, N_blades=3, N_wake=5, 
                 dpsi_deg=10.0, U_inf=10.0, Omega=1.6, a_w=0.25, dist="cosine"):
        self.R = R
        self.r_root = r_root
        self.N = N
        self.N_blades = N_blades
        self.Omega = Omega
        self.U_wake = U_inf * (1.0 - a_w)
        self.pitch = np.radians(-2.0)
        
        # Spanwise Discretization
        if dist == "cosine":
            i = np.arange(N + 1)
            self.r_node = r_root + 0.5 * (R - r_root) * (1 - np.cos(math.pi * i / N))
        else:
            self.r_node = np.linspace(r_root, R, N + 1)
            
        self.r_mid = 0.5 * (self.r_node[:-1] + self.r_node[1:])
        
        # Blade Properties
        self.chord_mid = 3.0 * (1 - self.r_mid / R) + 1.0
        self.twist_mid = np.radians(14.0 * (1 - self.r_mid / R))
        
        # Nodes and Control Points (Exactly as guide: both on z-axis initially)
        self.bound = np.zeros((N + 1, 3))
        self.bound[:, 2] = self.r_node
        self.cp = np.zeros((N, 3))
        self.cp[:, 2] = self.r_mid
        
        # Wake Discretization
        self.dpsi = math.radians(dpsi_deg)
        self.N_az = int((N_wake * 2 * math.pi) / self.dpsi)
        self.k_arr = np.arange(self.N_az + 1)
        self.t_k = self.k_arr * self.dpsi / self.Omega
        
        self.build_wake()

    def build_wake(self):
        chord_node = 3.0 * (1 - self.r_node / self.R) + 1.0
        twist_node = np.radians(14.0 * (1 - self.r_node / self.R))
        
        # Exact trailing edge offset prescribed by the guide
        dx_node = chord_node * np.sin(twist_node)
        dy_node = -chord_node * np.cos(twist_node)
        
        self.wake = np.zeros((self.N_blades, self.N + 1, self.N_az + 1, 3))
        for b in range(self.N_blades):
            psi_0 = 2 * math.pi * b / self.N_blades
            psi_k = psi_0 - self.k_arr * self.dpsi
            
            self.wake[b, :, :, 0] = dx_node[:, None] + self.U_wake * self.t_k[None, :]
            self.wake[b, :, :, 1] = dy_node[:, None] * np.cos(psi_k)[None, :] + self.r_node[:, None] * np.sin(psi_k)[None, :]
            self.wake[b, :, :, 2] = -dy_node[:, None] * np.sin(psi_k)[None, :] + self.r_node[:, None] * np.cos(psi_k)[None, :]

# =============================================================================
# 3. BIOT-SAVART AND INFLUENCE MATRICES (MODULE 2 & 3)
# =============================================================================
class InfluenceAssembler:
    @staticmethod
    def filament_velocity(X1, X2, XP, delta_frac=1e-4):
        """Katz & Plotkin exact closed-form expression with Van Garrel regularization."""
        d = X2 - X1
        R1 = XP - X1
        R2 = XP - X2
        
        cross = np.cross(R1, R2)
        cross_sq = np.einsum("mi,mi->m", cross, cross)
        
        R1_norm = np.linalg.norm(R1, axis=1)
        R2_norm = np.linalg.norm(R2, axis=1)
        d_norm = np.linalg.norm(d)
        
        # Algebraic regularization (geometric mode)
        delta_sq = (delta_frac * d_norm) ** 2
        cross_sq_reg = cross_sq + delta_sq * d_norm**2
        R1_norm_reg = np.sqrt(R1_norm**2 + delta_sq)
        R2_norm_reg = np.sqrt(R2_norm**2 + delta_sq)
        
        R0R1 = np.einsum("i,mi->m", d, R1)
        R0R2 = np.einsum("i,mi->m", d, R2)
        
        K = (1.0 / (4 * math.pi * cross_sq_reg)) * (R0R1 / R1_norm_reg - R0R2 / R2_norm_reg)
        return K[:, None] * cross

    @staticmethod
    def build_matrices(geom):
        N, N_blades, N_az = geom.N, geom.N_blades, geom.N_az
        U_mat, V_mat, W_mat = np.zeros((N, N)), np.zeros((N, N)), np.zeros((N, N))
        
        # Pre-rotate bound nodes as exactly specified in the guide
        bound_all = np.zeros((N_blades, N + 1, 3))
        for b in range(N_blades):
            psi_b = 2 * math.pi * b / N_blades
            bound_all[b, :, 0] = geom.bound[:, 0]
            bound_all[b, :, 1] = geom.bound[:, 1] * math.cos(psi_b) + geom.bound[:, 2] * math.sin(psi_b)
            bound_all[b, :, 2] = -geom.bound[:, 1] * math.sin(psi_b) + geom.bound[:, 2] * math.cos(psi_b)
            
        for j in range(N):
            vel = np.zeros((N, 3))
            for b in range(N_blades):
                # 1. Bound Segment
                vel += InfluenceAssembler.filament_velocity(bound_all[b, j], bound_all[b, j+1], geom.cp, delta_frac=1e-4)
                
                # 2 & 4. Connector Filaments
                vel += InfluenceAssembler.filament_velocity(bound_all[b, j], geom.wake[b, j, 0], geom.cp, delta_frac=1e-4)
                vel -= InfluenceAssembler.filament_velocity(bound_all[b, j+1], geom.wake[b, j+1, 0], geom.cp, delta_frac=1e-4)
                
                # 3 & 5. Helical Wake
                for k in range(N_az):
                    vel += InfluenceAssembler.filament_velocity(geom.wake[b, j, k], geom.wake[b, j, k+1], geom.cp, delta_frac=0.05)
                    vel -= InfluenceAssembler.filament_velocity(geom.wake[b, j+1, k], geom.wake[b, j+1, k+1], geom.cp, delta_frac=0.05)
                    
            U_mat[:, j], V_mat[:, j], W_mat[:, j] = vel[:, 0], vel[:, 1], vel[:, 2]
            
        return U_mat, V_mat, W_mat

# =============================================================================
# 4. SOLVER & POST-PROCESSING (MODULE 5 & 6)
# =============================================================================
class LLMSolver:
    def __init__(self, U_inf, polar, geom, U_mat, V_mat, W_mat):
        self.U_inf = U_inf
        self.polar = polar
        self.geom = geom
        self.U_mat = U_mat
        self.V_mat = V_mat
        self.W_mat = W_mat
        
    def solve(self, max_iter=1000, tol=1e-4):
        N = self.geom.N
        Gamma = np.zeros(N)
        
        # Aitken Dynamic Vector Relaxation
        w_dyn = 0.3
        R_prev = np.zeros(N)
        
        for k in range(max_iter):
            u_ind = self.U_mat @ Gamma
            v_ind = self.V_mat @ Gamma
            
            V_ax = self.U_inf + u_ind
            V_tan = self.geom.Omega * self.geom.r_mid - v_ind
            V_p = np.sqrt(V_ax**2 + V_tan**2)
            
            phi = np.arctan2(V_ax, V_tan)
            alpha = phi - self.geom.twist_mid - self.geom.pitch
            
            Cl, _ = self.polar.get_coefficients(alpha)
            Gamma_new = 0.5 * self.geom.chord_mid * V_p * Cl
            
            R_k = Gamma_new - Gamma
            
            # Convergence Check
            res = np.max(np.abs(R_k)) / max(np.max(np.abs(Gamma_new)), 1e-12)
            if res < tol:
                break
                
            # Update Aitken Weight
            if k > 0:
                delta_R = R_k - R_prev
                den = np.dot(delta_R, delta_R)
                if den > 1e-16:
                    w_dyn = -w_dyn * np.dot(R_prev, delta_R) / den
                w_dyn = np.clip(w_dyn, 0.01, 1.0)
                
            Gamma += w_dyn * R_k
            R_prev = R_k.copy()
            
        return self.post_process(Gamma, u_ind, v_ind, V_p, phi, alpha)

    def post_process(self, Gamma, u_ind, v_ind, V_p, phi, alpha):
        rho = 1.0
        Cl, Cd = self.polar.get_coefficients(alpha)
        
        L = 0.5 * rho * V_p**2 * self.geom.chord_mid * Cl
        D = 0.5 * rho * V_p**2 * self.geom.chord_mid * Cd
        
        F_ax = L * np.cos(phi) + D * np.sin(phi)
        F_azim = L * np.sin(phi) - D * np.cos(phi)
        
        # Integration via Trapezoid rule
        T = self.geom.N_blades * trapezoid(F_ax, self.geom.r_mid)
        Q = self.geom.N_blades * trapezoid(self.geom.r_mid * F_azim, self.geom.r_mid)
        P = self.geom.Omega * Q
        
        CT = T / (0.5 * rho * self.U_inf**2 * math.pi * self.geom.R**2)
        CP = P / (0.5 * rho * self.U_inf**3 * math.pi * self.geom.R**2)
        
        return {
            "r_R": self.geom.r_mid / self.geom.R,
            "phi": np.degrees(phi),
            "alpha": np.degrees(alpha),
            "a": -u_ind / self.U_inf,
            "a_prime": -v_ind / (self.geom.Omega * self.geom.r_mid),
            "F_ax": F_ax / (0.5 * rho * self.U_inf**2 * self.geom.R),
            "F_azim": F_azim / (0.5 * rho * self.U_inf**2 * self.geom.R),
            "Gamma_hat": Gamma * self.geom.N_blades * self.geom.Omega / (math.pi * self.U_inf**2),
            "CT": CT,
            "CP": CP
        }

# =============================================================================
# MAIN EXECUTION & SENSITIVITY PLOTTING
# =============================================================================
def run_pipeline(cfg, polar):
    geom = RotorGeometry(**cfg)
    U_m, V_m, W_m = InfluenceAssembler.build_matrices(geom)
    solver = LLMSolver(cfg['U_inf'], polar, geom, U_m, V_m, W_m)
    res = solver.solve()
    print(f"      CT = {res['CT']:.4f}  |  CP = {res['CP']:.4f}")
    return res

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "LLM_Plots_v3")
    os.makedirs(out_dir, exist_ok=True)
    
    polar = AirfoilPolar(os.path.join(script_dir, "polar DU95W180 (3).xlsx"))
    colors = ['blue', 'green', 'red']
    
    base_cfg = {
        "R": 50.0, "r_root": 10.0, "N": 30, "N_blades": 3,
        "N_wake": 5, "dpsi_deg": 10.0, "U_inf": 10.0, "a_w": 0.25, 
        "dist": "cosine"
    }

    # --- 1. BASELINE RUNS ---
    print("\n--- BASELINE EVALUATIONS ---")
    TSRs = [6, 8, 10]
    baseline_res = {}
    for tsr in TSRs:
        print(f"TSR {tsr}:")
        cfg = base_cfg.copy()
        cfg["Omega"] = tsr * base_cfg["U_inf"] / base_cfg["R"]
        baseline_res[tsr] = run_pipeline(cfg, polar)

    # Plotting Baseline (Lines only)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, tsr in enumerate(TSRs):
        c = colors[i % len(colors)]
        axes[0].plot(baseline_res[tsr]['r_R'], baseline_res[tsr]['phi'], color=c, linestyle='-', label=f'$\\lambda={tsr}$')
        axes[1].plot(baseline_res[tsr]['r_R'], baseline_res[tsr]['alpha'], color=c, linestyle='-', label=f'$\\lambda={tsr}$')
    axes[0].set(xlabel='r/R', ylabel='Inflow Angle (deg)', title='Radial Inflow Angle')
    axes[1].set(xlabel='r/R', ylabel='Angle of Attack (deg)', title='Radial Angle of Attack')
    axes[0].grid(); axes[0].legend(); axes[1].grid(); axes[1].legend()
    fig.savefig(os.path.join(out_dir, "Baseline_1_Angles.pdf"), bbox_inches='tight'); plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, tsr in enumerate(TSRs):
        c = colors[i % len(colors)]
        axes[0].plot(baseline_res[tsr]['r_R'], baseline_res[tsr]['a'], color=c, linestyle='-', label=f'$\\lambda={tsr}$')
        axes[1].plot(baseline_res[tsr]['r_R'], baseline_res[tsr]['a_prime'], color=c, linestyle='-', label=f'$\\lambda={tsr}$')
    axes[0].set(xlabel='r/R', ylabel='Axial Induction a', title='Axial Induction')
    axes[1].set(xlabel='r/R', ylabel='Tangential Induction a\'', title='Tangential Induction')
    axes[0].grid(); axes[0].legend(); axes[1].grid(); axes[1].legend()
    fig.savefig(os.path.join(out_dir, "Baseline_2_Induction.pdf"), bbox_inches='tight'); plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, tsr in enumerate(TSRs):
        c = colors[i % len(colors)]
        axes[0].plot(baseline_res[tsr]['r_R'], baseline_res[tsr]['F_ax'], color=c, linestyle='-', label=f'$\\lambda={tsr}$')
        axes[1].plot(baseline_res[tsr]['r_R'], baseline_res[tsr]['F_azim'], color=c, linestyle='-', label=f'$\\lambda={tsr}$')
    axes[0].set(xlabel='r/R', ylabel='F_axial (non-dim)', title='Axial Loading')
    axes[1].set(xlabel='r/R', ylabel='F_azim (non-dim)', title='Tangential Loading')
    axes[0].grid(); axes[0].legend(); axes[1].grid(); axes[1].legend()
    fig.savefig(os.path.join(out_dir, "Baseline_3_Loading.pdf"), bbox_inches='tight'); plt.close()

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, tsr in enumerate(TSRs):
        ax.plot(baseline_res[tsr]['r_R'], baseline_res[tsr]['Gamma_hat'], color=colors[i % len(colors)], linestyle='-', label=f'$\\lambda={tsr}$')
    ax.set(xlabel='r/R', ylabel='Circulation (non-dim)', title='Radial Circulation')
    ax.grid(); ax.legend()
    fig.savefig(os.path.join(out_dir, "Baseline_4_Circulation.pdf"), bbox_inches='tight'); plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    CTs = [baseline_res[tsr]['CT'] for tsr in TSRs]
    CPs = [baseline_res[tsr]['CP'] for tsr in TSRs]
    axes[0].plot(TSRs, CTs, color='blue', marker='o')
    axes[1].plot(TSRs, CPs, color='red', marker='s')
    axes[0].set(xlabel='$\\lambda$', ylabel='$C_T$', title='Thrust Coefficient')
    axes[1].set(xlabel='$\\lambda$', ylabel='$C_P$', title='Power Coefficient')
    axes[0].grid(); axes[1].grid()
    fig.savefig(os.path.join(out_dir, "Baseline_5_Performance.pdf"), bbox_inches='tight'); plt.close()

    # --- 2. SENSITIVITY SWEEPS ---
    print("\n--- SENSITIVITY SWEEPS (TSR=8) ---")
    tsr_sens = 8
    cfg_sens = base_cfg.copy()
    cfg_sens["Omega"] = tsr_sens * base_cfg["U_inf"] / base_cfg["R"]

    print("\nEvaluating Sensitivity: Discretization (Cosine vs Uniform)")
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, n in enumerate([10, 20, 30]):
        c = cfg_sens.copy()
        c["N"] = n
        res = run_pipeline(c, polar)
        ax.plot(res['r_R'], res['F_ax'], color=colors[i % len(colors)], linestyle='-', label=f'Cosine, N={n}')
    c_uni = cfg_sens.copy()
    c_uni["dist"] = "uniform"
    res_uni = run_pipeline(c_uni, polar)
    ax.plot(res_uni['r_R'], res_uni['F_ax'], color='black', linestyle='--', label='Uniform, N=30')
    ax.set(xlabel='r/R', ylabel='Axial Force', title='Sensitivity: Panels & Distribution')
    ax.legend(); ax.grid()
    fig.savefig(os.path.join(out_dir, "Sens_1_Panels.pdf"), bbox_inches='tight'); plt.close()

    print("\nEvaluating Sensitivity: Convection Speed (a_w)")
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, aw in enumerate([0.1, 0.25, 0.4]):
        c = cfg_sens.copy()
        c["a_w"] = aw
        res = run_pipeline(c, polar)
        ax.plot(res['r_R'], res['F_ax'], color=colors[i % len(colors)], linestyle='-', label=f'a_w={aw}')
    ax.set(xlabel='r/R', ylabel='Axial Force', title='Sensitivity: Convection Speed (a_w)')
    ax.legend(); ax.grid()
    fig.savefig(os.path.join(out_dir, "Sens_2_Convection.pdf"), bbox_inches='tight'); plt.close()

    print("\nEvaluating Sensitivity: Azimuthal Discretization")
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, dpsi in enumerate([5.0, 10.0, 20.0]):
        c = cfg_sens.copy()
        c["dpsi_deg"] = dpsi
        res = run_pipeline(c, polar)
        ax.plot(res['r_R'], res['F_ax'], color=colors[i % len(colors)], linestyle='-', label=f'dpsi={dpsi} deg')
    ax.set(xlabel='r/R', ylabel='Axial Force', title='Sensitivity: Azimuthal Step (dpsi)')
    ax.legend(); ax.grid()
    fig.savefig(os.path.join(out_dir, "Sens_3_Azimuthal.pdf"), bbox_inches='tight'); plt.close()

    print("\nEvaluating Sensitivity: Wake Length")
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, nw in enumerate([1, 3, 5]):
        c = cfg_sens.copy()
        c["N_wake"] = nw
        res = run_pipeline(c, polar)
        ax.plot(res['r_R'], res['F_ax'], color=colors[i % len(colors)], linestyle='-', label=f'N_wake={nw}')
    ax.set(xlabel='r/R', ylabel='Axial Force', title='Sensitivity: Wake Length')
    ax.legend(); ax.grid()
    fig.savefig(os.path.join(out_dir, "Sens_4_WakeLength.pdf"), bbox_inches='tight'); plt.close()
    
    print(f"\nExecution Complete! All plots stored in: {out_dir}")