"""Run the Assignment-1 BEM model at lambda in {6, 8, 10} and save spanwise
CSVs in figures/ for the lifting-line vs BEM comparison plots.

Re-implements the BEM core from Assignment 1 here (rather than importing
bem_solver_turbine.run_bem_solver) because that function only returns
Ct, Cp, and a' — we need the full per-radius vector of a, a', Gamma_hat,
F_hat_ax, F_hat_azim. The BEM physics is identical to Assignment 1; only
the post-processing is extended.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# ---- Constants matching Assignment 1 wind-turbine spec ----------------
R       = 50.0
R_ROOT  = 0.2 * R
B       = 3
U_INF   = 10.0
RHO     = 1.225
PITCH_DEG = -2.0

POLAR_PATH = os.path.join(os.path.dirname(__file__), "polar DU95W180 (3).xlsx")
FIGDIR     = os.path.join(os.path.dirname(__file__), "figures")
N_ANNULI   = 100   # cosine spacing along the span (BEM convention)


def load_polar(path):
    df = pd.read_excel(path, skiprows=3)
    df.columns = [str(c).strip() for c in df.columns]
    cl_func = interp1d(df["Alfa"], df["Cl"], kind="linear", fill_value="extrapolate")
    cd_func = interp1d(df["Alfa"], df["Cd"], kind="linear", fill_value="extrapolate")
    return cl_func, cd_func


def prandtl_correction(r_R, root_R, TSR, B, a):
    a = np.clip(a, -0.9, 0.99)
    d = 2.0 * np.pi / B * (1 - a) / np.sqrt(TSR**2 + (1 - a)**2)
    Ftip  = 2 / np.pi * np.arccos(np.exp(np.clip(-np.pi * (1 - r_R) / d,  -500, 0)))
    Froot = 2 / np.pi * np.arccos(np.exp(np.clip(-np.pi * (r_R - root_R) / d, -500, 0)))
    Ftip  = float(np.where(np.isnan(Ftip),  0, Ftip))
    Froot = float(np.where(np.isnan(Froot), 0, Froot))
    return Ftip * Froot


def make_blade(num_annuli):
    """Cosine-spaced radial nodes; chord and twist from the assignment."""
    nodes = R_ROOT + (R - R_ROOT) * (1 - np.cos(np.linspace(0, np.pi, num_annuli))) / 2
    dr    = np.diff(nodes)
    r_mid = nodes[:-1] + dr / 2

    c_mid     = 3.0 * (1 - r_mid / R) + 1.0
    twist_mid = np.radians(14.0 * (1 - r_mid / R) + PITCH_DEG)   # twist + pitch [rad]

    return r_mid, dr, c_mid, twist_mid


def run_bem(TSR, r_mid, dr, c_mid, twist_mid, cl_func, cd_func,
            relax=0.25, max_iter=300, tol=1e-5):
    """Per-annulus BEM with Glauert + Prandtl corrections (matches Assignment 1)."""
    omega   = TSR * U_INF / R
    n       = len(r_mid)
    a_arr   = np.zeros(n)
    ap_arr  = np.zeros(n)

    alpha_arr = np.zeros(n)
    phi_arr   = np.zeros(n)
    Cl_arr    = np.zeros(n)
    Cd_arr    = np.zeros(n)
    F_arr     = np.zeros(n)
    Vrel_arr  = np.zeros(n)
    dTdr_arr  = np.zeros(n)
    dQdr_arr  = np.zeros(n)

    Ct_prev = 0.0
    A_disk  = np.pi * R**2

    for it in range(max_iter):
        for i in range(n):
            r, c, beta = r_mid[i], c_mid[i], twist_mid[i]
            a, ap      = a_arr[i], ap_arr[i]

            sigma = B * c / (2 * np.pi * r)

            V_ax  = U_INF * (1 - a)
            V_tan = omega * r * (1 + ap)
            V_rel = np.hypot(V_ax, V_tan)
            phi   = np.arctan2(V_ax, V_tan)
            phi   = max(phi, 1e-5)

            alpha = phi - beta
            Cl = float(cl_func(np.degrees(alpha)))
            Cd = float(cd_func(np.degrees(alpha)))

            Cn = Cl * np.cos(phi) + Cd * np.sin(phi)
            Ct_aero = Cl * np.sin(phi) - Cd * np.cos(phi)

            F = max(prandtl_correction(r / R, R_ROOT / R, TSR, B, a), 1e-4)

            CT_loc = sigma * (1 - a) ** 2 * Cn / np.sin(phi) ** 2
            if CT_loc < 2 * np.sqrt(1.816) - 1.816:
                a_new = 0.5 - np.sqrt(1 - CT_loc) / 2
            else:
                a_new = 1 + (CT_loc - 1.816) / (4 * np.sqrt(1.816) - 4)

            ap_new = 1.0 / ((4 * np.sin(phi) * np.cos(phi) / (sigma * Ct_aero)) - 1)

            a_new  = np.clip(a_new  / F, -0.5, 0.95)
            ap_new = np.clip(ap_new / F, -0.5, 0.95)

            a_arr[i]   = (1 - relax) * a  + relax * a_new
            ap_arr[i]  = (1 - relax) * ap + relax * ap_new
            alpha_arr[i] = alpha
            phi_arr[i]   = phi
            Cl_arr[i]    = Cl
            Cd_arr[i]    = Cd
            F_arr[i]     = F
            Vrel_arr[i]  = V_rel
            dTdr_arr[i]  = 0.5 * RHO * V_rel ** 2 * B * c * Cn
            dQdr_arr[i]  = 0.5 * RHO * V_rel ** 2 * B * c * Ct_aero * r

        Ct_iter = np.sum(dTdr_arr * dr) / (0.5 * RHO * A_disk * U_INF ** 2)
        if it > 0 and abs(Ct_iter - Ct_prev) < tol:
            break
        Ct_prev = Ct_iter

    # ---- Integral coefficients ----
    T = np.sum(dTdr_arr * dr)
    Q = np.sum(dQdr_arr * dr)
    P = Q * omega
    CT = T / (0.5 * RHO * A_disk * U_INF ** 2)
    CP = P / (0.5 * RHO * A_disk * U_INF ** 3)

    # ---- Spanwise non-dim quantities matching the LL output ----
    # F_hat_ax/azim are PER-BLADE forces / (0.5*rho*U_inf^2*R)
    F_axial_per_blade = dTdr_arr / B                                     # N/m per blade
    F_azim_per_blade  = dQdr_arr / (B * r_mid)                           # N/m per blade (azimuthal force, NOT torque)
    F_hat_ax    = F_axial_per_blade / (0.5 * RHO * U_INF ** 2 * R)
    F_hat_azim  = F_azim_per_blade  / (0.5 * RHO * U_INF ** 2 * R)

    # Bound circulation per blade: Gamma = 0.5 * c * V_rel * Cl  (eq:gamma-closure)
    Gamma     = 0.5 * c_mid * Vrel_arr * Cl_arr
    Gamma_hat = Gamma * B * omega / (np.pi * U_INF ** 2)                 # eq:gamma-hat

    CQ = Q / (0.5 * RHO * A_disk * U_INF**2 * R)

    return {
        "r_R":       r_mid / R,
        "a":         a_arr,
        "a_prime":   ap_arr,
        "alpha_deg": np.degrees(alpha_arr),
        "phi_deg":   np.degrees(phi_arr),
        "Gamma":     Gamma,
        "Gamma_hat": Gamma_hat,
        "F_hat_ax":  F_hat_ax,
        "F_hat_azim":F_hat_azim,
        "CT":        CT,
        "CP":        CP,
        "CQ":        CQ,
        "iters":     it + 1,
    }


def main():
    print(f"Loading polar from {POLAR_PATH}")
    cl_func, cd_func = load_polar(POLAR_PATH)
    r_mid, dr, c_mid, twist_mid = make_blade(N_ANNULI)
    os.makedirs(FIGDIR, exist_ok=True)

    print(f"\nBEM cosine, N_annuli={N_ANNULI}, mid-points only")
    print("-" * 60)

    # ---- Spanwise CSVs for {6, 8, 10} (used by LL comparison plots) ----
    summary_rows = []
    for TSR in [6.0, 8.0, 10.0]:
        res = run_bem(TSR, r_mid, dr, c_mid, twist_mid, cl_func, cd_func)
        print(f"  lam={TSR:.0f}: CT={res['CT']:.4f}  CP={res['CP']:.4f}"
              f"  CQ={res['CQ']:.4f}  iters={res['iters']}")

        df = pd.DataFrame({
            "r_R":        res["r_R"],
            "a":          res["a"],
            "a_prime":    res["a_prime"],
            "alpha_deg":  res["alpha_deg"],
            "phi_deg":    res["phi_deg"],
            "Gamma":      res["Gamma"],
            "Gamma_hat":  res["Gamma_hat"],
            "F_hat_ax":   res["F_hat_ax"],
            "F_hat_azim": res["F_hat_azim"],
        })
        outpath = os.path.join(FIGDIR, f"bem_lam{int(TSR)}.csv")
        df.to_csv(outpath, index=False)
        print(f"           -> {outpath}  ({len(df)} rows)")
        summary_rows.append({"lam": TSR, "CT": res["CT"], "CP": res["CP"]})

    summary_path = os.path.join(FIGDIR, "bem_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"  -> {summary_path}")

    # ---- TSR sweep 6.0 … 10.0 step 0.25 (used by run_tsr_sweep.py) ----
    print("\nTSR sweep for CT/CP/CQ curves ...")
    print("-" * 60)
    sweep_rows = []
    for TSR in np.arange(6.0, 10.0 + 1e-9, 0.25):
        res = run_bem(TSR, r_mid, dr, c_mid, twist_mid, cl_func, cd_func)
        print(f"  lam={TSR:.2f}: CT={res['CT']:.4f}  CP={res['CP']:.4f}  CQ={res['CQ']:.4f}")
        sweep_rows.append({"lam": TSR, "CT": res["CT"], "CP": res["CP"], "CQ": res["CQ"]})

    sweep_path = os.path.join(FIGDIR, "bem_tsr_sweep.csv")
    pd.DataFrame(sweep_rows).to_csv(sweep_path, index=False)
    print(f"  -> {sweep_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
