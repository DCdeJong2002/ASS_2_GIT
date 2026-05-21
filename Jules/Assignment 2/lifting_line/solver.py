"""Module 5 — Fixed-point iteration for spanwise circulation.

Implements Algorithm alg:lifting-line from theory_background.tex.

With the influence matrices already in memory the per-iteration cost is
dominated by three O(N²) matrix-vector multiplies (eq:influence-matrix)
plus vectorised polar lookups — O(N) — so the total iteration cost is
O(k_max · N²), far cheaper than recomputing Biot–Savart each iteration.

Implements Algorithm alg:lifting-line from theory_background.tex.
Updated to utilize Vector Aitken Under-Relaxation for dynamic step sizing
to accelerate convergence and avoid oscillatory 2-cycle limits.
"""

import numpy as np
from .polar import airfoil_coefficients


def solve_circulation(
    U_mat: np.ndarray,
    V_mat: np.ndarray,
    W_mat: np.ndarray,
    geom: dict,
    polar: dict,
    U_inf: float,
    Omega: float,
    w_relax: float = 0.3,
    tol: float = 1e-5,
    max_iter: int = 1200,
) -> dict:
    """Iterate to find converged spanwise circulation Gamma.

    Implements alg:lifting-line with dynamic vector Aitken under-relaxation.

    Parameters
    ----------
    U_mat, V_mat, W_mat : (N, N) influence matrices
    geom    : geometry dict from build_geometry (must have r_mid, chord,
              twist, pitch set before calling)
    polar   : dict from load_polar
    U_inf   : freestream speed [m/s]
    Omega   : rotor angular velocity [rad/s]
    w_relax : Initial under-relaxation weight (default 0.1). Aitken will 
              dynamically adjust this value as the iterations progress.
    tol     : convergence threshold on max|ΔΓ|/max|Γ| (default 1e-2)
    max_iter: maximum number of iterations (default 1200)

    Returns
    -------
    dict with keys:
        Gamma     : (N,) converged circulation [m²/s]
        alpha     : (N,) angle of attack [rad]
        phi       : (N,) inflow angle [rad]
        u_ind     : (N,) axial induced velocity [m/s]
        v_ind     : (N,) tangential induced velocity [m/s]
        w_ind     : (N,) radial induced velocity [m/s]
        Cl, Cd    : (N,) lift and drag coefficients
        V_p       : (N,) resultant velocity magnitude [m/s]
        V_ax      : (N,) axial velocity component [m/s]
        V_tan     : (N,) tangential velocity component [m/s]
        iters     : int  number of iterations taken
        converged : bool
        final_w   : float final dynamic relaxation factor used
    """
    r_mid  = geom["r_mid"]
    chord  = geom["chord"]
    twist  = geom["twist"]    # [rad]
    pitch  = geom["pitch"]    # [rad] scalar

    N = len(r_mid)
    Gamma = np.zeros(N)
    converged = False

    # --- AITKEN INITIALIZATION ---
    R_prev = None 
    w_dynamic = w_relax 

    for k in range(max_iter):
        # Influence matrix multiplication — induced velocities at all control points
        u_ind = U_mat @ Gamma   # axial (x-component)
        v_ind = V_mat @ Gamma   # tangential (y-component, ref blade at ψ=0)
        w_ind = W_mat @ Gamma   # radial (z-component)

        # Velocity components and magnitude
        V_ax  = U_inf + u_ind
        V_tan = Omega * r_mid - v_ind
        V_p   = np.hypot(V_ax, V_tan)

        # Inflow angle and angle of attack
        phi   = np.arctan2(V_ax, V_tan)
        alpha = phi - twist - pitch

        # Polar lookup for aerodynamic coefficients
        Cl, Cd = airfoil_coefficients(alpha, polar)

        # Kutta–Joukowski closure for new circulation
        Gamma_new = 0.5 * chord * V_p * Cl

        # --- AITKEN DYNAMIC RELAXATION ---
        R = Gamma_new - Gamma  # Current residual vector
        
        if R_prev is not None:
            delta_R = R - R_prev
            denom_aitken = np.dot(delta_R, delta_R)
            
            # Avoid division by zero in the Aitken update
            if denom_aitken > 1e-16:
                # Aitken update formula for the relaxation factor
                w_dynamic = -w_dynamic * np.dot(R_prev, delta_R) / denom_aitken
                
                # Clamp the factor to prevent instability during highly nonlinear early steps
                w_dynamic = np.clip(w_dynamic, 0.01, 1.0)
                
        # Apply the dynamically computed step
        Gamma_next = Gamma + w_dynamic * R

        # Store current residual for the next iteration's Aitken calculation
        R_prev = R.copy()

        # Convergence check
        denom_conv = max(np.max(np.abs(Gamma_next)), 1e-12)
        res = np.max(np.abs(Gamma_next - Gamma)) / denom_conv
        Gamma = Gamma_next

        if res < tol:
            converged = True
            break

    return {
        "Gamma":     Gamma,
        "alpha":     alpha,
        "phi":       phi,
        "u_ind":     u_ind,
        "v_ind":     v_ind,
        "w_ind":     w_ind,
        "Cl":        Cl,
        "Cd":        Cd,
        "V_p":       V_p,
        "V_ax":      V_ax,
        "V_tan":     V_tan,
        "iters":     k + 1,
        "converged": converged,
        "final_w":   w_dynamic,
    }