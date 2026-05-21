"""Module 2 — Biot–Savart kernel for a straight vortex filament.

Implements the Katz & Plotkin closed-form expression (eq:katz-plotkin) via
the scalar prefactor K (eq:K-scalar) and component formula (eq:UVW-components).

The function is the innermost routine called during influence-matrix assembly
(Module 3) and must therefore be fully vectorised over control points.
"""

import warnings
import numpy as np


def filament_velocity(
    X1: np.ndarray,
    X2: np.ndarray,
    XP: np.ndarray,
    GAMMA: float,
    r_core: float = 0.0,
    delta_frac: float = 1e-4,
    cutoff_mode: str = "geometric",
) -> np.ndarray:
    """Velocity (u, v, w) at XP induced by straight vortex filament X1→X2.

    Implements eq:katz-plotkin factored through eq:K-scalar and
    eq:UVW-components with the algebraic (δ²) vortex-core regularisation:
    the kernel  Γ dl×r / [4π(|r|²+δ²)^{3/2}]  is integrated analytically
    over the segment, giving a smooth velocity field with no singularity.

    Two regularisation modes are supported:

    ``cutoff_mode="geometric"`` (default, Van Garrel 2003 convention):
        δ = delta_frac * l_0  where l_0 = ||X2 - X1|| is the *per-segment*
        filament length computed at evaluation time from the endpoints.
        Typical values: bound/LL segments delta_frac = 1e-4,
                        wake segments      delta_frac = 0.05.
        Ref: Van Garrel (2003) ECN-C--03-079 "Development of a Wind Turbine
        Aerodynamics Simulation Module", §3.2.

    ``cutoff_mode="fixed"``:
        δ = r_core (absolute metres, original behaviour, kept for comparison).

    Parameters
    ----------
    X1, X2      : (3,) endpoints of the filament
    XP          : (3,) or (M, 3) query point(s)
    GAMMA       : circulation strength [m²/s]
    r_core      : vortex core radius [m]; only used when cutoff_mode="fixed"
    delta_frac  : dimensionless cut-off fraction; only used when cutoff_mode="geometric"
    cutoff_mode : "geometric" (default) or "fixed"

    Returns
    -------
    uvw : (3,) or (M, 3) induced velocity [m/s]
    """
    scalar_input = XP.ndim == 1
    if scalar_input:
        XP = XP[np.newaxis, :]   # (1, 3)

    d = X2 - X1                  # filament direction vector, shape (3,)
    d_norm = np.linalg.norm(d)

    R1 = XP - X1                 # (M, 3)  eq:katz-plotkin R1
    R2 = XP - X2                 # (M, 3)  eq:katz-plotkin R2

    cross = np.cross(R1, R2)     # (M, 3)  R1 × R2
    cross_sq = np.einsum("mi,mi->m", cross, cross)  # (M,)  |R1×R2|²

    R1_norm = np.linalg.norm(R1, axis=1)  # (M,)
    R2_norm = np.linalg.norm(R2, axis=1)  # (M,)

    # Algebraic vortex-core regularisation (δ² in denominator, lecture formula):
    # integrating  dv = Γ dl×r / [4π(|r|²+δ²)^{3/2}]  over the straight segment
    # yields the K&P closed form with two substitutions:
    #   |R1×R2|²  →  |R1×R2|² + δ²·|d|²   (perpendicular distance: h²→h²+δ²)
    #   |Ri|      →  √(|Ri|² + δ²)          (cos-angle denom: √(h²+li²)→√(h²+δ²+li²))
    if cutoff_mode == "geometric":
        # Van Garrel (2003) ECN-C--03-079 §3.2: δ_eff = delta_frac · l_0,
        # where l_0 is the per-segment filament length (computed from endpoints,
        # not precomputed, so uneven helix discretisation is handled correctly).
        l0 = d_norm
        if l0 < 1e-10:
            warnings.warn(
                f"Degenerate vortex filament segment (l0={l0:.2e} m < 1e-10); "
                "clamping to 1e-10 m to avoid NaN.",
                stacklevel=2,
            )
            l0 = 1e-10
        delta_sq = (delta_frac * l0) ** 2
    else:  # "fixed"
        delta_sq = r_core ** 2

    cross_sq_reg = cross_sq + delta_sq * d_norm ** 2
    R1_norm_reg  = np.sqrt(R1_norm ** 2 + delta_sq)
    R2_norm_reg  = np.sqrt(R2_norm ** 2 + delta_sq)

    # scalar K from eq:K-scalar
    d_dot_R1 = np.dot(R1, d)             # (M,)  (X2-X1)·R1
    d_dot_R2 = np.dot(R2, d)             # (M,)  (X2-X1)·R2

    K = (GAMMA / (4.0 * np.pi * cross_sq_reg)) * (
        d_dot_R1 / R1_norm_reg - d_dot_R2 / R2_norm_reg
    )

    uvw = K[:, np.newaxis] * cross       # (M, 3)  eq:UVW-components

    if scalar_input:
        return uvw[0]
    return uvw


if __name__ == "__main__":
    # Sanity test: long finite filament ≈ infinite straight filament.
    #
    # An infinite filament along the x-axis with Γ=1 induces at (0,d,0):
    #   |u_ind| = Γ / (2π d)  in the +z direction (right-hand rule).
    # A finite filament from (-L,0,0) to (L,0,0) with L >> d gives the
    # same result to O(d²/L²).
    #
    # NOTE: the long-filament test uses cutoff_mode="fixed" with a tiny core
    # because in geometric mode delta = delta_frac*l0 = 1e-4*2e4 = 2 m which
    # is comparable to d_perp=1 m and would smear the result.

    GAMMA = 1.0
    d_perp = 1.0
    L = 1e4

    X1 = np.array([-L, 0.0, 0.0])
    X2 = np.array([+L, 0.0, 0.0])
    XP = np.array([0.0, d_perp, 0.0])
    r_core = 1e-6

    uvw = filament_velocity(X1, X2, XP, GAMMA, r_core=r_core, cutoff_mode="fixed")
    exact = GAMMA / (2.0 * np.pi * d_perp)

    print("=== Biot–Savart sanity check (fixed mode) ===")
    print(f"  computed  uvw = {uvw}")
    print(f"  expected  u=0, v=0, w={exact:.6f}")
    err = abs(uvw[2] - exact) / exact
    print(f"  relative error in w: {err:.2e}")
    assert err < 1e-3, f"Error too large: {err}"
    assert abs(uvw[0]) < 1e-10 and abs(uvw[1]) < 1e-10, "Non-zero u or v"
    print("  PASSED")

    # Vectorised call: two control points at once
    XP_multi = np.array([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]])
    uvw_multi = filament_velocity(X1, X2, XP_multi, GAMMA, r_core=r_core, cutoff_mode="fixed")
    exact2 = GAMMA / (2.0 * np.pi * 2.0)
    err2 = abs(uvw_multi[1, 2] - exact2) / exact2
    print(f"  vectorised (d=2): w={uvw_multi[1,2]:.6f}  exact={exact2:.6f}  err={err2:.2e}")
    assert err2 < 1e-3
    print("  VECTORISED PASSED")

    # Core test: point on the filament axis should return zero (cross=0 in both modes)
    XP_core = np.array([0.0, 0.0, 0.0])
    uvw_core = filament_velocity(X1, X2, XP_core, GAMMA, r_core=1.0)
    print(f"  core test (XP on axis): uvw = {uvw_core}  (expected zeros)")
    assert np.allclose(uvw_core, 0.0), "Core regularisation failed"
    print("  CORE PASSED")

    # Geometric mode test: short filament with tiny delta_frac → same result
    L_short = 100.0
    X1s = np.array([-L_short, 0.0, 0.0])
    X2s = np.array([+L_short, 0.0, 0.0])
    # delta = 1e-6 * 200 = 2e-4 m << d_perp=1 m, so geometric ≈ fixed
    uvw_geo = filament_velocity(X1s, X2s, XP, GAMMA, delta_frac=1e-6, cutoff_mode="geometric")
    # Exact analytic result for finite filament: Γ*L / (2π*h*√(L²+h²))
    exact_short = GAMMA * L_short / (2.0 * np.pi * d_perp * np.sqrt(L_short**2 + d_perp**2))
    err_geo = abs(uvw_geo[2] - exact_short) / exact_short
    print(f"\n=== Biot-Savart sanity check (geometric mode) ===")
    print(f"  l0={2*L_short:.0f} m, delta_frac=1e-6 -> delta={1e-6*2*L_short:.2e} m")
    print(f"  computed w={uvw_geo[2]:.6f}  exact={exact_short:.6f}  err={err_geo:.2e}")
    assert err_geo < 1e-3, f"Geometric mode error too large: {err_geo}"
    print("  GEOMETRIC MODE PASSED")
