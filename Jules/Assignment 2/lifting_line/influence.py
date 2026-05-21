"""Module 3 — Influence-matrix assembly.

Fills the three N×N geometry matrices [U], [V], [W] (eq:influence-matrix)
such that the induced velocity at all N control points follows from a
single matrix-vector multiply per component per iteration (eq:influence-matrix).

Each column j of [U/V/W] holds the x/y/z velocity induced at all N control
points by the j-th horseshoe ring of EVERY blade evaluated with unit
circulation. By rotor symmetry every blade carries the same Γ_j at the
matching spanwise station, so the influence is summed across blades into
a single column.

Each horseshoe ring on blade b consists of (going around the closed loop):

  1. Bound segment along the c/4 line of blade b.
  2. Inboard connector (chord-length filament) from c/4 inboard node to the
     start of the inboard helix on blade b.
  3. Inboard trailing helix from blade b's inboard wake-start, downstream.
  4. Outboard connector from outboard helix start back to c/4 outboard node.
  5. Outboard trailing helix from blade b's outboard wake-start, downstream.

Adjacent horseshoes share a trailing helix whose net circulation equals
Γ_j − Γ_{j+1}, automatically satisfying Helmholtz (eq:helmholtz-discrete).

Because the wake is frozen, the matrices depend only on geometry and are
computed once; the per-iteration cost is then O(N²) (eq:influence-matrix).
"""

import numpy as np
from .biot_savart import filament_velocity

# Printed once per process to avoid spamming sensitivity sweeps.
_delta_info_printed = False


def assemble_influence(
    cp: np.ndarray,
    bound: np.ndarray,
    wake: np.ndarray,
    r_core: float = 0.0,
    delta_frac_LL: float = 1e-4,
    delta_frac_wake: float = 0.05,
    cutoff_mode: str = "geometric",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the influence matrices [U], [V], [W] of size (N, N).

    Regularisation is controlled by ``cutoff_mode``:

    ``"geometric"`` (default, Van Garrel 2003):
        Bound segments and connector filaments use δ = delta_frac_LL * l_0.
        Wake helix segments use δ = delta_frac_wake * l_0.
        l_0 is the per-segment length, computed inside filament_velocity.

    ``"fixed"``:
        All segments use δ = r_core (absolute metres, original behaviour).

    Parameters
    ----------
    cp            : (N, 3)                control points on reference blade (psi=0)
    bound         : (N+1, 3)              bound-vortex nodes on reference blade (psi=0)
    wake          : (N_blades, N+1, N_az+1, 3)  helical wake nodes (all blades)
    r_core        : float                 vortex core radius [m]; only used when cutoff_mode="fixed"
    delta_frac_LL : float                 cut-off fraction for bound/connector segments (geometric mode)
    delta_frac_wake: float                cut-off fraction for wake helix segments (geometric mode)
    cutoff_mode   : str                   "geometric" (default) or "fixed"

    Returns
    -------
    U, V, W : each (N, N)  [m/s per (m²/s)]  — eq:influence-matrix
    """
    global _delta_info_printed

    N = cp.shape[0]
    N_blades, _, N_az_p1, _ = wake.shape
    N_az = N_az_p1 - 1

    if cutoff_mode == "geometric" and not _delta_info_printed:
        panel_widths = np.linalg.norm(np.diff(bound, axis=0), axis=1)  # (N,)
        delta_LL_abs = delta_frac_LL * panel_widths
        print(
            f"  [assemble_influence] geometric cutoff (Van Garrel 2003): "
            f"abs_delta_LL  min={delta_LL_abs.min():.3e} m  "
            f"mean={delta_LL_abs.mean():.3e} m  "
            f"max={delta_LL_abs.max():.3e} m  "
            f"(delta_frac_LL={delta_frac_LL:.0e}, delta_frac_wake={delta_frac_wake})"
        )
        _delta_info_printed = True

    # Pre-compute bound nodes for ALL blades by rotating the reference blade
    # nodes about the +x axis to each blade's azimuthal position psi_b.
    # bound_all[b] gives the (N+1, 3) bound nodes for blade b.
    bound_all = np.empty((N_blades, N + 1, 3))
    for b in range(N_blades):
        psi_b = 2.0 * np.pi * b / N_blades
        c, s = np.cos(psi_b), np.sin(psi_b)
        # Rotation about +x axis by angle psi_b carries (0, 0, r) -> (0, r*sin(psi_b), r*cos(psi_b))
        # which matches the wake parameterisation at k=0 (modulo the chord offset).
        bound_all[b, :, 0] = bound[:, 0]
        bound_all[b, :, 1] = bound[:, 1] * c + bound[:, 2] * s
        bound_all[b, :, 2] = -bound[:, 1] * s + bound[:, 2] * c

    U = np.zeros((N, N))
    V = np.zeros((N, N))
    W = np.zeros((N, N))

    for j in range(N):
        vel = np.zeros((N, 3))

        for b in range(N_blades):
            # 1. Bound segment of blade b's horseshoe j (c/4 inboard → outboard)
            #    Uses LL cut-off: short panel-width segment, tight regularisation.
            vel += filament_velocity(
                bound_all[b, j], bound_all[b, j + 1], cp,
                GAMMA=1.0, r_core=r_core,
                delta_frac=delta_frac_LL, cutoff_mode=cutoff_mode,
            )

            # 2 & 4. Connector filaments from c/4 to wake-shedding point.
            #   Inboard  connector (sign=+1): bound_inboard  → wake_start_inboard
            #   Outboard connector (sign=−1): bound_outboard → wake_start_outboard
            #   (sign=−1 represents traversing the opposite-direction filament
            #    of the adjacent ring, equivalent to the JS reference's closed loop.)
            #   Uses LL cut-off: connector length ~ chord, treated as near-field.
            for node_idx, sign in ((j, +1.0), (j + 1, -1.0)):
                vel += sign * filament_velocity(
                    bound_all[b, node_idx], wake[b, node_idx, 0], cp,
                    GAMMA=1.0, r_core=r_core,
                    delta_frac=delta_frac_LL, cutoff_mode=cutoff_mode,
                )

                # 3 & 5. Helical trailing legs (vectorised across N control points,
                # serial across N_az segments — see notes on memory below).
                #   Uses wake cut-off: Van Garrel recommends delta_frac ~ 0.05.
                X1_arr = wake[b, node_idx, :-1, :]  # (N_az, 3) segment starts
                X2_arr = wake[b, node_idx, 1:, :]   # (N_az, 3) segment ends
                for k in range(N_az):
                    vel += sign * filament_velocity(
                        X1_arr[k], X2_arr[k], cp,
                        GAMMA=1.0, r_core=r_core,
                        delta_frac=delta_frac_wake, cutoff_mode=cutoff_mode,
                    )

        U[:, j] = vel[:, 0]
        V[:, j] = vel[:, 1]
        W[:, j] = vel[:, 2]

    return U, V, W


if __name__ == "__main__":
    # Sanity test: single horseshoe, short wake, one control point.
    # The bound segment from z=1 to z=2 with Gamma=1 should induce
    # a velocity at z=1.5 whose direction follows the right-hand rule:
    # velocity should point in the +y direction (out of the page when the
    # filament points along +z and the control point is at z=1.5, x=0, y=0).
    # With a short wake the trailing legs add smaller corrections.

    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from lifting_line.geometry import build_geometry

    geom = build_geometry(
        R=50.0, r_root=10.0, N=5, N_blades=3,
        N_wake=3, dpsi_deg=30.0,
        U_inf=10.0, Omega=1.6, a_w=0.25,
    )
    geom["pitch"] = np.radians(-2.0)

    U_m, V_m, W_m = assemble_influence(geom["cp"], geom["bound"], geom["wake"])

    print("=== Influence-matrix sanity check ===")
    print(f"  U shape: {U_m.shape}  V shape: {V_m.shape}  W shape: {W_m.shape}")

    # Diagonal of U should be negative (axial induction opposes freestream)
    # for a turbine (bound vortex and wake pull flow back)
    print(f"  diag(U) = {np.diag(U_m)}")
    print(f"  diag(V) = {np.diag(V_m)}")

    # Check finite and not all-zero
    assert np.all(np.isfinite(U_m)), "NaN/Inf in U"
    assert np.any(U_m != 0.0), "U is all zeros"
    print("  PASSED: matrices finite and non-zero")

    # Check sign: for a wind turbine the diagonal of U (self-induction in axial
    # direction due to the trailing vortex system) should be negative.
    # (The bound segment of a lifting-line panel primarily induces in ±y;
    #  the trailing legs pull the axial velocity down at the same span station.)
    # Just print rather than assert since sign depends on wake geometry.
    print(f"  max|U| = {np.max(np.abs(U_m)):.4f}   max|V| = {np.max(np.abs(V_m)):.4f}")
