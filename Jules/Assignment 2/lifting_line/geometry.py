"""Module 1 — Blade and frozen-wake geometry.

Coordinate system
-----------------
x : axial (freestream direction, positive downstream)
y : in-plane, tangential direction for the reference blade at ψ=0
z : in-plane, radial direction for the reference blade at ψ=0

The reference blade (blade 0) points along +z at ψ=0.
Blade b has azimuth ψ_b = 2π·b/N_blades, laid out via eqs (wake-y/z).
"""

import numpy as np


def build_geometry(
    R: float,
    r_root: float,
    N: int,
    N_blades: int,
    N_wake: int,
    dpsi_deg: float,
    U_inf: float,
    Omega: float,
    a_w: float,
    distribution: str = "cosine",
    r_core: float | None = None,
) -> dict:
    """Return all geometry arrays needed by the influence-matrix assembly."""
    if r_core is None:
        r_core = 1e-3 * R

    dpsi = np.radians(dpsi_deg)

    # --- spanwise node positions (N+1 nodes) ---
    i = np.arange(N + 1, dtype=float)
    if distribution == "cosine":
        r_node = r_root + 0.5 * (R - r_root) * (1.0 - np.cos(np.pi * i / N))
    elif distribution == "half_cosine":
        # Dense at tip, coarse at root
        r_node = r_root + (R - r_root) * np.sin(0.5 * np.pi * i / N)
    else:
        r_node = r_root + (R - r_root) * i / N

    # panel midpoints (N values)
    r_mid = 0.5 * (r_node[:-1] + r_node[1:])

    # blade chord and twist at panel midpoints
    chord = 3.0 * (1.0 - r_mid / R) + 1.0          # [m]
    twist = np.radians(14.0 * (1.0 - r_mid / R))   # [rad]

    # --- 3-D bound-vortex nodes on reference blade (ψ=0, along +z) ---
    bound = np.zeros((N + 1, 3))
    bound[:, 2] = r_node  # (0, 0, r_j)

    # --- control points on reference blade ---
    cp = np.zeros((N, 3))
    cp[:, 2] = r_mid  # (0, 0, r_mid_i)

    # --- helical wake nodes ---
    U_wake = U_inf * (1.0 - a_w)
    N_az = int(round(N_wake * 2.0 * np.pi / dpsi))

    k = np.arange(N_az + 1, dtype=float)  # shape (N_az+1,)
    t_k = k * dpsi / Omega                 

    # Geometric chord and twist at the nodes for accurate wake offset
    chord_at_node = 3.0 * (1.0 - r_node / R) + 1.0       # (N+1,)
    twist_at_node = np.radians(14.0 * (1.0 - r_node / R)) # (N+1,)

    # Chord-line aligned offsets (from 1/4 chord to 5/4 chord)
    # The blade rotates towards +y, so the trailing edge trails in the -y direction.
    # Pitching the nose into the wind (+x) moves the trailing edge downstream (+x).
    dx_node = chord_at_node * np.sin(twist_at_node)
    dy_node = -chord_at_node * np.cos(twist_at_node)

    wake = np.empty((N_blades, N + 1, N_az + 1, 3))

    for b in range(N_blades):
        psi0 = 2.0 * np.pi * b / N_blades
        psi_k = psi0 - k * dpsi  

        # x: Offset physically aligned with geometric chordline
        wake[b, :, :, 0] = dx_node[:, np.newaxis] + U_wake * t_k[np.newaxis, :]

        # y: Incorporating tangential chord offset rotated to azimuth psi_k
        wake[b, :, :, 1] = (dy_node[:, np.newaxis] * np.cos(psi_k[np.newaxis, :]) +
                            r_node[:, np.newaxis] * np.sin(psi_k[np.newaxis, :]))

        # z: Incorporating tangential chord offset rotated to azimuth psi_k
        wake[b, :, :, 2] = (-dy_node[:, np.newaxis] * np.sin(psi_k[np.newaxis, :]) +
                            r_node[:, np.newaxis] * np.cos(psi_k[np.newaxis, :]))

    return {
        "bound": bound,
        "cp": cp,
        "wake": wake,
        "r_node": r_node,
        "r_mid": r_mid,
        "chord": chord,
        "twist": twist,
        "pitch": 0.0,
        "N": N,
        "N_az": N_az,
        "R": R,
        "r_core": r_core,
    }


if __name__ == "__main__":
    # Sanity test: default discretisation, lambda=8
    R = 50.0
    U_inf = 10.0
    lam = 8
    Omega = lam * U_inf / R

    geom = build_geometry(
        R=R, r_root=0.2 * R, N=30, N_blades=3,
        N_wake=10, dpsi_deg=10.0,
        U_inf=U_inf, Omega=Omega, a_w=0.25,
    )

    print("=== Geometry sanity check ===")
    print(f"  bound shape : {geom['bound'].shape}   (expected (31, 3))")
    print(f"  cp shape    : {geom['cp'].shape}   (expected (30, 3))")
    print(f"  wake shape  : {geom['wake'].shape}   (expected (3, 31, 361, 3))")
    print(f"  N_az        : {geom['N_az']}   (expected 360)")
    print(f"  r_core      : {geom['r_core']:.4f} m")

    # Check a specific wake node: blade 0, radial index 5, azimuthal index 36
    b, j, k_idx = 0, 5, 36
    node = geom["wake"][b, j, k_idx]
    
    r5 = geom["r_node"][j]
    chord5 = 3.0 * (1.0 - r5 / R) + 1.0
    twist5 = np.radians(14.0 * (1.0 - r5 / R))
    
    dx5 = chord5 * np.sin(twist5)
    dy5 = -chord5 * np.cos(twist5)
    
    dpsi_rad = np.radians(10.0)
    U_wake = U_inf * (1.0 - 0.25)
    t36 = 36 * dpsi_rad / Omega
    psi_k_val = 0.0 - 36 * dpsi_rad
    
    x_exp = dx5 + U_wake * t36
    y_exp = dy5 * np.cos(psi_k_val) + r5 * np.sin(psi_k_val)
    z_exp = -dy5 * np.sin(psi_k_val) + r5 * np.cos(psi_k_val)
    
    print(f"\n  wake[0,5,36] = {node}")
    print(f"  expected x   = {x_exp:.4f}  got {node[0]:.4f}")
    print(f"  expected y   = {y_exp:.4f}  got {node[1]:.4f}")
    print(f"  expected z   = {z_exp:.4f}  got {node[2]:.4f}")
    assert abs(node[0] - x_exp) < 1e-8, "x mismatch"
    assert abs(node[1] - y_exp) < 1e-8, "y mismatch"
    assert abs(node[2] - z_exp) < 1e-8, "z mismatch"
    print("  PASSED")