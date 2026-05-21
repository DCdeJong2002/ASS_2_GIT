"""Run the frozen-wake lifting line solver for λ ∈ {6, 8, 10} and save figures.

Usage
-----
    python run_baseline.py            # all three TSR values
    python run_baseline.py --lam 8   # single TSR for quick checks
"""

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd

from lifting_line.geometry    import build_geometry
from lifting_line.influence   import assemble_influence
from lifting_line.polar       import load_polar
from lifting_line.solver      import solve_circulation
from lifting_line.postprocess import post_process, make_figures

# ---- Turbine parameters -----------------------------------------------
R        = 50.0    # [m]
R_ROOT   = 0.2 * R
N_BLADES = 3
U_INF    = 10.0    # [m/s]
PITCH_DEG = -2.0   # [deg] collective pitch

# ---- Default discretisation (Table 1, code_structure.tex) -------------
N_DEFAULT      = 15
N_WAKE_DEFAULT = 10
DPSI_DEFAULT   = 10.0   # [deg]
A_W_DEFAULT    = 0.25
DIST_DEFAULT   = "cosine"

POLAR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "polar DU95W180 (3).xlsx")
FIGDIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def _bem_a_w(lam: float) -> float:
    """Span-weighted average axial induction from the BEM CSV for this TSR."""
    csv_path = os.path.join(FIGDIR, f"bem_lam{int(lam)}.csv")
    if not os.path.exists(csv_path):
        warnings.warn(f"BEM CSV not found ({csv_path}); falling back to a_w={A_W_DEFAULT}")
        return A_W_DEFAULT
    nodes = R_ROOT + (R - R_ROOT) * (1 - np.cos(np.linspace(0, np.pi, 100))) / 2
    dr = np.diff(nodes)
    a = pd.read_csv(csv_path)["a"].to_numpy()
    return float(np.dot(a, dr) / np.sum(dr))


def run_one(lam: float, polar: dict, verbose: bool = True) -> dict:
    Omega = lam * U_INF / R
    a_w = _bem_a_w(lam)

    t0 = time.perf_counter()
    geom = build_geometry(
        R=R, r_root=R_ROOT, N=N_DEFAULT, N_blades=N_BLADES,
        N_wake=N_WAKE_DEFAULT, dpsi_deg=DPSI_DEFAULT,
        U_inf=U_INF, Omega=Omega, a_w=a_w,
        distribution=DIST_DEFAULT,
    )
    geom["pitch"] = np.radians(PITCH_DEG)
    t_geom = time.perf_counter() - t0

    t1 = time.perf_counter()
    U_m, V_m, W_m = assemble_influence(geom["cp"], geom["bound"],
                                       geom["wake"], geom["r_core"])
    t_inf = time.perf_counter() - t1

    t2 = time.perf_counter()
    sol = solve_circulation(U_m, V_m, W_m, geom, polar,
                            U_inf=U_INF, Omega=Omega)
    t_sol = time.perf_counter() - t2

    cfg = dict(U_inf=U_INF, Omega=Omega, R=R, N_blades=N_BLADES, lam=lam,
               pitch_deg=PITCH_DEG, N=N_DEFAULT, N_wake=N_WAKE_DEFAULT,
               dpsi_deg=DPSI_DEFAULT, a_w=a_w)
    res = post_process(sol, geom, cfg)

    if verbose:
        status = "CONVERGED" if res["converged"] else "NOT CONVERGED"
        print(f"  lam={lam:2.0f}  a_w={a_w:.4f}  CT={res['CT']:.4f}  CP={res['CP']:.4f}"
              f"  iters={res['iters']:4d}  [{status}]"
              f"  (geom {t_geom:.1f}s  inf {t_inf:.1f}s  sol {t_sol:.2f}s)")

    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lam", type=float, nargs="+", default=[6, 8, 10])
    args = parser.parse_args()

    print(f"Loading polar from '{POLAR_PATH}' ...")
    polar = load_polar(POLAR_PATH)

    print(f"\nRunning lam = {args.lam}  (N={N_DEFAULT}, N_wake={N_WAKE_DEFAULT},"
          f" dpsi={DPSI_DEFAULT} deg, a_w=BEM-derived, dist={DIST_DEFAULT})")
    print("-" * 72)

    results = []
    for lam in args.lam:
        results.append(run_one(lam, polar))

    print("-" * 72)
    make_figures(results, FIGDIR)


if __name__ == "__main__":
    main()
