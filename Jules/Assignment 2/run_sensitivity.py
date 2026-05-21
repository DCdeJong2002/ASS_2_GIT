"""Sensitivity studies for the frozen-wake lifting line solver.

Runs four sweeps at the baseline TSR (lambda = 8) — one parameter at a
time, others held at the defaults from Table 1 of code_structure.tex —
and saves four multi-panel figures to the figures/ directory.

Sweeps (assignment Table 2):
    1. a_w       in [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    2. N         in {5, 10, 20, 40, 80}, BOTH uniform and cosine
    3. N_wake    in {1, 3, 5, 10, 20}
    4. dpsi_deg  in {5, 10, 20}
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
from lifting_line.postprocess import (
    post_process,
    sensitivity_sweep,
    make_sensitivity_figures,
)

# ---- Baseline operating point ----------------------------------------
LAM      = 8.0
R        = 50.0
R_ROOT   = 0.2 * R
N_BLADES = 3
U_INF    = 6.0
PITCH_DEG = -2.0

POLAR_PATH    = "Assignment 2/polar DU95W180 (3).xlsx"
FIGDIR        = "Assignment 2/figures"
_A_W_FALLBACK = 0.25


def _bem_a_w(lam: float) -> float:
    """Span-weighted average axial induction from the BEM CSV for this TSR."""
    csv_path = os.path.join(FIGDIR, f"bem_lam{int(lam)}.csv")
    if not os.path.exists(csv_path):
        warnings.warn(f"BEM CSV not found ({csv_path}); falling back to a_w={_A_W_FALLBACK}")
        return _A_W_FALLBACK
    nodes = R_ROOT + (R - R_ROOT) * (1 - np.cos(np.linspace(0, np.pi, 100))) / 2
    dr = np.diff(nodes)
    a = pd.read_csv(csv_path)["a"].to_numpy()
    return float(np.dot(a, dr) / np.sum(dr))


# ---- Default discretisation (the "baseline" of every sweep) ----------
DEFAULTS = dict(
    R          = R,
    r_root     = R_ROOT,
    N          = 15,
    N_blades   = N_BLADES,
    N_wake     = 10,
    dpsi_deg   = 10.0,
    U_inf      = U_INF,
    Omega      = LAM * U_INF / R,
    a_w        = _bem_a_w(LAM),
    distribution = "cosine",
    pitch_deg  = PITCH_DEG,
    lam        = LAM,
)


def run_baseline(polar):
    """Run the baseline configuration once for the sensitivity-figure overlay."""
    geom = build_geometry(
        R=DEFAULTS["R"], r_root=DEFAULTS["r_root"], N=DEFAULTS["N"],
        N_blades=DEFAULTS["N_blades"], N_wake=DEFAULTS["N_wake"],
        dpsi_deg=DEFAULTS["dpsi_deg"], U_inf=DEFAULTS["U_inf"],
        Omega=DEFAULTS["Omega"], a_w=DEFAULTS["a_w"],
        distribution=DEFAULTS["distribution"],
    )
    geom["pitch"] = np.radians(DEFAULTS["pitch_deg"])
    U_m, V_m, W_m = assemble_influence(geom["cp"], geom["bound"], geom["wake"], geom["r_core"])
    sol = solve_circulation(U_m, V_m, W_m, geom, polar,
                            U_inf=DEFAULTS["U_inf"], Omega=DEFAULTS["Omega"])
    return post_process(sol, geom, DEFAULTS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-N",       action="store_true", help="skip the N sweep (slow)")
    parser.add_argument("--skip-Nwake",   action="store_true")
    parser.add_argument("--skip-dpsi",    action="store_true")
    parser.add_argument("--skip-aw",      action="store_true")
    args = parser.parse_args()

    print(f"Loading polar from '{POLAR_PATH}' ...")
    polar = load_polar(POLAR_PATH)

    print("Running baseline (lam=8, defaults) for overlay ...")
    baseline = run_baseline(polar)
    print(f"  baseline CT={baseline['CT']:.4f}  CP={baseline['CP']:.4f}  iters={baseline['iters']}")

    sweep_results = {}

    # 1. Wake convection speed -----------------------------------------
    if not args.skip_aw:
        print("\n[1/4] a_w sweep: [0, 0.1, 0.2, 0.3, 0.4, 0.5]")
        t0 = time.perf_counter()
        sweep_results["a_w"] = sensitivity_sweep(
            "a_w", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], DEFAULTS, polar,
        )
        print(f"  done in {time.perf_counter()-t0:.1f}s")
        for r in sweep_results["a_w"]:
            print(f"    a_w={r['param_value']:.2f}  CT={r['CT']:.4f}  CP={r['CP']:.4f}  iters={r['iters']}")

    # 2. Spanwise resolution N -----------------------------------------
    if not args.skip_N:
        print("\n[2/4] N sweep (uniform AND cosine): {3, 5, 10, 20, 40}")
        t0 = time.perf_counter()
        for dist in ("uniform", "cosine"):
            cfg = {**DEFAULTS, "distribution": dist}
            res = sensitivity_sweep("N", [3, 5, 10, 20, 40], cfg, polar)
            sweep_results[f"N_{dist}"] = res
            for r in res:
                print(f"    N={r['param_value']:3d} ({dist}): CT={r['CT']:.4f}  CP={r['CP']:.4f}  iters={r['iters']}")
        print(f"  done in {time.perf_counter()-t0:.1f}s")

    # 3. Wake length ---------------------------------------------------
    if not args.skip_Nwake:
        print("\n[3/4] N_wake sweep: {1, 3, 5, 10, 20}")
        t0 = time.perf_counter()
        sweep_results["N_wake"] = sensitivity_sweep(
            "N_wake", [1, 3, 5, 10, 20], DEFAULTS, polar,
        )
        print(f"  done in {time.perf_counter()-t0:.1f}s")
        for r in sweep_results["N_wake"]:
            print(f"    N_wake={r['param_value']:2d}  CT={r['CT']:.4f}  CP={r['CP']:.4f}  iters={r['iters']}")

    # 4. Azimuthal step ------------------------------------------------
    if not args.skip_dpsi:
        print("\n[4/4] dpsi_deg sweep: {5, 10, 20}")
        t0 = time.perf_counter()
        sweep_results["dpsi_deg"] = sensitivity_sweep(
            "dpsi_deg", [5.0, 10.0, 20.0], DEFAULTS, polar,
        )
        print(f"  done in {time.perf_counter()-t0:.1f}s")
        for r in sweep_results["dpsi_deg"]:
            print(f"    dpsi={r['param_value']:.0f} deg  CT={r['CT']:.4f}  CP={r['CP']:.4f}  iters={r['iters']}")

    print("\nWriting sensitivity figures ...")
    make_sensitivity_figures(sweep_results, baseline, FIGDIR)
    print("Done.")


if __name__ == "__main__":
    main()
