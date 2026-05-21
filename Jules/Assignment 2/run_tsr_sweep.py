"""TSR sweep: run the lifting-line solver for λ = 6.0 … 10.0 (step 0.25).

Produces two figures saved to Assignment 2/figures/:
  tsr_sweep_CP.pdf   — CP vs TSR
  tsr_sweep_CT_CQ.pdf — CT and CQ vs TSR (shared axis)

Usage
-----
    python "Assignment 2/run_tsr_sweep.py"
"""

import sys
import os
import time
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow importing from the same directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from lifting_line.geometry    import build_geometry
from lifting_line.influence   import assemble_influence
from lifting_line.polar       import load_polar
from lifting_line.solver      import solve_circulation
from lifting_line.postprocess import post_process

# ---- Turbine parameters (identical to run_baseline.py) ----------------
R         = 50.0
R_ROOT    = 0.2 * R
N_BLADES  = 3
U_INF     = 10.0
PITCH_DEG = -2.0
RHO       = 1.225

# ---- Discretisation defaults ------------------------------------------
N_DEFAULT      = 15
N_WAKE_DEFAULT = 10
DPSI_DEFAULT   = 10.0
A_W_DEFAULT    = 0.25
DIST_DEFAULT   = "cosine"

POLAR_PATH = "Assignment 2/polar DU95W180 (3).xlsx"
FIGDIR     = "Assignment 2/figures"

_STYLE = {
    "font.family":       "serif",
    "axes.grid":         False,
    "figure.dpi":        150,
    "lines.linewidth":   1.5,
    "axes.spines.top":   False,
    "axes.spines.right": False,
}


def _bem_a_w(lam: float) -> float:
    import pandas as pd
    csv_path = os.path.join(FIGDIR, f"bem_lam{int(lam)}.csv")
    if not os.path.exists(csv_path):
        return A_W_DEFAULT
    nodes = R_ROOT + (R - R_ROOT) * (1 - np.cos(np.linspace(0, np.pi, 100))) / 2
    dr = np.diff(nodes)
    a = pd.read_csv(csv_path)["a"].to_numpy()
    return float(np.dot(a, dr) / np.sum(dr))


def run_one(lam: float, polar: dict) -> dict:
    Omega = lam * U_INF / R
    a_w   = _bem_a_w(lam)

    geom = build_geometry(
        R=R, r_root=R_ROOT, N=N_DEFAULT, N_blades=N_BLADES,
        N_wake=N_WAKE_DEFAULT, dpsi_deg=DPSI_DEFAULT,
        U_inf=U_INF, Omega=Omega, a_w=a_w,
        distribution=DIST_DEFAULT,
    )
    geom["pitch"] = np.radians(PITCH_DEG)

    U_m, V_m, W_m = assemble_influence(geom["cp"], geom["bound"],
                                       geom["wake"], geom["r_core"])

    sol = solve_circulation(U_m, V_m, W_m, geom, polar,
                            U_inf=U_INF, Omega=Omega)

    cfg = dict(U_inf=U_INF, Omega=Omega, R=R, N_blades=N_BLADES, lam=lam,
               pitch_deg=PITCH_DEG, N=N_DEFAULT, N_wake=N_WAKE_DEFAULT,
               dpsi_deg=DPSI_DEFAULT, a_w=a_w)
    res = post_process(sol, geom, cfg)

    q_ref = 0.5 * RHO * U_INF**2 * np.pi * R**2
    res["CQ"] = res["Q"] / (q_ref * R)   # CQ = Q / (q_ref * R)

    status = "OK" if res["converged"] else "NO CONV"
    print(f"  λ={lam:5.2f}  CT={res['CT']:.4f}  CP={res['CP']:.4f}"
          f"  CQ={res['CQ']:.4f}  [{status}]")
    return res


def _load_bem_sweep() -> dict | None:
    import pandas as pd
    path = os.path.join(FIGDIR, "bem_tsr_sweep.csv")
    if not os.path.exists(path):
        import warnings
        warnings.warn(f"BEM sweep CSV not found ({path}); run generate_bem_csv.py first.")
        return None
    df = pd.read_csv(path)
    return {"lam": df["lam"].to_numpy(),
            "CT":  df["CT"].to_numpy(),
            "CP":  df["CP"].to_numpy(),
            "CQ":  df["CQ"].to_numpy()}


def make_figures(results: list[dict]) -> None:
    os.makedirs(FIGDIR, exist_ok=True)
    lams = np.array([r["lam"] for r in results])
    CTs  = np.array([r["CT"]  for r in results])
    CPs  = np.array([r["CP"]  for r in results])
    CQs  = np.array([r["CQ"]  for r in results])

    bem = _load_bem_sweep()

    with plt.rc_context(_STYLE):

        # ---- CP vs TSR ----
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(lams, CPs, "-", color="steelblue", label="LL")
        if bem is not None:
            ax.plot(bem["lam"], bem["CP"], ":", color="steelblue", label="BEM")
        ax.set_xlabel(r"$\lambda$ [-]")
        ax.set_ylabel(r"$C_P$ [-]")
        ax.set_title("Power coefficient vs tip speed ratio")
        ax.legend(frameon=False)
        fig.tight_layout()
        path = os.path.join(FIGDIR, "tsr_sweep_CP.pdf")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path}")

        # ---- CT and CQ vs TSR (twin y-axes) ----
        fig, ax_cq = plt.subplots(figsize=(6, 4))
        ax_ct = ax_cq.twinx()

        line_cq, = ax_cq.plot(lams, CQs, "-",  color="firebrick",  label=r"$C_Q$ LL")
        line_ct, = ax_ct.plot(lams, CTs, "-",  color="steelblue",  label=r"$C_T$ LL")
        extra_handles = [line_cq, line_ct]

        if bem is not None:
            line_cq_bem, = ax_cq.plot(bem["lam"], bem["CQ"], ":", color="firebrick",
                                       label=r"$C_Q$ BEM")
            line_ct_bem, = ax_ct.plot(bem["lam"], bem["CT"], ":", color="steelblue",
                                       label=r"$C_T$ BEM")
            extra_handles += [line_cq_bem, line_ct_bem]

        ax_cq.set_xlabel(r"$\lambda$ [-]")
        ax_cq.set_ylabel(r"$C_Q$ [-]", color="firebrick")
        ax_cq.tick_params(axis="y", colors="firebrick")
        ax_cq.spines["left"].set_color("firebrick")

        ax_ct.set_ylabel(r"$C_T$ [-]", color="steelblue")
        ax_ct.tick_params(axis="y", colors="steelblue")
        ax_ct.spines["right"].set_visible(True)
        ax_ct.spines["right"].set_color("steelblue")
        ax_ct.spines["top"].set_visible(False)

        ax_cq.set_title("Thrust and torque coefficients vs tip speed ratio")
        ax_cq.legend(handles=extra_handles, frameon=False, fontsize=8,
                     loc="center right")
        fig.tight_layout()
        path = os.path.join(FIGDIR, "tsr_sweep_CT_CQ.pdf")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path}")


def main():
    lams = np.arange(6.0, 10.0 + 1e-9, 0.25)

    print(f"Loading polar from '{POLAR_PATH}' ...")
    polar = load_polar(POLAR_PATH)

    print(f"\nRunning λ = {lams[0]:.2f} … {lams[-1]:.2f}  ({len(lams)} points)")
    print("-" * 60)

    t0 = time.perf_counter()
    results = [run_one(lam, polar) for lam in lams]
    print(f"\nTotal wall time: {time.perf_counter() - t0:.1f} s")

    print("-" * 60)
    make_figures(results)


if __name__ == "__main__":
    main()
