"""Module 6 — Post-processing, sensitivity sweep, and plotting.

After convergence, computes all integral and distributed output quantities
(eq:T-Q-integration, eq:CT, eq:CP, eq:induction-factors, eq:gamma-hat,
eq:F-hat) and produces the figures required by the assignment.
"""

import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

from .geometry import build_geometry
from .influence import assemble_influence
from .polar import load_polar, airfoil_coefficients
from .solver import solve_circulation

# -------------------------------------------------------------------
# Plot style — serif fonts, no gridlines, suitable for LaTeX report
# -------------------------------------------------------------------
_STYLE = {
    "font.family":      "serif",
    "axes.grid":        False,
    "figure.dpi":       150,
    "lines.linewidth":  1.5,
    "axes.spines.top":  False,
    "axes.spines.right":False,
}
RHO = 1.225  # kg/m³


# -------------------------------------------------------------------
# Core post-processing
# -------------------------------------------------------------------

def post_process(sol: dict, geom: dict, cfg: dict) -> dict:
    """Compute all output quantities from a converged solution.

    Parameters
    ----------
    sol  : dict returned by solve_circulation
    geom : dict returned by build_geometry (with pitch set)
    cfg  : dict with keys U_inf, Omega, R, N_blades, lam (TSR)

    Returns
    -------
    dict with all distributed and integral quantities (SI + non-dimensional).
    """
    U_inf    = cfg["U_inf"]
    Omega    = cfg["Omega"]
    R        = cfg["R"]
    N_blades = cfg["N_blades"]

    r_mid  = geom["r_mid"]
    chord  = geom["chord"]

    Gamma = sol["Gamma"]
    phi   = sol["phi"]
    Cl    = sol["Cl"]
    Cd    = sol["Cd"]
    V_p   = sol["V_p"]
    u_ind = sol["u_ind"]
    v_ind = sol["v_ind"]

    # eq:induction-factors
    a       = -u_ind / U_inf
    a_prime = -v_ind / (Omega * r_mid)

    # eq:lift, eq:drag — sectional (per unit span)
    L = 0.5 * RHO * V_p**2 * chord * Cl
    D = 0.5 * RHO * V_p**2 * chord * Cd

    # eq:F-axial, eq:F-azim
    F_ax   = L * np.cos(phi) + D * np.sin(phi)
    F_azim = L * np.sin(phi) - D * np.cos(phi)

    # eq:T-Q-integration
    T = N_blades * trapezoid(F_ax,          r_mid)
    Q = N_blades * trapezoid(r_mid * F_azim, r_mid)
    P = Omega * Q

    # eq:CT, eq:CP
    q_ref = 0.5 * RHO * U_inf**2 * np.pi * R**2
    CT = T / q_ref
    CP = P / (q_ref * U_inf)

    # eq:gamma-hat
    Gamma_hat = Gamma * N_blades * Omega / (np.pi * U_inf**2)

    # eq:F-hat
    F_hat_ax   = F_ax   / (0.5 * RHO * U_inf**2 * R)
    F_hat_azim = F_azim / (0.5 * RHO * U_inf**2 * R)

    return {
        "r_R":        r_mid / R,
        "Gamma":      Gamma,
        "Gamma_hat":  Gamma_hat,
        "phi_deg":    np.degrees(phi),
        "alpha_deg":  np.degrees(sol["alpha"]),
        "a":          a,
        "a_prime":    a_prime,
        "F_ax":       F_ax,
        "F_azim":     F_azim,
        "F_hat_ax":   F_hat_ax,
        "F_hat_azim": F_hat_azim,
        "CT":         CT,
        "CP":         CP,
        "T":          T,
        "Q":          Q,
        "P":          P,
        "iters":      sol["iters"],
        "converged":  sol["converged"],
        "lam":        cfg.get("lam", Omega * R / U_inf),
    }


# -------------------------------------------------------------------
# Baseline figures — six plots per TSR
# -------------------------------------------------------------------

def make_figures(results: list[dict], figdir: str = "figures") -> None:
    """Save all six deliverable figure sets for the baseline TSR sweep.

    Parameters
    ----------
    results  : list of post_process dicts, one per TSR value
    figdir   : output directory (created if absent)
    """
    os.makedirs(figdir, exist_ok=True)
    lams   = [r["lam"] for r in results]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(results)))

    with plt.rc_context(_STYLE):

        # ---- Fig 1: phi and alpha vs r/R ----
        fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=False)
        for res, c, lam in zip(results, colors, lams):
            axes[0].plot(res["r_R"], res["phi_deg"],   color=c, label=f"λ={lam:.0f}")
            axes[1].plot(res["r_R"], res["alpha_deg"], color=c, label=f"λ={lam:.0f}")
        axes[0].set_xlabel(r"$r/R$"); axes[0].set_ylabel(r"$\Phi$ [deg]")
        axes[0].set_title("Inflow angle")
        axes[1].set_xlabel(r"$r/R$"); axes[1].set_ylabel(r"$\alpha$ [deg]")
        axes[1].set_title("Angle of attack")
        axes[1].legend(loc="upper right", frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, "baseline_phi_alpha.pdf"), bbox_inches="tight")
        plt.close(fig)

        # ---- Fig 2: a and a' vs r/R ----
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        for res, c, lam in zip(results, colors, lams):
            axes[0].plot(res["r_R"], res["a"],       color=c, label=f"λ={lam:.0f}")
            axes[1].plot(res["r_R"], res["a_prime"], color=c, label=f"λ={lam:.0f}")
        axes[0].set_xlabel(r"$r/R$"); axes[0].set_ylabel(r"$a$")
        axes[0].set_title("Axial induction factor")
        axes[1].set_xlabel(r"$r/R$"); axes[1].set_ylabel(r"$a'$")
        axes[1].set_title("Tangential induction factor")
        axes[1].legend(loc="upper right", frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, "baseline_induction.pdf"), bbox_inches="tight")
        plt.close(fig)

        # ---- Fig 3: non-dim loading vs r/R ----
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        for res, c, lam in zip(results, colors, lams):
            axes[0].plot(res["r_R"], res["F_hat_ax"],   color=c, label=f"λ={lam:.0f}")
            axes[1].plot(res["r_R"], res["F_hat_azim"], color=c, label=f"λ={lam:.0f}")
        axes[0].set_xlabel(r"$r/R$"); axes[0].set_ylabel(r"$\hat{F}_\mathrm{ax}$")
        axes[0].set_title("Non-dim axial loading")
        axes[1].set_xlabel(r"$r/R$"); axes[1].set_ylabel(r"$\hat{F}_\mathrm{azim}$")
        axes[1].set_title("Non-dim azimuthal loading")
        axes[1].legend(loc="lower center", frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, "baseline_loading.pdf"), bbox_inches="tight")
        plt.close(fig)

        # ---- Fig 4: Gamma_hat vs r/R ----
        fig, ax = plt.subplots(figsize=(6, 4))
        for res, c, lam in zip(results, colors, lams):
            ax.plot(res["r_R"], res["Gamma_hat"], color=c, label=f"λ={lam:.0f}")
        ax.set_xlabel(r"$r/R$")
        ax.set_ylabel(r"$\hat{\Gamma}$")
        ax.set_title(r"Non-dimensional bound circulation")
        ax.legend(loc="lower center", frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, "baseline_gamma_hat.pdf"), bbox_inches="tight")
        plt.close(fig)

        # ---- Fig 5: CT and CP table / bar ----
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.axis("off")
        rows = [[f"{r['lam']:.0f}", f"{r['CT']:.4f}", f"{r['CP']:.4f}",
                 f"{r['iters']}", "✓" if r["converged"] else "✗"]
                for r in results]
        tbl = ax.table(cellText=rows,
                       colLabels=["λ", "CT", "CP", "iters", "conv."],
                       loc="center", cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(10)
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, "baseline_CT_CP_table.pdf"), bbox_inches="tight")
        plt.close(fig)
        _write_baseline_latex_table(results, figdir)

        # ---- Fig 6: BEM comparison stub ----
        _plot_bem_comparison(results, figdir, colors)

    print(f"Figures written to '{figdir}/'")


def _plot_bem_comparison(results: list[dict], figdir: str, colors) -> None:
    """Compare lifting-line results against BEM data, mirroring the baseline
    plot layout: one figure per quantity, all TSRs overlaid. LL curves are
    solid, BEM curves are dashed in matching colors.

    Expected CSV (one file per TSR, named bem_lam{lam:.0f}.csv) with columns
        r_R, a, a_prime, alpha_deg, phi_deg, Gamma_hat, F_hat_ax, F_hat_azim
    TSRs whose CSV is missing are silently skipped from the BEM overlay.
    """
    import pandas as pd

    lams = [r["lam"] for r in results]
    bem_data = []
    for lam in lams:
        bem_path = os.path.join(figdir, f"bem_lam{lam:.0f}.csv")
        if os.path.exists(bem_path):
            bem_data.append(pd.read_csv(bem_path))
        else:
            warnings.warn(
                f"BEM comparison file not found: {bem_path}. "
                "Run generate_bem_csv.py to produce it.",
                stacklevel=2,
            )
            bem_data.append(None)

    if all(b is None for b in bem_data):
        return

    def _overlay(ax, key, scale=1.0):
        for bem, c in zip(bem_data, colors):
            if bem is not None and key in bem.columns:
                ax.plot(bem["r_R"], bem[key] * scale, ls="--", color=c)

    # ---- Fig 1: phi and alpha vs r/R ----
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for res, c, lam in zip(results, colors, lams):
        axes[0].plot(res["r_R"], res["phi_deg"],   color=c, label=f"λ={lam:.0f}")
        axes[1].plot(res["r_R"], res["alpha_deg"], color=c, label=f"λ={lam:.0f}")
    _overlay(axes[0], "phi_deg")
    _overlay(axes[1], "alpha_deg")
    axes[0].set_xlabel(r"$r/R$"); axes[0].set_ylabel(r"$\Phi$ [deg]")
    axes[0].set_title("Inflow angle")
    axes[1].set_xlabel(r"$r/R$"); axes[1].set_ylabel(r"$\alpha$ [deg]")
    axes[1].set_title("Angle of attack")
    _ll_bem_legend(axes[1], colors, lams)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "bem_comparison_phi_alpha.pdf"), bbox_inches="tight")
    plt.close(fig)

    # ---- Fig 2: a and a' vs r/R ----
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for res, c, lam in zip(results, colors, lams):
        axes[0].plot(res["r_R"], res["a"],       color=c, label=f"λ={lam:.0f}")
        axes[1].plot(res["r_R"], res["a_prime"], color=c, label=f"λ={lam:.0f}")
    _overlay(axes[0], "a")
    _overlay(axes[1], "a_prime")
    axes[0].set_xlabel(r"$r/R$"); axes[0].set_ylabel(r"$a$")
    axes[0].set_title("Axial induction factor")
    axes[1].set_xlabel(r"$r/R$"); axes[1].set_ylabel(r"$a'$")
    axes[1].set_title("Tangential induction factor")
    axes[1].set_ylim(bottom=0.0, top=0.15)
    _ll_bem_legend(axes[1], colors, lams)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "bem_comparison_induction.pdf"), bbox_inches="tight")
    plt.close(fig)

    # ---- Fig 3: non-dim loading vs r/R ----
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for res, c, lam in zip(results, colors, lams):
        axes[0].plot(res["r_R"], res["F_hat_ax"],   color=c, label=f"λ={lam:.0f}")
        axes[1].plot(res["r_R"], res["F_hat_azim"], color=c, label=f"λ={lam:.0f}")
    _overlay(axes[0], "F_hat_ax")
    _overlay(axes[1], "F_hat_azim")
    axes[0].set_xlabel(r"$r/R$"); axes[0].set_ylabel(r"$\hat{F}_\mathrm{ax}$")
    axes[0].set_title("Non-dim axial loading")
    axes[1].set_xlabel(r"$r/R$"); axes[1].set_ylabel(r"$\hat{F}_\mathrm{azim}$")
    axes[1].set_title("Non-dim azimuthal loading")
    _ll_bem_legend(axes[1], colors, lams, loc="lower center")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "bem_comparison_loading.pdf"), bbox_inches="tight")
    plt.close(fig)

    # ---- Fig 4: Gamma_hat vs r/R ----
    fig, ax = plt.subplots(figsize=(6, 4))
    for res, c, lam in zip(results, colors, lams):
        ax.plot(res["r_R"], res["Gamma_hat"], color=c, label=f"λ={lam:.0f}")
    _overlay(ax, "Gamma_hat")
    ax.set_xlabel(r"$r/R$")
    ax.set_ylabel(r"$\hat{\Gamma}$")
    ax.set_title(r"Non-dimensional bound circulation")
    _ll_bem_legend(ax, colors, lams, loc="lower center")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "bem_comparison_gamma_hat.pdf"), bbox_inches="tight")
    plt.close(fig)

    # ---- Fig 5: CT / CP comparison table ----
    summary_path = os.path.join(figdir, "bem_summary.csv")
    if os.path.exists(summary_path):
        bem_summary = pd.read_csv(summary_path).set_index("lam")
        rows = []
        for res, lam in zip(results, lams):
            if lam in bem_summary.index:
                ct_b = bem_summary.loc[lam, "CT"]
                cp_b = bem_summary.loc[lam, "CP"]
                dct_pct = 100.0 * (res['CT'] - ct_b) / ct_b
                dcp_pct = 100.0 * (res['CP'] - cp_b) / cp_b
                rows.append([f"{lam:.0f}",
                             f"{res['CT']:.4f}", f"{ct_b:.4f}",
                             f"{dct_pct:+.2f}\\%",
                             f"{res['CP']:.4f}", f"{cp_b:.4f}",
                             f"{dcp_pct:+.2f}\\%"])
        if rows:
            fig, ax = plt.subplots(figsize=(7.5, 1.8))
            ax.axis("off")
            tbl = ax.table(
                cellText=rows,
                colLabels=["λ", "CT (LL)", "CT (BEM)", "δCT [%]",
                           "CP (LL)", "CP (BEM)", "δCP [%]"],
                loc="center", cellLoc="center")
            tbl.auto_set_font_size(False); tbl.set_fontsize(10)
            fig.tight_layout()
            fig.savefig(os.path.join(figdir, "bem_comparison_CT_CP_table.pdf"),
                        bbox_inches="tight")
            plt.close(fig)
            _write_bem_comparison_latex_table(rows, figdir)
    else:
        warnings.warn(
            f"BEM summary file not found: {summary_path}. "
            "Re-run generate_bem_csv.py to produce CT/CP for the comparison table.",
            stacklevel=2,
        )


def _write_baseline_latex_table(results: list[dict], figdir: str) -> None:
    """Write the baseline CT/CP summary as a ready-to-include LaTeX table."""
    lines = [
        r"\begin{table}[htbp]",
        r"    \centering",
        r"    \caption{Thrust and power coefficients computed by the frozen-wake"
        r" lifting-line solver for tip speed ratios $\lambda \in \{6, 8, 10\}$.}",
        r"    \label{tab:baseline_CT_CP}",
        r"    \begin{tabular}{cccc}",
        r"        \toprule",
        r"        $\lambda$ [-] & $C_T$ [-] & $C_P$ [-] & Iterations \\",
        r"        \midrule",
    ]
    for res in results:
        lines.append(
            f"        {res['lam']:.0f} & {res['CT']:.4f} & {res['CP']:.4f}"
            f" & {res['iters']} \\\\"
        )
    lines += [
        r"        \bottomrule",
        r"    \end{tabular}",
        r"\end{table}",
    ]
    path = os.path.join(figdir, "baseline_CT_CP_table.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  LaTeX table  → '{path}'")


def _write_bem_comparison_latex_table(rows: list, figdir: str) -> None:
    """Write the LL vs BEM CT/CP comparison as a ready-to-include LaTeX table."""
    lines = [
        r"\begin{table}[htbp]",
        r"    \centering",
        r"    \caption{Thrust and power coefficients from the frozen-wake lifting-line"
        r" (LL) and blade element momentum (BEM) methods. Percentage differences are"
        r" defined as $\delta = (C^\mathrm{LL} - C^\mathrm{BEM})/C^\mathrm{BEM} \times 100\%$.}",
        r"    \label{tab:bem_comparison_CT_CP}",
        r"    \begin{tabular}{ccccccc}",
        r"        \toprule",
        r"        $\lambda$ [-] & $C_T^\mathrm{LL}$ & $C_T^\mathrm{BEM}$"
        r" & $\delta C_T$ [\%] & $C_P^\mathrm{LL}$ & $C_P^\mathrm{BEM}$ & $\delta C_P$ [\%] \\",
        r"        \midrule",
    ]
    for row in rows:
        lines.append("        " + " & ".join(row) + r" \\")
    lines += [
        r"        \bottomrule",
        r"    \end{tabular}",
        r"\end{table}",
    ]
    path = os.path.join(figdir, "bem_comparison_CT_CP_table.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  LaTeX table  → '{path}'")


def _ll_bem_legend(ax, colors, lams, loc="upper right") -> None:
    """Two-block legend: per-TSR color entries (LL) plus solid/dashed style key."""
    from matplotlib.lines import Line2D
    handles  = [Line2D([0], [0], color=c, label=f"λ={lam:.0f}")
                for c, lam in zip(colors, lams)]
    handles += [Line2D([0], [0], color="black", ls="-",  label="LL"),
                Line2D([0], [0], color="black", ls="--", label="BEM")]
    ax.legend(handles=handles, loc=loc, frameon=False, fontsize=8)


# -------------------------------------------------------------------
# Sensitivity sweep
# -------------------------------------------------------------------

def sensitivity_sweep(
    parameter: str,
    values: list,
    defaults: dict,
    polar: dict,
) -> list[dict]:
    """Re-run the full pipeline for each value of one parameter.

    Parameters
    ----------
    parameter : one of 'a_w', 'N', 'N_wake', 'dpsi_deg'
    values    : list of values to sweep
    defaults  : dict of default keyword arguments for build_geometry +
                keys U_inf, Omega, R, N_blades, pitch_deg, lam
    polar     : dict from load_polar

    Returns
    -------
    list of dicts; each is post_process output plus 'param_value'.
    """
    results = []
    for val in values:
        cfg = {**defaults, parameter: val}

        geom = build_geometry(
            R          = cfg["R"],
            r_root     = cfg["r_root"],
            N          = cfg["N"],
            N_blades   = cfg["N_blades"],
            N_wake     = cfg["N_wake"],
            dpsi_deg   = cfg["dpsi_deg"],
            U_inf      = cfg["U_inf"],
            Omega      = cfg["Omega"],
            a_w        = cfg["a_w"],
            distribution = cfg.get("distribution", "cosine"),
        )
        geom["pitch"] = np.radians(cfg["pitch_deg"])

        U_m, V_m, W_m = assemble_influence(geom["cp"], geom["bound"],
                                           geom["wake"], geom["r_core"])

        sol = solve_circulation(
            U_m, V_m, W_m, geom, polar,
            U_inf   = cfg["U_inf"],
            Omega   = cfg["Omega"],
            w_relax = cfg.get("w_relax", 0.3),
            tol     = cfg.get("tol", 1e-4),
            max_iter= cfg.get("max_iter", 2000),
        )

        res = post_process(sol, geom, cfg)
        res["param_value"] = val
        results.append(res)

    return results


def make_sensitivity_figures(
    sweep_results: dict[str, list[dict]],
    baseline: dict,
    figdir: str = "figures",
) -> None:
    """Four multi-panel sensitivity figures; baseline curve highlighted.

    Parameters
    ----------
    sweep_results : dict mapping parameter name → list[dict] from sensitivity_sweep
    baseline      : post_process dict for the baseline configuration
    figdir        : output directory
    """
    os.makedirs(figdir, exist_ok=True)

    param_meta = {
        "a_w":     (r"$a_w$",         "a_w"),
        "N":       (r"$N$",           "N"),
        "N_wake":  (r"$N_\mathrm{wake}$", "N_wake"),
        "dpsi_deg":(r"$\Delta\psi$ [deg]", "dpsi_deg"),
    }

    quantities = [
        ("Gamma_hat", r"$\hat{\Gamma}$"),
        ("a",         r"$a$"),
        ("CT",        None),     # scalar — plotted as a function of param value
        ("CP",        None),
    ]

    with plt.rc_context(_STYLE):
        for param, results in sweep_results.items():
            label_tex, _ = param_meta.get(param, (param, param))
            n_res = len(results)
            cmap  = plt.cm.plasma(np.linspace(0.1, 0.9, n_res))

            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            axes = axes.flatten()

            # Panel 0 & 1: Gamma_hat and a vs r/R
            for res, c in zip(results, cmap):
                lw = 2.5 if res["param_value"] == baseline.get(param.replace("_deg","")) \
                     or res["param_value"] == baseline.get(param) else 1.0
                axes[0].plot(res["r_R"], res["Gamma_hat"], color=c,
                             lw=lw, label=f"{res['param_value']}")
                axes[1].plot(res["r_R"], res["a"],         color=c, lw=lw)

            axes[0].set_xlabel(r"$r/R$"); axes[0].set_ylabel(r"$\hat{\Gamma}$")
            axes[0].set_title(r"Bound circulation $\hat{\Gamma}$")
            axes[0].legend(title=label_tex, frameon=False, fontsize=8)

            axes[1].set_xlabel(r"$r/R$"); axes[1].set_ylabel(r"$a$")
            axes[1].set_title("Axial induction factor")

            # Panel 2 & 3: CT and CP vs parameter value
            pvals = [r["param_value"] for r in results]
            CTs   = [r["CT"] for r in results]
            CPs   = [r["CP"] for r in results]

            axes[2].plot(pvals, CTs, "o-", color="steelblue")
            axes[2].axhline(baseline["CT"], ls="--", color="gray", lw=1, label="baseline")
            axes[2].set_xlabel(label_tex); axes[2].set_ylabel(r"$C_T$")
            axes[2].set_title(r"Thrust coefficient $C_T$")
            axes[2].legend(frameon=False, fontsize=8)

            axes[3].plot(pvals, CPs, "s-", color="firebrick")
            axes[3].axhline(baseline["CP"], ls="--", color="gray", lw=1, label="baseline")
            axes[3].set_xlabel(label_tex); axes[3].set_ylabel(r"$C_P$")
            axes[3].set_title(r"Power coefficient $C_P$")
            axes[3].legend(frameon=False, fontsize=8)

            fig.suptitle(f"Sensitivity to {label_tex}", y=1.01)
            fig.tight_layout()
            fname = os.path.join(figdir, f"sensitivity_{param}.pdf")
            fig.savefig(fname, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved {fname}")
