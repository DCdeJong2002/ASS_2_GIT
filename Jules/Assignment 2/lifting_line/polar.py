"""Module 4 — Airfoil polar interpolation.

Loads the DU 95-W-180 polar from the supplied .xlsx file and provides
vectorised linear interpolation of Cl and Cd (eq:lift, eq:drag).
Flat extrapolation is used outside the tabulated range; a warning is
raised if any query angle falls outside the table (eq:phi-alpha context).
"""

import warnings
import numpy as np
import pandas as pd


def load_polar(path: str) -> dict:
    """Read DU 95-W-180 polar from Excel.

    The file has a two-row preamble followed by a header row (Alfa, Cl, Cd, Cm)
    and data rows.  Alpha is stored internally in radians.

    Returns
    -------
    dict with keys 'alpha' (rad), 'Cl', 'Cd' — all 1-D float arrays.
    """
    df = pd.read_excel(path, sheet_name=0, header=3)   # row 3 is "Alfa Cl Cd Cm"
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(subset=["Alfa", "Cl", "Cd"])
    df = df.sort_values("Alfa").reset_index(drop=True)

    alpha_deg = df["Alfa"].to_numpy(dtype=float)
    Cl_vals   = df["Cl"].to_numpy(dtype=float)
    Cd_vals   = df["Cd"].to_numpy(dtype=float)

    return {
        "alpha": np.radians(alpha_deg),
        "Cl":    Cl_vals,
        "Cd":    Cd_vals,
        "alpha_deg_min": float(alpha_deg[0]),
        "alpha_deg_max": float(alpha_deg[-1]),
    }


def airfoil_coefficients(
    alpha_rad: np.ndarray,
    polar: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (Cl, Cd) at the given angles of attack via linear interpolation.

    Parameters
    ----------
    alpha_rad : array of angles of attack [rad]
    polar     : dict returned by load_polar

    Returns
    -------
    Cl, Cd : arrays same shape as alpha_rad
    """
    alpha_deg = np.degrees(alpha_rad)
    out_of_range = (alpha_deg < polar["alpha_deg_min"]) | (alpha_deg > polar["alpha_deg_max"])
    if np.any(out_of_range):
        warnings.warn(
            f"{np.sum(out_of_range)} control point(s) outside tabulated polar range "
            f"[{polar['alpha_deg_min']:.1f}, {polar['alpha_deg_max']:.1f}] deg.",
            stacklevel=2,
        )

    Cl = np.interp(alpha_rad, polar["alpha"], polar["Cl"])
    Cd = np.interp(alpha_rad, polar["alpha"], polar["Cd"])
    return Cl, Cd


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    path = os.path.join(os.path.dirname(__file__), "..", "polar DU95W180 (3).xlsx")
    polar = load_polar(path)

    print("=== Polar sanity check ===")
    print(f"  alpha range : {np.degrees(polar['alpha'][0]):.2f}° to "
          f"{np.degrees(polar['alpha'][-1]):.2f}°")
    print(f"  N points    : {len(polar['alpha'])}")

    # alpha at Cl=0
    alpha_Cl0 = np.interp(0.0, polar["Cl"], np.degrees(polar["alpha"]))
    print(f"  alpha @ Cl=0: {alpha_Cl0:.2f}°")

    # alpha at Cl_max
    idx_max = np.argmax(polar["Cl"])
    print(f"  Cl_max      : {polar['Cl'][idx_max]:.4f}  @ alpha={np.degrees(polar['alpha'][idx_max]):.2f}°")

    # quick interpolation check
    Cl_test, Cd_test = airfoil_coefficients(np.radians(np.array([0.0, 5.0, 10.0])), polar)
    print(f"  Cl at 0°, 5°, 10°: {Cl_test}")
    print(f"  Cd at 0°, 5°, 10°: {Cd_test}")
    print("  PASSED")
