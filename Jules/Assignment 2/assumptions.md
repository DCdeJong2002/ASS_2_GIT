# Main Assumptions of the Lifting-Line Model

This document lists the principal assumptions made in the frozen-wake lifting-line
solver implemented in [Assignment 2/lifting_line/](lifting_line/) and explains
how each one affects the predicted loads, induction, and circulation.

---

## 1. Frozen (prescribed) helical wake

**Assumption.** The trailing vortex filaments are laid down on a rigid helix
that propagates downstream at a constant convection speed
`U_wake = U_inf · (1 − a_w)`, where `a_w` is a single scalar axial-induction
factor chosen *a priori* (in the baseline, it is the span-averaged BEM
induction — see [run_baseline.py:41-50](run_baseline.py#L41-L50)). The wake
geometry is built once in [geometry.py:57-89](lifting_line/geometry.py#L57-L89)
and never updated.

**Impact.**
- Decouples wake geometry from circulation, so the Biot–Savart influence
  matrices `[U], [V], [W]` are assembled exactly once
  ([influence.py](lifting_line/influence.py)); the per-iteration cost
  collapses to three O(N²) matrix–vector products in
  [solver.py:80-82](lifting_line/solver.py#L80-L82).
- The wake cannot self-deform, so radial contraction/expansion and roll-up
  of tip vortices are ignored. Tip-region induction is therefore biased —
  typically the predicted tip loading is slightly too high because the
  wake is held at the rotor radius rather than contracting inboard.
- Results depend on the choice of `a_w`. Using the BEM-averaged value
  ([run_baseline.py:55](run_baseline.py#L55)) couples the two assignments
  but means the LL solution inherits BEM error in the wake convection
  speed. This sensitivity is quantified in `run_sensitivity.py`.

## 2. Steady, axisymmetric, uniform inflow

**Assumption.** The freestream is a steady, uniform `U_inf` aligned with the
rotor axis (`+x`). All blades carry the same circulation distribution and
the problem is solved only for the reference blade at ψ=0; the other
`N_blades − 1` blades enter only through their image contributions in the
influence matrices ([influence.py:56-67](lifting_line/influence.py#L56-L67)).

**Impact.**
- No yaw, shear, tower shadow, or unsteady inflow effects are captured.
- Rotor symmetry collapses `N_blades · N` unknowns into just `N`, which
  is the main reason the solver is cheap.
- Loads and induction are azimuth-averaged by construction; per-blade
  azimuthal variation cannot be recovered.

## 3. Inviscid, incompressible potential flow + sectional viscous polar

**Assumption.** The induced velocity field is built from inviscid
Biot–Savart filaments ([biot_savart.py](lifting_line/biot_savart.py)),
but the sectional aerodynamic response uses the 2-D viscous polar
`Cl(α), Cd(α)` of the DU 95-W-180 airfoil
([polar.py](lifting_line/polar.py)). Lift closure follows Kutta–Joukowski
`Γ = ½ · c · V_p · Cl` ([solver.py:97](lifting_line/solver.py#L97)).

**Impact.**
- This is the standard "strip-theory" coupling: 3-D induction is potential-
  flow, but each spanwise station "sees" a 2-D viscous airfoil at the
  local effective angle of attack.
- Reynolds and Mach effects, finite-span 3-D stall delay, and rotational
  augmentation at the root are not modelled.
- `Cd` enters the tangential force but does *not* feed back into `Γ` (only
  `Cl` does), which is consistent with classical lifting-line theory but
  underestimates the effect of profile drag on the wake.
- Outside the tabulated polar range (≈ −16° to +30°), `np.interp` falls
  back to flat extrapolation. This is only reached at the innermost panels
  for low TSR; the solver issues a warning when it happens but otherwise
  trusts the flat-extrapolated values.

## 4. Straight bound vortex on the c/4 line + point control point at c/4

**Assumption.** Each spanwise panel carries a single straight bound
filament along the quarter-chord line of the reference blade
([geometry.py:50-55](lifting_line/geometry.py#L50-L55)); the control
point at which the boundary condition is enforced sits at the panel
midpoint, also on the c/4 line. Camber, thickness, and the airfoil's
detailed pressure distribution are *not* discretised — they live
entirely in the polar.

**Impact.**
- This is Prandtl's classical lifting-line, not a higher-order vortex-
  lattice or panel method. The induced velocity at the c/4 line is the
  one used for both Γ closure and angle-of-attack evaluation.
- No correction is made for swept or coned blades; the blade is straight
  and untilted. For the given geometry (no sweep, no cone) this is exact.

## 5. Horseshoe topology with shed wake aligned to chord line

**Assumption.** Each panel is closed by two short connector filaments
running from the c/4 nodes to the trailing-edge wake-shedding nodes,
located at `5c/4` along the chord line (i.e. at the trailing edge if
the chord is c). These connectors are shown in
[influence.py:13-23](lifting_line/influence.py#L13-L23) and the offsets
are built in [geometry.py:68-89](lifting_line/geometry.py#L68-L89).
Adjacent horseshoes share trailing legs whose net circulation is the
jump `Γⱼ − Γⱼ₊₁`, automatically satisfying Helmholtz's theorem.

**Impact.**
- The wake is shed from the trailing edge in line with the local twisted
  chord, not from the c/4 line. This is physically more correct than
  shedding straight from c/4 and slightly improves the predicted
  tangential induction.
- The connector filaments are short and approximately aligned with the
  chord, so they contribute little to the bound circulation closure but
  are needed to keep the horseshoe topologically closed.

## 6. Vortex-core regularisation by solid-body rotation

**Assumption.** Inside a finite core radius `r_core` (default `1e-6 · R`
in [geometry.py:30-31](lifting_line/geometry.py#L30-L31)) the Biot–Savart
kernel is regularised by clipping `|R₁ × R₂|²` and `|R₁|, |R₂|` from
below ([biot_savart.py:58-60](lifting_line/biot_savart.py#L58-L60)),
giving a solid-body-rotation profile inside the core rather than the
singular `1/r` behaviour.

**Impact.**
- Prevents NaN/Inf when a control point is very close to a filament
  (e.g. on its own bound vortex). The matrix entries stay bounded.
- With `r_core = 1e-6 · R` the regularisation is "tight": it does not
  visibly smear the induced velocity field. A larger `r_core` would
  smooth Γ near the tip and root but artificially reduce peak loading.

## 7. Helmholtz / vorticity-conservation laws

**Assumption.** The vortex system satisfies Helmholtz's theorems exactly
by construction: bound circulation strength varies only spanwise,
shed trailing vorticity equals the spanwise gradient of Γ, and
filaments form closed loops (horseshoe topology in
[influence.py:13-23](lifting_line/influence.py#L13-L23)). No starting
vortex is modelled because the flow is steady.

**Impact.**
- The trailing-vortex strengths automatically match `dΓ/dr` without
  needing an explicit conservation equation in the solver.
- Steady-state assumption means no shed vorticity due to time-varying
  bound circulation (no `dΓ/dt` term).

## 8. Single-airfoil polar across the entire span

**Assumption.** The same DU 95-W-180 polar is used from `r_root = 0.2 R`
to `R` ([run_baseline.py:37](run_baseline.py#L37)). The root cylinder
(`r < 0.2 R`) is excluded from the calculation entirely.

**Impact.**
- Real wind-turbine blades use a thick cylindrical/transition profile
  near the root; assuming the design airfoil all the way in
  overestimates root lift and underestimates root drag.
- Loads and torque integrals are not affected much (small lever arm,
  small chord-times-Cl), but local α and Γ near `r/R = 0.2` are biased.

## 9. Neglect of 3-D rotational effects on the sectional polar

**Assumption.** The 2-D polar `Cl(α), Cd(α)` is used unmodified at every
spanwise station ([polar.py](lifting_line/polar.py),
[solver.py:94](lifting_line/solver.py#L94)). The Coriolis and centrifugal
("fictitious") forces that act on the boundary layer of a rotating blade
are not modelled, and no Snel/Du–Selig/Chaviaropoulos-type 3-D correction
is applied.

**Impact.**
- On a rotating blade the centrifugal pumping drives the near-wall flow
  outward and the Coriolis force pushes it toward the trailing edge,
  delaying separation and raising `Cl_max` and the effective stall angle
  — the "rotational augmentation" or "Himmelskamp" effect.
- If the polar was measured on a non-rotating section (as is typical for
  the DU 95-W-180 table), the tabulated stall angle is *under-predicted*
  for the rotating blade. The effect is strongest at the inboard
  stations where `Ω·r / U_inf` is small and the chord-to-radius ratio is
  large.
- Consequence: the solver tends to predict premature stall at the root,
  under-estimating Γ and the inboard contribution to power. The bias is
  small in CP/CT integrals but visible in the spanwise α and Cl curves
  near `r/R ≈ 0.2`.

## 10. Infinitely rigid blade (no aeroelastic deflection)

**Assumption.** The blade geometry built in
[geometry.py:46-47](lifting_line/geometry.py#L46-L47) uses only the
static, built-in chord, twist, and collective pitch
([run_baseline.py:28](run_baseline.py#L28)). The blade does not bend,
twist, or cone under load, and no coupling between aerodynamic loading
and structural deflection is included.

**Impact.**
- Real composite wind-turbine blades undergo significant flapwise
  bending and torsional deflection at rated conditions. Torsional
  deflection ("bend-twist coupling") changes the *effective* local
  twist, which feeds directly into `α = φ − twist − pitch` in
  [solver.py:91](lifting_line/solver.py#L91); ignoring it shifts `α`,
  `Cl`, and `Cd` away from their true operating values.
- Flapwise bending causes the blade to adopt a downwind cone shape,
  reducing the effective swept area and the component of the relative
  velocity perpendicular to the rotor plane.
- Because both effects are load-relieving, a perfectly rigid model
  tends to *over-predict* CT and CP at high loading (low TSR / high
  thrust). For a stiff design at moderate loading the bias is small,
  but it grows with rotor size and with operating point.
- No structural natural frequencies are present, so the steady-flow
  assumption (2) is not violated by aeroelastic coupling — but the
  solver also cannot be used to investigate flutter or whirl-flutter
  margins.

## 11. Fixed-point iteration with vector-Aitken under-relaxation

**Assumption.** Γ is solved by Picard iteration with a Kutta–Joukowski
closure ([solver.py:78-127](lifting_line/solver.py#L78-L127)). The
relaxation factor is recomputed every step from the current and previous
residual vectors (vector Aitken update) and clamped to `[0.01, 1.0]`.

**Impact.**
- Numerical assumption only — it does not change the converged solution,
  only how fast it is reached. Without dynamic relaxation the scheme can
  fall into a stable 2-cycle limit near stall (high `dCl/dα`); Aitken
  breaks that limit cycle.
- Convergence is monitored on `max|ΔΓ| / max|Γ|` with `tol = 1e-5`. The
  tolerance is tight enough that round-off, not iteration error, sets
  the accuracy of the integrated CT, CP.

---

## Summary — which assumptions matter most?

| Assumption | Strongest effect on |
|---|---|
| Frozen wake (1) | Tip-region induction & sensitivity to `a_w` |
| Steady uniform inflow (2) | Excludes yaw / shear / unsteady physics entirely |
| Inviscid + polar coupling (3) | Cd's missing feedback on Γ |
| Single airfoil (8) | Inboard load distribution |
| No 3-D rotational augmentation (9) | Premature root stall, inboard Γ bias |
| Rigid blade (10) | Over-prediction of CT, CP at high loading |
| Core regularisation (6) | Numerical stability only at chosen `r_core` |

Assumptions 4, 5, 7, 11 are standard lifting-line ingredients that do not
introduce additional physical error beyond what is already implied by
assumptions 1–3 and 9–10.
