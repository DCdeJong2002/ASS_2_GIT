# Context for analysing the BEM-vs-Lifting-Line comparison plots

This is a self-contained briefing for a fresh Claude conversation. The goal is
to give whoever reads it everything they need to interpret the three figures

- `bem_comparison_lam6.pdf`
- `bem_comparison_lam8.pdf`
- `bem_comparison_lam10.pdf`

produced by [Assignment 2/run_baseline.py](run_baseline.py) +
[Assignment 2/lifting_line/postprocess.py](lifting_line/postprocess.py)
`_plot_bem_comparison`, without needing access to the rest of the repo.

---

## 1. Course and assignment context

Coursework for **AE4135 Rotor & Wake Aerodynamics** (TU Delft, Q3 2025/26).
Two assignments analyse the same rotor:

- **Assignment 1** — Blade-Element-Momentum (BEM) solver with Glauert
  high-induction correction + Prandtl tip/root loss.
- **Assignment 2** — Frozen-wake lifting-line (LL) solver. *Main deliverable.*
  Results are validated against the Assignment-1 BEM output.

Both share the same airfoil polar (DU 95-W-180) and the same nominal
geometry. The three comparison plots are the headline cross-validation:
do two independent models, applied to the same turbine, agree?

## 2. Turbine specification (identical in both solvers)

| Quantity | Value |
|---|---|
| Rotor radius `R` | 50 m |
| Root cut-out `r_root` | 0.2 R = 10 m |
| Number of blades `B` | 3 |
| Freestream `U_inf` | 10 m/s |
| Air density `ρ` | 1.225 kg/m³ |
| Collective pitch | −2° |
| Chord distribution | `c(r) = 3·(1 − r/R) + 1` [m] |
| Twist distribution | `θ(r) = 14·(1 − r/R)` [deg] |
| Airfoil | DU 95-W-180 (single polar across the full span) |
| Tip-speed ratios analysed | λ ∈ {6, 8, 10} |

`Ω = λ · U_inf / R`. So Ω ≈ 1.2, 1.6, 2.0 rad/s for the three TSRs.

## 3. What the comparison plots show

Each PDF is a 2×2 grid of subplots, all with `x = r/R` on the horizontal axis
(spanwise position from root cut-out 0.20 to tip 1.00). Plotted curves:

| Subplot | Quantity | Symbol | Non-dim group |
|---|---|---|---|
| top-left  | Axial induction          | `a`           | `−u_ind / U_inf` |
| top-right | Tangential induction     | `a'`          | `−v_ind / (Ω r)` |
| bot-left  | Bound circulation        | `Γ̂` (Gamma_hat) | `Γ · B · Ω / (π U_inf²)` |
| bot-right | Axial loading per unit span | `F̂_ax`     | `F_ax / (½ ρ U_inf² R)` |

Two curves per subplot:

- **Solid coloured** — Lifting Line result (one colour per TSR; viridis
  palette).
- **Black dashed** — BEM result at the same TSR.

Title of each figure: `Lifting line vs BEM, λ=…`.

> The CSVs the dashed BEM curves come from are
> `figures/bem_lam{6,8,10}.csv`, produced by
> [Assignment 2/generate_bem_csv.py](generate_bem_csv.py). Their columns are
> `r_R, a, a_prime, alpha_deg, phi_deg, Gamma, Gamma_hat, F_hat_ax, F_hat_azim`.
> Only the four quantities above are actually plotted; the rest are
> available if you want to ask deeper questions.

## 4. BEM model — what produced the dashed curves

Implementation: [Assignment 2/generate_bem_csv.py](generate_bem_csv.py).

### Algorithm (per annulus, iterated to convergence)

For each radial annulus at radius `r` with width `dr`, chord `c`, and twist+pitch `β`:

1. Velocity triangle with current `(a, a')`:
   - `V_ax  = U_inf · (1 − a)`
   - `V_tan = Ω r · (1 + a')`
   - `V_rel = √(V_ax² + V_tan²)`
   - `φ = atan2(V_ax, V_tan)` (inflow angle)
   - `α = φ − β`
2. Polar lookup → `Cl(α)`, `Cd(α)` from the DU 95-W-180 table
   (`scipy.interpolate.interp1d`, linear, flat extrapolation).
3. Decompose into normal / tangential aerodynamic coefficients:
   - `Cn = Cl cos φ + Cd sin φ`
   - `Ct = Cl sin φ − Cd cos φ`
4. **Prandtl tip + root loss** `F = F_tip · F_root` with
   `d = (2π/B)(1 − a)/√(λ² + (1 − a)²)`,
   `F_tip = (2/π) acos(exp(−π (1 − r/R)/d))`,
   `F_root = (2/π) acos(exp(−π (r/R − r_root/R)/d))`.
5. **Glauert high-induction correction** for `a`:
   - local thrust coefficient `CT_loc = σ (1 − a)² Cn / sin²φ` with
     solidity `σ = B c / (2π r)`.
   - If `CT_loc < 2√1.816 − 1.816 ≈ 0.9609`: momentum branch
     `a_new = ½ − ½ √(1 − CT_loc)`.
   - Else: empirical Glauert branch
     `a_new = 1 + (CT_loc − 1.816)/(4√1.816 − 4)`.
6. Tangential induction: `a'_new = 1 / [ 4 sinφ cosφ / (σ Ct) − 1 ]`.
7. Divide both inductions by `F` (Prandtl correction is applied to the
   inductions, not to the loads), clip to `[−0.5, 0.95]`.
8. Under-relaxed update with factor 0.25; loop until `|ΔCT|<1e−5` or
   300 iterations.

### BEM discretisation

- `N_annuli = 100`, cosine-spaced nodes between `r_root` and `R`,
  mid-points of each annulus are used as evaluation points.
- This is **much finer** than the LL's `N=30` spanwise stations.

### Key implicit assumptions in BEM

- Each annulus is independent — no cross-talk between radii.
- Wake is fully developed, axially aligned, and ideally captured by 1-D
  momentum theory + the Glauert empirical fit above `a ≈ 0.4`.
- Prandtl correction is the only mechanism for finite-blade-number /
  tip-loss effects.
- Steady, axisymmetric, uniform inflow.

## 5. Lifting-line model — what produced the solid curves

Implementation: 6-module pipeline in
[Assignment 2/lifting_line/](lifting_line/). The full set of assumptions
is in [Assignment 2/assumptions.md](assumptions.md); here is the short
version aimed at making sense of the plots.

### Discretisation defaults (baseline)

| Parameter | Value | Meaning |
|---|---|---|
| `N` | 30 | spanwise panels, cosine-spaced |
| `N_wake` | 10 | number of full rotor revolutions of frozen helical wake |
| `dpsi` | 10° | azimuthal step of wake discretisation |
| `a_w` | span-averaged BEM `a` for that TSR | wake convection slowdown |
| `r_core` | `1e-6 · R` | Biot–Savart core radius |
| distribution | cosine | spanwise node spacing |

### Algorithm summary

1. **Geometry** — `N` cosine-spaced bound panels on the c/4 line of the
   reference blade (+z axis at ψ=0). For each panel, two trailing helical
   filaments are released from the trailing edge (offset 5c/4 along the
   chord line). The helix propagates at `U_wake = U_inf · (1 − a_w)` for
   `N_wake` revolutions. The other `B−1` blades are rotated copies; they
   contribute only through the influence matrix (rotor symmetry collapses
   `B·N` unknowns into `N`).
2. **Influence matrices** — `[U], [V], [W]` are N×N, built once from
   Biot–Savart over every filament (Katz & Plotkin closed form,
   solid-body core regularisation by clipping `|R₁ × R₂|²`,
   `|R₁|`, `|R₂|`).
3. **Fixed-point loop**, per iteration:
   - Induced velocities `u_ind = U·Γ`, `v_ind = V·Γ`, `w_ind = W·Γ`.
   - Local velocity at control point, inflow angle φ, angle of attack
     `α = φ − twist − pitch`.
   - Polar lookup `Cl(α), Cd(α)`.
   - Kutta–Joukowski closure: `Γ_new = ½ · c · V_p · Cl`.
   - Vector Aitken under-relaxation; convergence on
     `max|ΔΓ|/max|Γ| < 1e−5`.
4. **Post-processing** — induction factors come *directly* from the
   converged induced velocities:
   - `a = −u_ind / U_inf`
   - `a' = −v_ind / (Ω r)`

   Note: in LL there is **no Prandtl correction and no Glauert correction**
   — finite-blade tip-loss is captured implicitly by the discrete vortex
   wake, and high-induction effects are simply *not modelled* (the LL is
   a potential-flow code).

### Key LL assumptions

(Distilled from [assumptions.md](assumptions.md); see that file for the
full list of 11.)

1. **Frozen helical wake** at fixed `a_w` — wake cannot self-deform, so
   no contraction, no roll-up. Tip-region induction is typically biased
   slightly high.
2. **Steady, axisymmetric, uniform inflow**; rotor symmetry exploited.
3. **Inviscid Biot–Savart + 2-D viscous polar** ("strip-theory coupling").
   `Cd` enters the tangential force but does **not** feed back into Γ —
   only `Cl` does.
4. **Straight bound vortex on c/4**, point control point at the panel
   midpoint on c/4. Camber and thickness live entirely in the polar.
5. **Horseshoe topology**: each panel closed by short connector filaments
   from c/4 to 5c/4 along the chord line; shed wake leaves from the
   trailing edge, not from c/4. Helmholtz satisfied by construction.
6. **No 3-D rotational augmentation** (no Snel/Du–Selig/Chaviaropoulos
   correction). Premature root stall is a known bias.
7. **Single airfoil across the entire span** from `r/R = 0.2` to 1.0.
   Real blades have a thick transition profile at the root; assuming the
   design airfoil there overestimates root lift.

## 6. Expected differences between BEM and LL — and why

This is the analytical core. The two models *should* agree in the
"momentum-theory regime" (moderately loaded, attached flow, away from
tip and root), and they *should* disagree in specific, interpretable
ways elsewhere.

### Mid-span (≈ 0.4 ≲ r/R ≲ 0.85)

- Both models reduce to nearly the same physics: 2-D polar in a
  potential flow with corrections for finite blade number.
- Expect **good agreement** on all four quantities. Discrepancies here
  hint at solver bugs, not modelling differences.

### Tip region (r/R → 1)

- **BEM** uses Prandtl `F_tip` to pull `a` to a finite value at the tip;
  `Γ → 0` is forced indirectly through the loss factor on the inductions.
- **LL** captures finite-blade tip-loss through the discrete trailing
  vortex sheet itself. Γ → 0 is enforced automatically by Kutta-
  Joukowski (because `V_p` rotates so that `Cl → 0` only when the
  effective `α` drops; in practice the LL has *no* fixed boundary
  condition Γ(R)=0 — the wake handles it).
- A frozen, non-contracting helix at the rotor radius typically gives
  **higher induced velocity** near the tip than reality, so the LL
  often predicts *higher* `a` and *higher* tip-region `F̂_ax` than BEM.
  Γ̂ shape near the tip is the most diagnostic — both curves should
  drop to zero, but the LL one may drop more sharply over a smaller
  span fraction.

### Root region (r/R → 0.2)

- **BEM** applies Prandtl `F_root`; with the same airfoil all the way
  in, this pulls `a` down at the root.
- **LL** has no root-loss correction (root vortex sheet does some of
  this job, but only weakly), and the polar may flat-extrapolate if α
  exceeds ±30° in the inboard panels. The combination of large local
  chord, large twist, and missing 3-D rotational augmentation can
  produce premature stall in LL → an inboard dip in Γ̂ and `F̂_ax`
  that BEM may smooth over.

### High-TSR (λ = 10)

- Inflow angles are small, α is small everywhere, both models are in
  the linear `Cl(α)` regime. Expect the **cleanest agreement** of the
  three TSRs.
- `a` is moderate (≲ 0.4), so the Glauert branch of BEM is not
  triggered. The two models share their assumed regime of validity.

### Low-TSR (λ = 6)

- High loading inboard: BEM may enter its Glauert empirical branch
  (`CT_loc > 0.96`) at some radii, while LL has no such correction.
- Inboard α can be large enough to brush against stall. Differences
  here are amplified by `dCl/dα` non-linearity.
- Expect the **largest BEM/LL discrepancies** at λ=6, mostly inboard.

### Design TSR (λ = 8)

- Near-design operating point. Should be intermediate: clean agreement
  outboard, modest inboard differences.

### Quantity-specific notes

- **`a`**: BEM has its momentum-theory closure, LL has its wake-induced
  velocity directly. The shape of `a(r)` is the most direct comparison
  of "how much axial slowdown does each model see at each radius?".
  Step / dip artefacts in LL near the tip are a sign that `a_w` (wake
  speed) is mis-matched to the actual loading.
- **`a'`**: BEM gets `a'` from local angular momentum balance per
  annulus. LL gets it from the azimuthal component of the induced
  velocity. These can differ visibly even when `a` agrees — the
  tangential induction is more sensitive to wake topology.
- **`Γ̂`**: Both models compute Γ from `½ c V_rel Cl`. So any
  disagreement traces back to a difference in `α` (and hence `Cl`)
  and / or in `V_rel`. This makes Γ̂ a *sensitive* indicator of the
  underlying difference in inflow geometry, not an independent check.
- **`F̂_ax`**: For BEM this is `(dT/dr)/B` non-dimensionalised; for LL
  it is `L cosφ + D sinφ` non-dimensionalised. Drag enters here but
  not in Γ̂, so a small offset between `F̂_ax` curves with matching
  `Γ̂` curves is the drag contribution.

## 7. Numerical details that can affect the plots

- **`a_w` coupling**: the LL solver uses the **BEM span-averaged `a`**
  for that TSR as its frozen wake convection factor
  ([run_baseline.py:41-50](run_baseline.py#L41-L50)). This is a
  deliberate choice: it means the two models are *not* independent —
  the LL inherits BEM's overall induction level by construction. The
  disagreement seen in the plots is therefore the disagreement in the
  *distribution*, not in the mean.
- **Different spanwise grids**: BEM has 100 cosine-spaced mid-points,
  LL has 30 cosine-spaced control points. Both grids cluster nodes at
  the tip and root, but the BEM curve will look smoother.
- **`r/R` range**: both start at `r/R ≈ 0.2`. LL's first control point
  is slightly inside (because cosine spacing starts at the panel edge
  `r_root`, with the first mid-point a fraction above 0.2); BEM's first
  mid-point is also slightly above 0.2. Small `x`-axis offsets at the
  innermost station are expected.
- **No post-stall extension** on the polar — flat extrapolation beyond
  the table. If you see weird `Γ̂` or `F̂_ax` artefacts at the innermost
  panels of LL only, suspect the polar going flat.

## 8. Reference equations (used in both models)

Non-dimensionalisations used in the comparison CSVs (so that BEM and LL
can be plotted on the same axes):

```
Γ̂      = Γ · B · Ω / (π · U_inf²)             (bound circulation)
F̂_ax   = F_ax  / (½ · ρ · U_inf² · R)         (axial loading per unit span)
F̂_azim = F_azim / (½ · ρ · U_inf² · R)        (azimuthal loading per unit span)
```

Where `F_ax`, `F_azim` are per-blade sectional forces (N/m) and `Γ` is
the per-blade bound circulation (m²/s).

Kutta–Joukowski closure (both solvers):

```
Γ = ½ · c · V_rel · Cl(α)
```

Velocity triangle:

```
V_ax  = U_inf · (1 − a)        # for BEM
V_tan = Ω · r · (1 + a')       # for BEM
V_rel = √(V_ax² + V_tan²)
φ     = atan2(V_ax, V_tan)
α     = φ − (twist + pitch)
```

In LL, `V_ax` and `V_tan` are instead built from the **converged induced
velocity** at the control point, then `a` and `a'` are *recovered* from
those velocities; they are not iteration variables.

## 9. Quick analyst checklist for the plots

When asked to "analyse the BEM comparison plots", a good answer covers:

1. **Quality of agreement** at each TSR, in each region (root / mid-span
   / tip), for each of the four quantities.
2. **Where the two curves disagree most**, and which of the listed
   physical / numerical differences most plausibly explains it
   (Prandtl loss vs frozen-wake-vs-real-wake topology; Glauert branch
   activation; root stall; etc.).
3. **TSR trend**: does the agreement get better at higher λ (as
   expected for a less-loaded rotor)?
4. **Internal consistency**: if `Γ̂` curves agree but `F̂_ax` doesn't,
   that's a `Cd`-driven effect. If `a` agrees but `a'` doesn't, that's
   wake-topology-driven.
5. **Whether the disagreement is *physical* (different models capturing
   different physics) or *numerical* (grid resolution, polar
   extrapolation)**.

The two solvers are deliberately built from the same airfoil polar,
same geometry, same `Ω`, and (via `a_w`) the same mean induction.
Disagreements in the spanwise *distribution* are the interesting
physics — not solver bugs.
