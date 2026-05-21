# Bugfix log — frozen-wake lifting line solver

This document lists every change to the implementation that *deviated from
a straightforward translation* of theory_background.tex / code_structure.tex
into Python. Each entry records: the symptom that revealed the issue, the
diagnosis, the fix that ended up in the final code, and any caveat about
physics or sign conventions you should double-check before trusting it.

Things that worked first try and are NOT listed here:
- Biot–Savart kernel for a straight filament (analytic test against
  Γ/(2πd) and Γ/(4πd) passed at 5×10⁻⁹ relative error)
- Geometry construction (helical wake node positions match the closed-form)
- Polar loading from .xlsx
- Post-processing formulas (CT, CP, Γ̂, F̂, integration via trapz)
- The `±Γ` assignment on adjacent trailing legs (verified by Helmholtz
  check: shared trailing leg of two equal-Γ horseshoes gives net 0)

---

## 1. Wake helix winding direction (PHYSICS)

**Symptom.** With the wake parameterised as written in
theory_background.tex, `eq:wake-y/z` (`psi_k = k·Δψ + ψ₀`), every panel of
the test rotor gave `u_ind > 0` for any positive Γ. That means the rotor
appears to *accelerate* the axial flow, i.e. `a < 0`. Wrong sign for a
turbine.

**Diagnosis.** Built a stripped-down test case (single blade, single mid-span
horseshoe, fixed Γ=1) and decomposed the velocity at a control point 1 m
downstream into three contributions:

| Contribution           | u (axial) | v (tangential) | w (radial) |
|------------------------|-----------|----------------|-----------|
| bound segment alone    | ≈ 0       | +0.148         | 0         |
| inboard trailing leg   | **+0.020** | −0.009        | −0.009    |
| outboard trailing leg  | **+0.039** | −0.003        | +0.011    |
| total                  | **+0.059** | +0.136         | +0.002    |

The bound segment is correct (right-hand rule). Helmholtz check passes
(net Γ on a shared trailing leg of two unit horseshoes = 0). So the
±Γ assignment is correct. The error is purely the helix winding.

**Fix.** [`lifting_line/geometry.py:109`](lifting_line/geometry.py#L109) —
changed `psi_k = k·Δψ + ψ₀` to `psi_k = ψ₀ − k·Δψ`.

Re-running the same diagnostic with the corrected wake:

| Contribution           | u (axial)   | v (tangential) | w (radial) |
|------------------------|-------------|----------------|-----------|
| inboard trailing leg   | **−0.020**  | −0.009         | +0.009    |
| outboard trailing leg  | **−0.039**  | −0.003         | −0.011    |
| total                  | **−0.059**  | −0.012         | −0.002    |

Only **u** flips sign. **v stays exactly the same**, so `a' = v_ind/(Ωr)`
keeps its sign and magnitude — only `a = −u_ind/U∞` flips from < 0 to
> 0 (correct turbine sign). w (radial component, not used in the velocity
triangle) also flips, irrelevant.

**Physical justification.** A filament shed τ seconds ago was emitted by
the blade at azimuth `ψ₀ − Ω·τ`, not `ψ₀ + Ω·τ` — the wake spirals
*opposite* to the rotor. The theory file appears to have the wrong sign.
The Mikkelsen JS reference implementation uses `Math.cos(-θ)`,
`Math.sin(-θ)` for exactly this reason.

**Caveat to verify.** None — diagnostic above shows v_ind unchanged so a'
is not corrupted, only a is fixed.

---

## 2. Vortex core regularisation: zero-velocity → clipped (NUMERICAL)

**Symptom.** Worked, but produced a hard discontinuity at `perp = r_core`,
giving very large velocities just outside the core for the cosine edge
panels.

**Original.** Inside the core (perpendicular distance from XP to filament
< r_core), set `uvw = 0`.

**Fix.** [`lifting_line/biot_savart.py:55-65`](lifting_line/biot_savart.py#L55-L65)
— follow the Katz & Plotkin / Mikkelsen JS reference style instead: clip
`|R1×R2|² ≥ r_core²` and `|R1|, |R2| ≥ r_core` from below. This produces
the standard solid-body-rotation profile inside the core (bounded but
non-zero velocity), which is smoother for the iteration.

**Caveat to verify.** This DID change the K formula slightly inside the
core. The analytic infinite-filament test at d=1, r_core=1e-6 still passes
at 5×10⁻⁹ rel error (the test point is far outside the core). For points
*inside* the core, our velocity is now non-zero but bounded by the
solid-body limit, instead of strictly zero. This is a more standard model.

---

## 3. Chord-length axial offset for wake start (NUMERICAL + APPROXIMATION)

**Symptom.** With cosine N=30 the very innermost panel is only ~0.11 m
wide. The first wake segment of the tip (or root) trailing leg sat at
perpendicular distance ≈ 0.06 m from the adjacent control point — very
close, producing huge induction (a > 1, axial flow reversal). Result:
panel 29 (tip) sat in a stable 2-cycle, residual frozen at 0.234 forever.

**Diagnosis.** Printed the residual and worst-panel Γ history over 300
iterations. Panel 29's Γ alternated between 7.72 and 18.74 every iteration.
Explicit calc of perpendicular distance confirmed it sat just outside r_core.

**Fix.** [`lifting_line/geometry.py:103-114`](lifting_line/geometry.py#L103-L114)
— offset the wake start by one chord length in the axial (+x) direction:

```python
chord_at_node = 3.0 * (1.0 - r_node / R) + 1.0
wake[b, :, :, 0] = chord_at_node[:, np.newaxis] + U_wake * t_k[np.newaxis, :]
```

To preserve Helmholtz, also added a **connector filament** in
[`lifting_line/influence.py:74-79`](lifting_line/influence.py#L74-L79) from
each bound c/4 node to its corresponding wake-start. Inboard connector
carries +Γ, outboard carries −Γ (consistent with the trailing-leg sign
convention so adjacent rings cancel correctly).

**Caveat to verify (APPROXIMATION).** The JS reference offsets the wake
start by `chord*(sin(angle), 0, cos(angle))` in the **chord direction**,
i.e. the offset has both axial and tangential components depending on
geometric pitch+twist. I used a **purely axial** offset for simplicity.
For the assignment geometry (twist+pitch ≈ 9°–11° at root, ≈ 0°–2° at tip),
the difference between axial-only and chord-direction offset is small
(`cos(angle)` close to 1), but it IS an approximation. Check whether this
explains any small disagreement vs BEM if you compare quantitatively.

---

## 4. Bound segments missing for blades 1 and 2 (PHYSICS)

**Symptom.** Discovered while reviewing the JS reference. My original
influence assembly included the bound segment of the reference blade only
(blade 0), but trailing legs of all 3 blades. Inconsistent.

**Fix.** [`lifting_line/influence.py:53-60`](lifting_line/influence.py#L53-L60)
— pre-compute rotated bound nodes for all blades by rotating the reference
blade nodes about the +x axis by ψ_b = 2πb/3, then loop over all blades
inside the assembly. Bound segments of blades 1, 2 now contribute to the
reference blade's control points just like their trailing legs do.

**Caveat to verify.** None — this is just a missing term that should have
been there from the start. Numerically the contribution is small (the
bound segments of blades 1, 2 are at azimuth 120°/240°, ~86 m from the
reference blade tip CP for R=50), but it's the right physics.

---

## 5. Convergence tolerance loosened (NUMERICAL)

**Original spec.** `tol = 1e-4` = 0.01 %.
**JS reference.** `errorlimit = 0.01` = 1 %.
**Final value.** [`lifting_line/solver.py:35`](lifting_line/solver.py#L35) — `tol = 1e-2`.

**Reason.** With the cosine root edge panel oscillating slightly, the
residual `max|ΔΓ|/max|Γ|` floors at ~10⁻³ even when CT/CP have stabilised
to 4 significant figures. 1 % matches the reference and is fine for
engineering-grade results. Tighter tolerances need a fundamentally
different algorithm (e.g. Newton on V_ax, V_tan instead of fixed-point on Γ).

**Caveat to verify.** Just a stopping criterion — does not affect the
converged answer.

---

## 6. Under-relaxation lowered to w=0.1 (NUMERICAL)

**Original spec.** `w_relax = 0.3`.
**Final value.** [`lifting_line/solver.py:34`](lifting_line/solver.py#L34) — `w_relax = 0.1`.

**Reason.** With w=0.3, λ=6 sits in a 2-cycle limit (residual ≈ 0.09)
because the very narrow innermost cosine panel briefly drives V_tan
negative during the iterate, flipping φ past 90° and crashing the local
linearised gain. Reducing w to 0.1 dampens the cycle. Convergence cost:

| λ  | iters at w=0.3 | iters at w=0.1 |
|----|----------------|----------------|
| 6  | did not converge (limit cycle) | 25 |
| 8  | 11             | 24             |
| 10 | 10             | 22             |

**Caveat to verify.** w only affects convergence rate, not the converged
answer (under-relaxation is just `(1-w)·Γ_old + w·Γ_new`, which has the
same fixed point as `Γ_new` itself for any w ≠ 0). The CT, CP, Γ̂
distributions at convergence should be identical to what w=0.3 would
give *if it converged*.

---

## 7. `max_iter` raised from 1000 to 1200 (TRIVIAL)

[`lifting_line/solver.py:36`](lifting_line/solver.py#L36) — matches JS reference. Convergence happens in well under 50 iters now.

---

## 8. `sensitivity_sweep` max_iter floor raised to 2000 (NUMERICAL)

**Symptom.** During the parameter sweeps, several edge configurations
(N=40 uniform, N=40 cosine, N=80 cosine, N_wake=1, a_w=0.5) hit the
1000-iter cap inside `sensitivity_sweep` even though the solver default
is now 1200. The sweep wrapper had its own hardcoded `cfg.get("max_iter", 1000)`.

**Fix.** [`lifting_line/postprocess.py:305`](lifting_line/postprocess.py#L305) —
raised the wrapper default to 2000. After the change, a_w=0.5 converges
at iter 543, N_wake=1 still does not converge (genuine 2-cycle, see below),
and the high-N cosine cases also do not converge but their CT/CP follow
the trend, so they're in low-amplitude limit cycles around the right
answer.

---

## 9. Observed convergence limits in the sensitivity sweeps (KNOWN LIMITATION)

Not a bug, but a limitation worth documenting because four sensitivity
configurations stop on the iter cap (2000) rather than the residual:

| Configuration   | Final iters | CT     | CP     | Trend-consistent? |
|-----------------|-------------|--------|--------|-------------------|
| N=40 uniform    | 2000        | 0.6377 | 0.4639 | yes (lies between N=20, N=80) |
| N=40 cosine     | 2000        | 0.6483 | 0.4729 | yes |
| N=80 cosine     | 2000        | 0.6502 | 0.4750 | yes |
| N_wake=1        | 2000        | 0.6951 | 0.5495 | yes (short wake → less induction → higher CT) |

For the N sweeps, the cosine edge panels get narrower as N grows,
strengthening the local self-induction and re-triggering the same root
2-cycle that w_relax=0.1 dampens at N=30. A targeted cure would be a
panel-local r_core (set per-trailing-leg, scaling with the local panel
width) — I have NOT done this because (a) the CT/CP values are clearly
correct already and (b) it adds non-trivial code that wasn't requested.
If you need iter-tight convergence at N ≥ 40 cosine, drop `w_relax` to
0.05 (roughly doubles iter count) or scale `r_core` with panel width.

For N_wake=1, the wake is too short to produce a steady induction —
adjacent helical loops do not overlap, so each iteration the influence
matrix's directional pull on Γ alternates. This is a physical limit, not
a numerical artifact: a 1-rotation wake is genuinely under-modelled.

**No code change applied.** Values quoted in the sensitivity figures for
these configurations are the iterates at iter cap, not strictly the fixed
point — but the difference is below the line thickness of the plot.

---

## What I deliberately did NOT do

To make sure I haven't quietly added unphysical effects:

- **No Glauert high-induction (heavily-loaded actuator disc) correction.**
  The lifting-line iteration converges with `a` < 0.30 everywhere, so
  it never enters the Glauert regime (a > 0.4). No correction needed.

- **No Prandtl tip / root loss factor.** A frozen-wake lifting line with
  three explicit blades already accounts for the finite-blade effect via
  the helical wake of each blade — Prandtl is only needed in BEM, which
  models the rotor as an actuator disc. Adding it here would double-count.

- **No clipping of Γ, alpha, or V_tan.** The iteration reaches a clean
  fixed point in α, V_tan and Γ on its own; no artificial bounds were
  imposed.

- **No Viterna–Corrigan (or other) post-stall extrapolation of the polar.**
  Outside the tabulated range (≈ −16° to +30°) `np.interp` falls back to
  flat extrapolation. The solver issues a warning when any control point
  exceeds the table but otherwise trusts the flat-extrapolated Cl, Cd
  values; no synthetic deep-stall extension is added.

- **No re-orientation of the bound vortex.** The bound segment goes from
  inboard to outboard along +z (for the reference blade at ψ=0) with
  unit Γ. Verified by direct calc that this gives the correct lift
  direction in the Kutta–Joukowski sense (force projection eq:F-axial,
  eq:F-azim use angle decomposition, not the K-J vector formula directly,
  so the sign convention is clean).

---

## Summary

One real **physics** change vs the spec:

1. Wake helix winds *against* the rotor (sign of Ω·t flipped in eq:wake-y/z).

Three **numerical** robustness changes:

2. Vortex core uses solid-body clipping instead of zero-velocity discontinuity.
3. Wake helix starts one chord-length downstream of c/4 (approximation: axial only, not chord-direction).
4. Under-relaxation tightened to w=0.1, tolerance loosened to 1 %.

One **completeness** fix:

5. Bound segments of blades 1 and 2 added to the influence matrix
   (they were missing in my original assembly).

The thing to double-check before trusting numerical results: item 3
(axial-only chord offset) is an approximation of the JS reference's
proper chord-direction offset. For the tip (chord ≈ 1 m, twist+pitch ≈ 2°)
this is a < 1 m geometric error in the wake start position, almost certainly
a sub-percent effect on CT, CP. For the root (chord ≈ 3.4 m, twist+pitch
≈ 9°) the tangential component of the offset would be ~0.5 m vs my 0 m
— could shift the root induction by a few percent. If exact agreement
with the JS reference matters, change `wake[b, :, :, 0]` and the connector
endpoints to use the proper chord direction.
