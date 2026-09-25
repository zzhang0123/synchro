# Changelog

## 0.3.0 (2026-09-25)

The package is renamed SyncMoments (`pip install syncmoments`,
`import syncmoments`), and this release improves the numerics, dependency
robustness and model options of `syncmoments.model`. Every basis column of
0.2.0 is reproduced within 1e-12 of the largest column of its Stokes block
(worst measured 6.8e-14).

### Renamed

- The distribution and import name change from `synchro` to `syncmoments`;
  the display name is SyncMoments. `synchro` on PyPI is an unrelated project,
  so `importlib.metadata.version("synchro")` could report the wrong package.
- There is no compatibility alias. Replace `import synchro` and
  `from synchro...` with `syncmoments`, and set `SYNCMOMENTS_RUN_SLOW=1`
  instead of `SYNCHRO_RUN_SLOW=1` to run the slow tests.
- The repository moves to `github.com/zzhang0123/syncmoments` and the
  documentation to `syncmoments.readthedocs.io`. The v0.2.0 tag keeps the old
  name.

### Dependency robustness

- Population weights are normalised without a subnormal reciprocal. On
  JAX 0.10.2 (CPU) `x / m` compiles as `x * (1 / m)`, so `5e307 / 1e308`
  evaluated to 0 and weights with a maximum near 1e308 gave wrong averages.
  The shared helper (`syncmoments.rm._relative_weights`, used by `rm`,
  `faraday`, `expansion.mixed_moments`, the `syncmoments.model` sample moments
  and the harmonic tail probe) reads an exact power of two from the bit
  pattern of the weights' own float format and divides by the rescaled
  maximum, which lies in `[1, 2)`; the arithmetic is float64. The result is
  `w / max(w)`, bit-identical to 0.2.0 wherever the 0.2.0 expression stayed
  in the normal range, for largest weights from `2^-1022` (about 2.2e-308)
  to 1.79e308, eager and under `jax.jit`. The derivative is `dw / max(w)`
  as before (a `custom_jvp`; the Hessian is tested).
- A largest weight below `2^-1022` (subnormal) raises an error that names
  the threshold, eagerly, under `jax.jit` and under `grad`, `jacfwd`,
  `jvp` and `hessian`; 0.2.0 also refused it. Negative weights are refused
  by their sign bit, so `-5e-324` is refused eagerly and under `jax.jit`,
  where flush-to-zero would otherwise read it as 0.
- On XLA CPU a weight whose ratio to the largest weight is below `2^-1022`
  is flushed to zero, which drops at most `n 2^-1022` of the normalised
  mass for `n` weights. Only ratios below `2^-2044` are zero on every
  backend.
- The normalised weights of `rm`, `faraday`, `expansion.mixed_moments` and
  `syncmoments.model` come from one helper, `syncmoments.rm._normalised_weights`.
  It applies an exact factor `2^-s` to primal and tangent when
  `max(w) < 2^-958`, so forward-mode tangents stay finite down to
  `max(w) = 2^-1022`. A unit tangent on every weight at `2^-1022` gives an
  exact zero tangent of `mixed_moments`; 0.2.0 gave `-inf` and NaN there.
- Second derivatives with respect to the weights are not refused.
  `w / sum(w)` is a custom JVP (`rm._divide_by_sum`). Its tangent,
  `rm._sum_tangent`, repeats the operations of the plain division's JVP, so
  values and first derivatives are bit-identical to the previous 0.3.0 state
  (1993 arrays compared in both environments: primal, `jvp`, `vjp`,
  `jacfwd`, `jacrev`, RM variance and `mixed_moments` Jacobians, eager, jit
  and vmap). The JVP of `_sum_tangent` is the closed-form second tangent
  `d2p[dr, y] = -((dr - p dR) Y + (y - p Y) dR) / R^2` with `R = sum(r)`,
  `dR = sum(dr)`, `Y = sum(y)`. It is evaluated on `dr 2^-k` (below 1) and
  `R 2^s` (in `[1, n]`), summed, and scaled once by the exact `2^(k + 2s)`,
  so an entry beyond float64 overflows to inf with its sign instead of
  forming `inf - inf`. For one weight every derivative is exactly 0. This
  rule serves forward-mode inner derivatives (`jacfwd` over `jacfwd`,
  `jacrev` over `jacfwd`, vmapped); `jax.hessian` and reverse-over-reverse
  transpose the division JVP.
- Against an exact `Fraction` oracle (`tests/test_weight_normalisation_hessian.py`,
  `tests/test_weight_normalisation_second_order.py`; "exact" means within
  the float64 rounding bound `64 eps sum|terms|`):
  - Hessians in log-weights (`w = exp(theta)`) equal the softmax Hessian
    within 5.2e-16 relative in `jax.hessian`, `jacfwd` over `jacfwd`,
    `jacrev` over `jacrev`, `jacrev` over `jacfwd` and Hessian-vector
    products, eager and jit, at shifts 0, -300, -690, -700 and -708 of
    `theta` (largest weights down to about 1e-307).
  - Hessians in `w` with `max(w) >= 2^-958` are exact, or inf with the exact
    sign where the exact entry exceeds float64, at `max(w)` of 1, 1e-100,
    `2^-511`, 1e-160 and 1e-200 in `jax.hessian`, `jacrev` over `jacrev`,
    `jacrev` over `jacfwd` and Hessian-vector products. `jacfwd` over
    `jacfwd` is exact while `sum(w) >= 2^-512`, except for the RM-variance
    case under "Known limitations"; the second tangent of `w / W` itself is
    exact or a signed inf down to `2^-1022`.
  - Hessian-vector products along small directions at `max(w)` of 1e-300 and
    `2^-1022` are exact in forward-over-reverse, reverse-over-forward and
    forward-over-forward.
  - Third derivatives match the plain division within 1e-14 relative.
- The previous 0.3.0 state refused every second derivative in the weights
  for `max(w) < 2^-958` ("weights: second derivatives with respect to the
  weights are refused ..."). That refusal also rejected representable
  results: log-weight Hessians, `n = 1` (exact Hessian 0) and small-direction
  Hessian-vector products. The refusal and its message are removed; the
  cases the closed form does not cover are listed under "Known limitations".
  `jacfwd` over `jacfwd` of `mixed_moments` and the RM variance at
  `max(w)` 1e-200 and 1e-160 (`w = [1, 2, 3, 4] max(w)/4`) is still NaN in
  15 to 16 of 16 entries, as before: the second tangent of `w / W` is now
  signed inf there, and the consumer's sum over it is NaN. 0.2.0 returned
  mostly inf in the headroom region. Because `w / sum(w)` is
  scale-invariant, the Hessian at `w` is `lambda^2` times the Hessian at
  `lambda w`, so rescaled weights give it exactly.
- `mixed_moments` refuses invalid weights with the shared message
  ("weights must be finite, nonnegative and have positive total mass";
  0.2.0 said "positive mass"). With JAX 0.10.2 its value under an outer
  `jax.jit` now equals its eager value; the previous in-line normalisation
  differed from its own eager result by 1 to 2 ulp at some inputs.
- Eager refusals of invalid weights and of non-finite samples raise
  `equinox.EquinoxRuntimeError`, as in 0.2.0, and print nothing to stderr
  (the validation helpers are compiled with `equinox.filter_jit`; tested
  through nine entry points for weights and six for samples). Under an
  outer `jax.jit` the error arrives wrapped in a `JaxRuntimeError`, as for
  every `equinox` check.
- float16, bfloat16 and float32 weights are normalised in float64 (matching
  a float64 oracle to 4e-16 in the tests). The bits are read in the
  weights' own format, so float32 subnormal weights are not flushed by a
  conversion to float64; samples keep their dtype. Other float
  widths (float8) raise `ValueError`.
- `syncmoments.transfer.transfer_slab` scales the source column by an exact
  power of two `2^k`, `k` the binary exponent of `max|eps ds|` clipped to
  `[0, 1022]`, instead of dividing by `max(1, max|eps ds|)`. On JAX 0.10.2
  the division returned an all-zero slab for `max|eps ds| > 2^1022`. Values
  and reverse-mode derivatives are correct to `max|eps ds| = 1.79e308`
  (tests at 4e307, 1e308, 1.79e308). For `max|eps ds| < 1` the result is
  bit-identical to 0.2.0; on `[1, 2)` the scale is 2 where 0.2.0 divided by
  `max|eps ds|`, so values there can differ by rounding. Derivatives are
  described under "Transfer derivatives and refusals".
- `PopulationSamples.product` normalises each marginal through the same
  helper (its `w / sum(w)` overflowed at 1.79e308); invalid marginal weights
  now raise at construction. It validates and normalises all six marginals
  in one compiled call, and the `PopulationSamples` constructor runs its
  value checks (finite values, `gamma >= 1`, `B >= 0`, `|mu|, |eta| <= 1`,
  weights) in one compiled call. The product weights are bit-identical to
  per-marginal normalisation. When several fields are invalid at once, the
  message reported can differ from 0.2.0, which checked the fields one
  after another; a single invalid field gives the 0.2.0 message.
- The tests no longer import `mpmath`. `scripts/reference_constants.py`
  (mpmath 1.3.0, 40 digits) generates the Bessel, continuum `F`/`G` and
  bump-area references that the tests hold as literals; `--check`
  recomputes them at 50 digits (largest difference 3.5e-32).
- Two tests whose tolerance was below float64 roundoff now derive it: the
  `required_m_max +/- 1` boundary test uses the summation error
  `2 gamma_K sum |x_i|` with `K = n_nodes + m_max` (JAX 0.10.2 gave a gap of
  3.85e-34 against a tolerance of 4.10e-32), and the fit-integration
  covariance test uses a norm-wise tolerance proportional to
  `cond(X) eps` against two pseudo-inverse oracles (`cond(F) = 4.2e8` in the
  Gaussian-screen case).

### Transfer derivatives and refusals

- `transfer_slab` differentiates through a custom JVP rule. The slab is
  linear in its inputs, `out = Phi S + G eps` with `Phi = e^{-K ds}` and
  `G = int_0^ds e^{-K t} dt`, and the tangent is
  `Phi dS + G deps + dPhi S + dG eps`, with `Phi`, `G` and their
  derivatives in `(K, ds)` taken from one 8x8 exponential
  `exp([[-K ds, I], [0, 0]])`. No tangent passes through the scaled source
  column. The value is unchanged (bit-identical in float64 and float32).
  The 8x8 exponential is `syncmoments._expm.expm_pade13` (below); the
  value's 5x5 exponential is still `jax.scipy.linalg.expm`.
- Without the rule (0.2.0, and 0.3.0 before it), forward-mode Jacobians
  with respect to `eps` were finite but wrong from about
  `max|eps ds| ~ 1e297`, depending on `K` and `ds`, and zero near 1e307.
  For an optically thick non-diagonal `K` the pre-rule 0.3.0 code had
  relative errors of 1e-4 at 1e303 and 0.1 at 1e306; 0.2.0 had errors of
  the same order (2e-2 where the pre-rule code had 4e-2, near 2e306). With
  the rule, forward and reverse Jacobians with respect to `S` and `eps`
  agree with SciPy to 1e-13 of their largest entry from 1e297 to 1e307 for
  non-diagonal `K`, and to 1.79e308 for `K = a I`, `a` in {0, 2, 50};
  `jax.hessian` is finite and symmetric, and its `eps`-`K` block matches
  SciPy's `expm_frechet`.
- Second derivatives that differentiate the value computation are
  accurate at extreme source scale. JAX takes them when an outer derivative acts on
  the primal output of `jvp`, `vjp` or `linearize` (for example two chained
  slabs that both depend on `K`) and inside `lax.scan` under `vjp`, whose
  partial evaluation inlines the primal. The JVP rule computes its primal
  by re-entering itself, and at extreme scale the value carries the
  derivatives of `lin = Phi S + G eps` from one extra 8x8 exponential:
  `stop_gradient(value) - (stop_gradient(lin) - lin)`, bit-identical to the
  value (`-0.0` included). Before this fix these nestings differentiated
  the scaled 5x5 exponential, as 0.2.0 did: at `max|eps ds| = 1e306` the
  relative error was 0.115 for `jacfwd` of a `jvp` primal (thick `K`,
  `ds = 20`), 1.7e-2 for forward-over-reverse and forward-over-forward
  through two chained slabs (weak `K`, `ds = 1`), whose mixed Hessian block
  was therefore asymmetric, and 3.3e-2 for forward-over-reverse through
  `transfer_los`. Now every tested nesting agrees with plain autodiff of
  `Phi S + G eps` to at most 6.1e-16, and the Hessians of two chained
  slabs and of a 3-slab `transfer_los` agree and are symmetric in every
  second-order nesting (`tests/test_transfer_higher_order.py`).
- The extra exponential about doubles the cost of the primal (1.9 to 2.1x
  for jit `transfer_los`, 1.4x under `jit(vmap)`), so it is used only from
  a threshold. The exposure is
  `k + max(0, -e(ds)) + max(0, e(max|K ds|))`, with `e` the binary exponent
  and `2^k` the source scale; the value carries the extra derivatives
  when the exposure reaches `T = maxexp - 128` (896 in float64; 0 in float32,
  so float32 always uses them). On a grid of `|K|` from 1e-6 to 1e6, `ds`
  from 1e-9 to 1e6 and `|K ds| <= 2e4` the plain method's derivative error
  first exceeded 1e-13 at exposures 950 to 1014 and grew about twofold per
  unit, so the threshold leaves at least 54 binary orders of margin on
  that grid; outside it the margin is not measured. At exposures `T - 8`,
  `T - 1`, `T`, `T + 1` and `T + 8` both methods agree with each other and
  with the reference to 1e-13 (`tests/test_transfer_companion_boundaries.py`).
  `transfer_slab` chooses per slab with a `lax.cond`; `transfer_los`
  chooses once per path (any slab at or above the threshold), because a
  per-slab `cond` made `jit(grad)` of the scan 1.4x slower. Under `vmap`
  the choice is taken over the whole batch, so one extreme ray, or one
  extreme slab of a path, puts every member on the extra exponential. A
  user-written `lax.scan` over `transfer_slab` pays the per-slab `cond`.
  The float32 cost was not measured.
- Reverse-mode Jacobians with respect to `K` and `ds` are finite near
  `max|eps ds| = 1.7e308`. The `(K, ds)` tangent `dPhi S + dG eps` is
  formed as `((dPhi 2^a) (S 2^-j) + (dG 2^a) (eps 2^-j)) 2^(j - a)`, with
  `2^j` the power of two of `max(|S|, |eps ds|)` above `2^512` and
  `a = j // 2`, so neither the forward pass nor its transpose forms
  `dG eps` or the cotangent `ct eps` at full scale. Before, `jacrev` in `K`
  and in `ds` returned 16 to 32 non-finite entries at 1.7e308 (thick `K`
  with `ds` 1 and 20, weak `K` with `ds = 20`), where the true entries are
  below 3.5e305; `jacfwd` was right. Now both agree with SciPy
  `expm_frechet` to 1.8e-16 to 2.8e-15 there
  (`tests/test_transfer_extreme_jacobians.py`). Below `2^512` every factor
  is 1 and the gradients are bit-identical to the unsplit rule.
- Reverse mode through `vmap` over a batched `K` or `ds` works: `grad`,
  `vjp`, `jacrev`, `hessian` and Hessian-vector products of a sum over
  `vmap`, or over nested `vmap`, of `transfer_slab`, `transfer_los` and
  `moment_driven_slab(_cgs)`, eager and jit. The previous 0.3.0 state
  raised `NotImplementedError: Transpose rule (for reverse-mode
  differentiation) for 'stop_gradient' not implemented` there (0.2.0, which
  had no rule, did not). The rule differentiates the 8x8 exponential with
  `jax.jvp`, and `jax.scipy.linalg.expm` chooses its Pade degree with
  `lax.switch` and each squaring with `lax.cond` on the operand's values;
  under a batched `vmap` both become selects that wrap the tangents in
  `stop_gradient`, which cannot be transposed. `expm_pade13` always uses
  Pade 13. Each squaring step runs under a `lax.cond` on whether any member
  of the batch needs it (`_expm.any_member`, a `custom_vmap` that reduces
  the batch axis at every `vmap` level, so the `cond` stays a branch) and
  is applied per member (`_expm.batched_only`, which folds away when
  unbatched). `grad` of a sum over `vmap` equals `vmap` of `grad`. For the
  tested batched slabs, paths and `moment_driven_slab` spectrum, the
  gradients and Hessians agree with 0.2.0 (computed in a separate process)
  within 1.1e-14 of the largest entry (`tests/test_transfer_batched_reverse.py`,
  `tests/test_ad_transform_matrix_nested.py`); rotation-dominated slabs
  differ more, see "Changed numbers" below.
- `expm_pade13` takes `n = ceil(log2(|A|_1 / theta_13))` squarings, where
  `jax.scipy.linalg.expm` takes `floor`, and falls back to `floor` when
  `ceil` exceeds `max_squarings`, so it returns NaN exactly where `expm`
  does. Boundary validation of the two squaring counts, evaluated directly
  at `theta_13 2^m (1 - 1e-9)`, `theta_13 2^m` and `theta_13 2^m (1 + 1e-9)`
  for `m` in {0, 1, 3, 10, 20, 31}, found that `floor` applies Pade 13 up to
  `2 theta_13`: for a rotation-dominated matrix just below
  `2^(m+1) theta_13` its relative error against SciPy is 1e-10 at `m = 0`,
  1.2e-7 at `m = 10`, 1.2e-4 at `m = 20` and 0.13 at `m = 31`, while `n` and
  `n + 1` squarings with `ceil` agree with each other and with SciPy within
  `1e-14 2^m`, the conditioning bound (`tests/test_transfer_expm_boundaries.py`).
- Changed numbers (derivatives only; values are bit-identical): on 120
  random slabs the Jacobians in `eps` change by up to 1.5e-13 and in `K` by
  up to 1.9e-10 relative against 0.2.0, toward the exact values (against
  mpmath at 40 digits the worst `K` entry now errs by 1.3e-14, against
  1.9e-10 in 0.2.0). The change is larger where the exponential needs many
  squarings, in rotation-dominated slabs with `|K ds|` from about 80 to
  2e4: for `moment_driven_slab_cgs` with `n_e = 0.03 cm^-3`,
  `B_par = 3 uG`, `L = 1 kpc` and 50 MHz to 20 GHz, per-frequency Jacobian
  entries change by up to 5e-6 relative (a chi^2 gradient by 9.6e-8, its
  Hessian by 2.6e-8); against mpmath the new derivatives err by at most
  9e-13, the 0.2.0 ones by up to 5e-6. `d out / d ds` changes by up to O(1)
  relative only in ill-conditioned slabs, where every version differs from
  mpmath by that much (see "Known limitations").
- The propagators' Pade-13 coefficients are held at the exact scale
  `2^-40`. With the textbook `b_0 = 6.5e16` the transpose of the solve
  divided cotangents by `b_0`, so float32 reverse-mode derivatives in `K`
  became zero for sources below about 1e-15 (`ds = 20`) or 1e-22 (`ds = 1`);
  they now match forward mode to 1e-6 down to 1e-20 and 1e-25, and float64
  reverse mode is exact down to sources of 1e-290
  (`tests/test_transfer_scale_and_budget.py`). Values and JVPs are
  unchanged.
- `max_squarings` accepts numpy integers and integral floats again (the
  compiled cores need a Python int); a non-integral or negative value raises
  `TypeError` or `ValueError`.
- An AD-transform matrix (`tests/test_ad_transform_matrix.py`) covers 19
  entry-point families that pass through a `custom_jvp`, `custom_vmap` or
  `filter_jit`: `transfer_slab`, `transfer_los`,
  `moment_driven_slab(_cgs)`, `gaussian_rm_cumulants`,
  `screen_polarisation`, `mixed_moments`, `faraday.emission_polarisation`,
  `faraday.joint_faraday_moments`, `PopulationSamples.normalised_weights`
  and `.product`, `JointMoments.from_samples`, `EmpiricalScreen`,
  `model.harmonic.harmonic_lines`, and the private Bessel-recurrence rules
  `_bessel_recurrence.bessel_band` and `neighbours_recurrence`. Its 63
  argument groups each run 27 transforms (eager, jit, `vmap`, nested
  `vmap`, `grad` of a sum over `vmap` and over nested `vmap`, `jacfwd`,
  `jacrev`, `hessian` and the other second-order nestings, `linearize`,
  `vjp`, Hessian-vector products, `checkpoint`, `scan` and others). Each of
  the 1701 cells is compared with jitted forward-mode autodiff of an
  independent plain-`jnp` reference within 1e-12 of the largest entry per
  block, and must be finite. The matrix found that `grad` of a sum over
  nested `vmap` of the transfer functions still raised the `stop_gradient`
  error in 17 argument groups: the `custom_vmap` rule of `any_member`
  reduced its batch axis with a plain `jnp.any`, which stayed batched under
  an enclosing `vmap`. The rules of `any_member` and `batched_only` now
  re-enter themselves, and all 1701 cells pass with JAX 0.10.0 and 0.10.2.
  Projections of the references' values, Jacobians and Hessians and of the
  package's values and Jacobians are pinned to 0.2.0 within 1e-12
  (`tests/test_ad_transform_matrix_v020.py`; the two JAX versions agree to
  2.1e-14). The matrix covers the normal floating-point range; the
  extreme-scale corners stay with the `test_transfer_*` and
  `test_weight_normalisation_*` files.
- A derivative whose own value exceeds float64 overflows to inf without an
  error, because only the value is checked; this needs `eps` within a few
  orders of 1.8e308. Derivatives beyond second order were not tested at
  extreme scale.
- Cost of jit-compiled derivatives of `transfer_los` over 512 slabs, before
  and after the rule, with `jax.scipy.linalg.expm` for the propagators (JAX
  0.10.0, median of 7 runs, shared machine; not remeasured with
  `expm_pade13`):
  `jacfwd` in `(eps, K)` 1.28 -> 3.35 s and `jacrev` 7.8 -> 16.8 ms, from
  the Frechet derivative of the 8x8 exponential; in `eps` alone `jacfwd`
  421 -> 16 ms and `jacrev` 8.3 -> 4.4 ms, because the rule skips the
  matrix derivative when `dK = dds = 0`. The primal is unchanged
  (1.7 to 1.9 ms). JAX 0.10.2 gives the same pattern.
- Cost of the value-derivative fixes against the rule alone (jit, 64
  slabs unless stated, serial runs at load averages 9 to 18, JAX 0.10.0 /
  0.10.2): `transfer_los` primal 0.99 / 0.98x (512 slabs 1.03 / 1.02x),
  `jit(vmap)` over 256 rays 0.99 / 1.00x, `grad` in `eps` 0.97 / 0.99x,
  `grad` in `K` 1.17 / 1.28x (1.13 to 1.15x in earlier runs at lower load;
  the cause was not isolated). Values and these gradients are
  bit-identical in the normal range. With the extra exponential active
  (sources near 1e300) the primal takes 1.96 / 2.03x. Compile time at 64
  slabs rose from 0.46 to 0.74 s for `grad` and from 0.14 to 0.23 s for
  `hessian`, because `transfer_los` compiles two scans.
- Cost of `expm_pade13` and `filter_jit`, jit-compiled `transfer_los` over 64
  slabs unless stated, ms, best of 9 blocks in separate processes, two
  runs with JAX 0.10.0 at load averages 4.7 to 5.9 (JAX 0.10.2 gives the
  same pattern):

  | | previous 0.3.0 | 0.3.0 | 0.2.0 |
  |---|---|---|---|
  | primal | 0.276-0.281 | 0.271-0.284 | 0.265-0.281 |
  | primal, 512 slabs | 2.20-2.23 | 2.23-2.28 | 2.04-2.08 |
  | `jit(vmap)`, 256 rays | 114.6-116.3 | 116.5-120.8 | 112.2-118.5 |
  | `grad` in `eps` | 0.56-0.57 | 0.55-0.61 | 0.98-1.01 |
  | `grad` in `K` | 1.84-1.98 | 1.78-1.95 | 1.00-1.02 |
  | `jvp` in `K` | 1.21-1.31 | 1.21-1.28 | 0.58-0.60 |
  | `vmap(grad)` in `K`, 16 rays | 30.4-31.6 | 17.8-18.3 | 26.4-27.4 |
  | `grad` of a sum over `vmap` in `K`, 16 rays | raises | 16.2-18.1 | 31.6-32.8 |
  | `hessian`, 3 slabs (60 inputs) | 1.65-1.71 | 1.75-1.89 | 0.82-0.86 |

  The derivatives in `K` cost about 1.8 to 2.2 times 0.2.0's, from the
  linear tangent rule. The batched reverse modes are faster than before,
  because under `vmap` `expm` evaluated all five Pade branches and all 32
  squarings as selects, and `expm_pade13` does not. Compile
  times were not measured.
- Behaviour change: `transfer_slab` and `transfer_los` refuse a non-finite
  result with an `equinox` error that names the cause: non-finite inputs,
  `eps * ds` beyond the float64 range, an exponential that needs more than
  `max_squarings` squarings, or a value beyond float64. The last reads
  "the result overflowed float64 (accumulated intensity S e^{-K ds} + G eps
  beyond the range, or extreme gain: negative absorption, e^{-K ds} beyond
  the floating-point range)"; an earlier 0.3.0 state blamed extreme gain
  also when an absorbing slab overflowed by accumulation. The sign of the
  absorption is not tested, so the value needs no eigenvalue computation.
  The check runs eagerly, under `jax.jit`, `vmap`,
  `jacfwd`, `jacrev` and inside the scan of `transfer_los`; 0.2.0 returned
  NaN or inf without an error. `moment_driven_slab` and
  `moment_driven_slab_cgs` inherit the refusal. Eager calls, including
  the scan of `transfer_los`, eager `grad` and eager `jacrev`, raise
  `EquinoxRuntimeError` and print nothing to stderr; under an outer
  `jax.jit` the error arrives wrapped in a `JaxRuntimeError` and a callback
  traceback is printed to stderr. The previous 0.3.0 state raised a
  `JaxRuntimeError` from eager `transfer_los`, with a traceback on stderr.
  Finite limit cases
  (`max|eps ds| = 1.79e308`, optical depth 1e6, gain `K = -700 I`) are not
  refused.
- `transfer_slab` and `transfer_los` compile their cores with
  `equinox.filter_jit` (once per shape, `max_squarings` static; shape
  checks stay outside). The previous 0.3.0 state built new `lax.cond`
  branch closures on every eager call and compiled one `cond` per call.
  Eager cost per call, JAX 0.10.0 and 0.10.2, ms (previous 0.3.0 / 0.3.0 /
  0.2.0): `transfer_slab` 94-103 / 0.054-0.062 / 0.48-0.51, `grad` in `K`
  of one slab 100-110 / 1.15-1.34 / 3.5-3.9, `transfer_los` over 50 slabs
  206-224 / 0.29-0.32 / 96-102, `moment_driven_slab` 108-118 / 11-13.5 /
  11-13.6 (most of the remaining time is its eager `kirchhoff`
  operations). A second eager
  call with the same shapes compiles nothing (`tests/test_transfer_eager.py`).

### Memory and speed

- `HarmonicKernel(chunk_budget=CHUNK_BUDGET)` with
  `syncmoments.model.harmonic.CHUNK_BUDGET = 2**20` (the private 0.2.0 value was
  `2**26`): one `lax.map` step of `angular_projection`, `angular_taylor`
  and the harmonic tail probe holds at most `max(chunk_budget, n_nodes)`
  primal Bessel integrand values; one angular point of one harmonic is the
  floor. Angular product cells and tensor nodes are summed in checkpointed
  point blocks; `m_chunk` (default 64) is now an upper cap on harmonics per
  step. Blocking changes only the summation order. The budget counts
  primal values; forward tangents of an outer `jacfwd` multiply the working
  set unless the caller divides the budget with `for_tangents` (the basis
  build does).
- `channel_modes` plans for one point, so a caller that vmaps it over `S`
  samples holds `S` times that. `direct_channel_average` now caps its
  `batch_size` by the kernel's `samples_per_step(channels, requested)` (102
  samples for `HarmonicKernel(40)` at the default budget); `batch_size` is
  an upper cap. For the benchmark channels with `S = 4096` and
  `batch_size = 1024`, peak RSS fell from 0.72 to 0.54 GB (JAX 0.10.0) and
  from 1.23 to 0.63 GB (JAX 0.10.2), and the time rose from 1.3 to 1.9 s; a
  larger `chunk_budget` restores the old speed at the old memory.
- `ContinuumKernel(chunk_budget=CHUNK_BUDGET)` bounds the `F`/`G`
  integrand values of one `lax.map` step: `angular_projection` runs its
  `eta` nodes in blocks of `eta_block(channels)` (`2 n_ch n_nu n_nodes_F`
  values per node), and it gains `samples_per_step` and `for_tangents`. One
  `eta` node or one sample is the floor. `describe()` records
  `chunk_budget`. Blocking changes only the summation order: against the
  unblocked code the bases agree within 6.6e-16 per block and the probe
  envelopes within 7.7e-15; the refined-rule `numerical` term, a
  difference of two bases, changes by 2e-9 of itself (1e-15 of the basis
  block).
- The nested-`jacfwd` route of the remainder probe (`RemainderInputs.from_samples`
  for a kernel without `angular_taylor`, such as `ContinuumKernel`, or with
  `derivatives="autodiff"`) runs the projection with
  `kernel.for_tangents(tangent_divisor(kernel, q))` for an order-`q`
  nesting in `z_3`. The divisor is `4^q` (the primal and three tangents per
  level) for `ContinuumKernel`, `HarmonicKernel(derivatives="autodiff")`
  and unknown kernels, and 1 for the analytic `HarmonicKernel`, whose
  tangents do not pass through the Bessel quadrature (its default probe is
  `angular_taylor`; the private `method="jacfwd"` of
  `syncmoments.model._remainder_probe` reaches the nested route). The
  margin envelope builds one nesting per margin order that occurs. `angular_residual` caps its sample batch by the
  kernel's `samples_per_step` and projects with `for_tangents(batch)`. The
  refined `ContinuumKernel` of the convergence rebuild keeps the caller's
  `chunk_budget`. `scripts/model_examples_continuum.py` maps
  `incident_polarisation` over the rays in batches of
  `samples_per_step(channels, 256)` instead of vmapping over all of them.
  Only summation orders change: probe outputs agree with the unbudgeted
  code to 2.2e-15 relative, and the printed examples are unchanged apart
  from timings.
- `syncmoments.bessel.bessel_jn_neighbours(n, x)` returns `J_{n-1}`, `J_{n+1}`
  and `J_n'` from one periodic quadrature on the order-`n` contour (three
  transcendentals per node instead of nine). `harmonic_lines` uses it;
  `bessel_jn_and_prime` is unchanged. Its guard is `n + 1 + |x| <=
  n_nodes/2` with `n >= 1`; outside it the result is NaN.
- `HarmonicKernel(derivatives="analytic")` (default) computes the
  `(z_gamma, z_B)` Taylor tensors of `build_basis` with no tangent through
  the Bessel quadrature: Bessel derivatives come from the order recurrence
  `J_n' = (J_{n-1} - J_{n+1}) / 2` on one band of orders per point, and the
  Legendre contraction is assembled by the Leibniz rule. New method
  `HarmonicKernel.angular_taylor(..., order, orders=None)`.
  `derivatives="autodiff"` nests `jacfwd` through the quadrature and the
  contraction: the 0.2.0 structure, but on the shared-contour rule
  `bessel_jn_neighbours`, so it is not bit-identical to 0.2.0, and its
  Bessel derivatives lose accuracy at `|x| << m` (at `m = 1`, `x = 1e-6`
  the third derivative is off by more than its own size; the recurrence is
  accurate to about 1e-15 there). Both are the same automatic
  differentiation of the same program, node motion included, and agree
  within 1e-12 per Stokes block and derivative order at
  `gamma0 in {1.01, 2, 20, 50}` x `B0 in {1e-2, 10, 100}` G, `N = 0..3`.
  Neither is the derivative of the computed values in affine-follow cells
  (`0 < lo < 0.1 hi`) or where a channel edge meets a line at `t = 0`,
  where the rule is not differentiable; there the difference is the rule's
  quadrature error, estimated by `numerical` and not bounded.
- `Truncation(max_orders=...)` skips the derivative directions that the
  retained rows do not need (at `N = 2`: 9 forward tangents uncapped, 6 with
  `N_B = 1`, 4 with `N_B = 0`).
- `RemainderInputs.from_samples` probes derivatives with `angular_taylor`
  for an analytic `HarmonicKernel` (nested `jacfwd` otherwise). The harmonic
  tail probe sums its samples in blocks under the same budget.
- `HarmonicKernel.describe()`, and so the kernel record of every
  provenance, gains the keys `chunk_budget` and `derivatives`.

Peak RSS and `build_basis` time at benchmark size (`HarmonicKernel(40)`, the
three benchmark channels, `Truncation(8, 8, 2)`, 64 x 64 product cells,
`convergence=False`; one shared CPU machine, so times are noisy):

| | JAX 0.10.0 | JAX 0.10.2 |
|---|---|---|
| 0.2.0 | 10.9 GB, 20.0 s | 45.6 GB, 32.6 s |
| 0.3.0 | 0.82 GB, 6.5 s | 0.93 GB, 7.4 s |

Remainder probe at the same size (`RemainderInputs.from_samples`, probe
order 3). 0.2.0 with JAX 0.10.0: peak RSS 70.7 to 72.2 GB and 123 to 159 s
per probe point including compile, in two-point runs (JAX 0.10.2 not
measured). 0.3.0 default (`angular_taylor`), JAX 0.10.0 / 0.10.2: peak RSS
3.3 / 3.7 GB, 8.0 / 8.2 s per further point (13.3 s per point including
compile with JAX 0.10.0; 3.2 / 3.7 GB, 6.7 / 7.2 s in later serial runs on
an idle machine). 0.3.0 with nested `jacfwd`:

| kernel, budget divisor | JAX 0.10.0 | JAX 0.10.2 |
|---|---|---|
| analytic harmonic, 1 (default) | 2.15 GB, 22.8 s per further point | 2.02 GB, 21.9 s per further point |
| analytic harmonic, `4^q` (earlier 0.3.0 state) | 1.68 GB, 46 s per further point | 1.77 GB, 43 s per further point |
| `derivatives="autodiff"`, `4^q` (default) | 1.96 GB, 178 s for one point with compile | 2.89 GB, 206 s for one point with compile |
| `derivatives="autodiff"`, 1 | 3.97 GB, 103 s for one point with compile | 10.09 GB, 124 s for one point with compile |

The `4^q` analytic row is from serial runs on an idle machine, the others
from serial runs at load averages 9 to 30. An earlier version of this
changelog attributed the undivided analytic figures (then 2.2 / 2.0 GB) to
the `derivatives="autodiff"` kernel. All divisors give the same `H` to
3.3e-15 relative per block, and the new analytic default is bit-identical
to divisor 1.

Other measurements, 0.2.0 -> 0.3.0 (JAX 0.10.0 / 0.10.2): harmonic tail
probe over 4096 samples 2.86 -> 0.48 GB / 6.49 -> 0.56 GB; XLA temporaries
of the reverse-mode memory regression test (`jacfwd`, `grad`) 6.7, 5.8 GB
-> 33, 33 MB / 25.3, 26.0 GB -> 33, 40 MB; XLA temporaries of the nested
second-order `jacfwd` of `angular_projection` at the benchmark size
9.7 -> 0.078 GB / 44.0 -> 0.086 GB. The 0.3.0 `N = 3` build takes 2.82 GB,
32.3 s / 3.44 GB, 25.9 s, mostly XLA compile memory.

Continuum README examples, whole example with `/usr/bin/time -l`, peak RSS
and wall time (JAX 0.10.0 / 0.10.2; 0.2.0 gave 6.4 / 26.7 GB for example
(b)). "No budget" is the 0.3.0 tree before the continuum budget,
"continuum budget" adds it, and "final" also divides the probe budget and
batches the example script (the last column from serial runs on an
otherwise idle machine):

| example | no budget | continuum budget | final |
|---|---|---|---|
| (b) | 6.71 GB, 23.2 s / 27.07 GB, 19.8 s | 1.92 GB, 22.2 s / 5.06 GB, 15.0 s | 1.16 GB, 14.7 s / 1.79 GB, 14.4 s |
| (d) | 4.55 GB, 10.2 s / 12.22 GB, 11.3 s | 2.92 GB, 9.3 s / 3.13 GB, 9.2 s | 0.85 GB, 7.1 s / 1.12 GB, 7.3 s |

With the continuum budget alone the basis build of example (b) fell from
4.23 to 0.65 GB and from 11.87 to 0.88 GB, and the peaks were set by the
nested-`jacfwd` remainder probe in (b) (1.85 / 4.97 GB) and by the example
script's vmap over all rays in (d) (2.5 GB in both environments); the
final column budgets both.

### Model options

- `Truncation(L_mu, L_eta, N, depth_degree=None, *, max_orders=None)` with
  `max_orders = (N_gamma, N_B, N_depth)` (each an int or `None`) keeps only
  rows with `r <= N_gamma`, `s <= N_B`, `b <= N_depth` in addition to the
  total cutoff `[extension]`. Caps that remove no row normalise to `None`,
  so the default layout and counts are unchanged. New methods `caps()`,
  `is_capped()`, `margin_rows()`.
- The retained multi-indices of a capped truncation form a lower set
  `Lambda`. Its Taylor remainder is bounded by a sum over the telescoped
  margin `S(Lambda)` (`syncmoments.model.lower_set_margin`), derived by
  one-dimensional Taylor steps; for total degree `S` is the shell
  `|beta| = N + 1`. `S` can be larger than the set of minimal elements of the
  complement, and the minimal elements alone do not bound the remainder:
  for `Lambda = {0,1}^2` and `f = (x - x^2)(y - y^2)` at `(1, 1)` the
  remainder is 1 while the sum over `(2,0)` and `(0,2)` with the box
  supremum is 0.5.
- `RemainderInputs(..., margin_H=None, margin_moments=None)`: the
  per-multi-index inputs of the margin form. `basis_remainder` of a capped
  basis uses them; the operator-norm inputs `H`, `absolute_moments` leave it
  `unbounded` with a note. `from_samples` fills them.
- `pitch_symmetric(index)` and `field_reversal_symmetric(index)`
  `[extension]`: declared symmetries as `ParameterMap`s (one group, no
  independence assumed, discrepancy `unbounded` unless
  `discrepancy="measured"` or an allowance). Pitch symmetry
  (`mu -> -mu` at fixed other variables) drops every odd-`l` row. Field
  reversal (`B -> -B` with the electrons fixed, `(mu, eta, phi) ->
  (-mu, -eta, phi + pi)` at fixed `gamma`, `B`, depth) drops every odd
  `l + k` row, so the finite `V` response is exactly zero. The reversal is
  local: it does not flip the Faraday depth. Free parameters at
  `(2, 2, 2)`: 115, 129 and 103 for both (153 without).
- Combined with a factorisation, a symmetry is applied group by group. Each
  symmetry acts one variable at a time, so under a product measure the
  joint invariance holds exactly when every group marginal is invariant,
  and a row vanishes when any group factor has an odd sum of flipped
  exponents. With one group this is the rule above; when `mu` and `eta`
  are in different groups, field reversal drops the rows with odd `l` or
  odd `k`. `fully_independent` with field reversal has 10 free parameters
  at `(2, 2, 2)` and 16 at `(8, 8, 2)` (12 and 24 without it).
- A `delta` or `fixed_table` closure whose marginal is itself not symmetric
  is accepted together with a symmetry; only the entries the symmetry keeps
  enter, so the map represents the symmetrised marginal. Both assumptions
  are recorded; nothing checks the combination.
- A `fixed_table` combined with a symmetry takes the full table, the shape
  it has in the same partition without the symmetry, both in one
  `ParameterMap.build` with both closures and through `assume`, and the two
  routes give identical maps and bitwise-equal moments. The entries the
  symmetry drops (for example `<P_1(mu)>` on a `mu` singleton under field
  reversal with `mu` and `eta` separated) are ignored: nonzero values there
  give the same moments as zeros. The record keeps the full input table in
  `hyper` and names the dropped entries in `closure_kind`, e.g.
  `fixed_table (full table, symmetrised: dropped <P_1(mu)>)+field_reversal_symmetric`.
  A reduced table of the kept entries only is refused on both routes with
  a `ValueError` naming the full shape and the dropped entries. An earlier
  0.3.0 state required the reduced table from `build`, and `assume` could
  not combine the closure. Records of two closures whose tables differ only
  in dropped entries differ although their moments are equal, so `assume`
  treats them as conflicting. Groups where the symmetry drops no entry
  (for example `gamma`) are unchanged.
- The `fully_independent` docstring gives its free count as
  `N_gamma + N_B + max_b + L_mu + L_eta + 2` with the caps of
  `truncation.caps()` (`2N + max_b + L_mu + L_eta + 2` uncapped).
- Provenance `truncation` records and `JointMoments.to_dict()["index"]`
  include `max_orders` (`null` or `[N_gamma, N_B, N_depth]`).
- New exports from `syncmoments.model`: `lower_set_margin`, `pitch_symmetric`,
  `field_reversal_symmetric`.

### Not added

- A field-averaged kernel route (a kernel averaged over a declared
  distribution of `B`, independent of the other variables) was designed and
  not implemented. In the target populations the field strength is
  correlated with the electron energy, the Faraday depth and the field
  direction, so the independence it requires does not hold. Per-variable
  caps (`N_B`) reduce the cost of the field direction without that
  assumption.

### Known limitations (to be addressed in 0.3.1)

- Some second derivatives in the weights return NaN or infinities without
  an error. They are not refused, because primal values cannot tell them
  from the exact log-weight Hessians at the same weights:
  - For `max(w) < 2^-958`, where the normaliser applies its headroom factor,
    every entry of a Hessian in `w` along unit directions exceeds float64
    (`|H| > 1e580` for four weights). `jax.hessian`, `jacrev` over `jacfwd`,
    `jacfwd` over `jacfwd` and Hessian-vector products return NaN there.
    `jacrev` over `jacrev` returns infinities, some with the wrong sign: 10
    of 16 entries for `mixed_moments` at `max(w)` 1e-300 and `2^-1022`, and
    3 of 16 for a `JointMoments.from_samples` moment under `jax.jit`. The
    cause is the operation order of the first-order division JVP, which
    was kept bit-identical; a gauge reformulation turns these entries into
    NaN but changes first-derivative rounding with JAX 0.10.2.
  - `jacfwd` over `jacfwd` once `sum(w) < 2^-512`: the second tangent of
    `w / W` has entries `(2 p_i - delta_ia - delta_ib) / W^2` that are
    infinities of both signs, so a consumer's sum over them is NaN, also
    where the consumer's exact Hessian is representable (for example 0 for
    a constant). A fix needs a second-order rule in each consumer.
  - The RM variance under `jacfwd` over `jacfwd` for 16 weights at
    `max(w) = 2^-511`, where the exact entries are at most `2^1022`: its own
    products of first derivatives overflow (31 inf, 2 inf of the wrong sign,
    1 NaN of 256 entries; no finite wrong value).
  - A Hessian-vector product at `max(w) = 2^-1022` along a direction of size
    `2^-1033` loses accuracy, because a reverse-mode intermediate is
    subnormal and XLA CPU flushes it (as in the previous 0.3.0 state).
- The value of `transfer_slab` still uses `jax.scipy.linalg.expm`, whose
  `floor` squaring count applies Pade 13 up to `2 theta_13`. For a
  Faraday-dominated slab (`aI = 1e-3`, `rV = 1`, `rQ = 0.3`, `rU = -0.2`)
  the value's relative error against mpmath is 9.6e-12 just below
  `|K ds| = 10.74`, against 1.7e-16 just above 5.37; the `floor` method's
  error grows with the number of squarings. For the realistic
  rotation-dominated slabs of the `moment_driven_slab_cgs` example above
  it is 1.2e-10 to 4.4e-10. The value's exponential is
  unchanged since 0.2.0 and was kept so that values stay bit-identical to
  the previous 0.3.0 state; computing it with `expm_pade13` would change
  values at the 1e-11 level. Near those norms the derivatives, which come
  from `expm_pade13`, are more accurate than the value.
- `d out / d ds` of `transfer_slab` in ill-conditioned slabs, where the
  derivative is far smaller than the terms that cancel in it, is accurate
  only to the roundoff of those terms and is returned without a flag:
  relative errors against mpmath of 0.2 to 56 on 4 of 120 random slabs
  (0.2.0: 1.5e-4 to 110 on the same slabs).
- Third derivatives through `expm_pade13` were checked by probes against
  autodiff of `jax.scipy.linalg.expm` (1e-16 to 7e-14 in four nestings) but
  are not in the test suite. `expm_pade13` uses the float64 `theta_13` in
  float32 too, so its float32 squaring count differs from that of
  `jax.scipy.linalg.expm`. Compile times, and the 512-slab derivative costs
  with `expm_pade13`, were not measured.
- The AD-transform matrix leaves out the private `model.basis._build_core`
  (its derivatives are covered by `tests/model/test_b_analytic_derivatives.py`
  and `test_basis_convergence.py`) and the private Taylor-rule path of
  `_bessel_recurrence` (`derivative_coefficients`, `taylor_neighbours`). It
  covers the normal floating-point range only. Its fast tier (195 tests
  with the nested and 0.2.0 pin files) takes about 115 s; the other 1573
  tests are `slow` (about 23 min per environment). The script that
  computed the 0.2.0 pins is not in the repository; the test docstring
  states how they were computed.

### Packaging and tests

- The manuscript's benchmark reference (`validation/full_response.py`,
  `validation/full_response_product.py` and
  `validation/full_response_results.json`) is a verbatim copy in
  `tests/model/reference/validation`, so the benchmark tests run without
  the manuscript repository; they previously read it through an absolute
  path. `tests/model/test_reference_provenance.py` checks that the saved
  results record the SHA-256 of the two copied scripts and that the copy
  lies inside the repository. `tests/model/reference/README.md` lists the
  hashes and the update procedure; ruff and black exclude the directory.
  `scripts/model_examples.py` (README example (a)) reads the saved results
  from this copy too; it previously read them from the manuscript
  repository, two levels above the checkout.
- New private module `syncmoments/_expm.py` (`expm_pade13` and the
  `custom_vmap` helpers `any_member` and `batched_only`), used by
  `syncmoments.transfer`. New test helpers without tests:
  `tests/_ad_matrix.py`, `tests/_ad_matrix_cases.py`,
  `tests/_ad_matrix_model_cases.py` and
  `tests/test_weight_normalisation_oracle.py` (the exact `Fraction` oracle).
- Licence metadata follows PEP 639 (`license = "MIT"`,
  `license-files = ["LICENSE"]`) and the build requires `setuptools>=77`;
  setuptools deprecates the TOML-table form of 0.2.0.
- `MANIFEST.in` adds `CHANGELOG.md`, `.readthedocs.yaml`, the tests
  (including `tests/model` and the reference copy), the scripts and the
  documentation sources to the sdist. The 0.2.0 sdist carried only the
  top-level `tests/test_*.py` files, so its tests could not run.
- The test inventory of `docs/DESIGN.md` (Section 12) is taken from the
  collection (`pytest --co`).

### Validation

- Run on the finished tree in both environments (JAX 0.10.0 with Equinox
  0.13.7, NumPy 2.3.5, SciPy 1.16.3; JAX 0.10.2 with Equinox 0.13.8,
  NumPy 2.5.3, SciPy 1.18.1, no mpmath): 5166 tests, 1635 of them `slow`.
  Fast suite 3531 passed in each; slow suite 1635 passed in each; no
  failures and no skips (`docs/DESIGN.md` Section 12.10).
- After the last two fixes (the propagators' Pade scale and the
  `max_squarings` coercion, with `tests/test_transfer_scale_and_budget.py`)
  the 20 test files that reach the transfer code were rerun in both
  environments: fast tier 916 passed in each, and the slow AD-matrix cells
  of transfer, paths and `moment_driven_slab` 565 passed in each. The final
  tree collects 5196 tests (1635 `slow`).
- 3790 new tests in 36 files, and 6 more in two existing files
  (`docs/DESIGN.md` Sections 12.7 and 12.10); 1705 of them are the
  AD-transform matrix. 0.2.0 failed 7 tests and skipped 10 with JAX
  0.10.2; all of them now pass.
- The documentation builds with `-W` under Sphinx 8.2.3 and 9.1.0.
  `review-results/radiation.json`, regenerated, has all 2401 numerical
  entries equal to the 0.2.0 record.
- The manuscript benchmark reproduces the saved finite predictions to
  5.55e-10 of channel intensity and the second-order errors 1.644e-4 and
  1.753e-3 in both environments.
- The README examples were rerun in both environments after the third
  review round; only timings and
  the printed basis of a three-dimensional null space in example (c)
  changed (with JAX 0.10.2 also the eighth and ninth digits of example
  (a)'s finite-response differences). The whole script takes 1 min 43 s
  and 6.1 GB with JAX 0.10.0, 1 min 50 s and 7.0 GB with JAX 0.10.2. A
  rerun after the last review round, with example (a) reading the
  reference copy, printed the same lines apart from timings.

## 0.2.0 (2026-09-24)

First tagged release. Adds `synchro.model`, the finite joint response of
channel Stokes spectra from joint population statistics (manuscript
Section 4 and Appendix B), with explicit error budgets, declared population
assumptions and spectral fits.

### Added

- `build_basis` / `SpectralBasis`: Legendre-projected, channel-integrated
  spectral bases with derivatives in energy, field strength and Faraday
  depth (`eq: smooth channel kernel`, `eq: channel derivative coefficients`),
  for the exact harmonic kernel (product-coordinate angular quadrature,
  moving lines, per-line Faraday phase) and the ultra-relativistic continuum
  kernel.
- `predict` and `direct_channel_average`, returning a `Prediction` with
  channel Stokes, moments, a typed `ErrorBudget` (each term `bound`,
  `measured`, `estimate`, `unbounded` or `not_applicable`) and provenance.
- `JointMoments`, `MomentIndex` and `ParameterMap`: declared assumptions as
  parametrisations of the moment vector (independent parameter groups,
  independent screen, field independence, azimuth separability, isotropic
  pitch, Gaussian depth, nodal populations), recorded in every output, with
  measured discrepancies, manuscript factorisation bounds or caller
  allowances (`assumption_allowances`).
- Screen phase routes: `GaussianScreen`, `CumulantScreen`, `EmpiricalScreen`,
  `LaplaceScreen`, `GammaScreen` (`eq: illustrated screens`).
- `synchro.model.fit`: `StokesData`, `fit_linear` (closed-form weighted least
  squares with column equilibration), `LogDensity` and `fit_bfgs`,
  `fit_nodal`, `identifiability`, `feasibility_checks`, `FitResult`.
- Sphinx documentation for Read the Docs; `docs/DESIGN.md` design and
  acceptance record; `scripts/model_examples.py`.
- `synchro.__version__`.

### Changed

- `stokes_harmonic` accepts an optional static `n_nodes` (Bessel quadrature
  resolution); the default behaviour is unchanged.
- Slow tests are marked `slow` and run with `-m slow` or `SYNCHRO_RUN_SLOW=1`.

### Validation

- 187 existing tests, 1168 new tests and 42 slow tests pass.
- The manuscript's Section 5.3.1 smooth-channel example is reproduced with
  automatic differentiation: second-order errors 1.644e-4 and 1.753e-3 of
  channel intensity (manuscript: 1.64e-4 and 1.75e-3), and the saved finite
  predictions to within 5.6e-10 of channel intensity.

## 0.1.0

Unreleased development versions: exact harmonic Stokes kernels, fixed-harmonic
quadratic contractions, Faraday-screen and joint emission-depth averages,
polarised transfer modules.
