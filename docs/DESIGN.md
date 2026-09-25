# `syncmoments.model`: finite joint response and spectral fits

This document describes the top-level layer of the package: how a finite set
of joint population moments predicts channel-integrated Stokes spectra with an
explicit error budget, how assumptions about the population are declared, and
how the same response matrix serves prediction and fitting. It follows the
manuscript (Zhang & Chluba, *A statistical framework for synchrotron
emission*), whose equation labels are quoted in backticks; anything the
manuscript does not define is tagged `[extension]`.

Section 12 records the test inventory of the 0.2.0 acceptance run on
2026-09-24, after three rounds of review fixes (test files, what each
establishes), and the numbers the executed README examples printed.
Section 12.10 records the changes of 0.3.0, released 2026-09-25 (task T-004:
dependency robustness, bounded memory, recurrence-based derivatives,
per-variable caps, symmetry assumptions, and the derivative rules of the
transfer and weight-normalisation code) and their acceptance in two
environments. The pass/fail record of the suite is kept outside this
document.

## 1. Principles

1. `SpectralBasis` is the one expensive object (kernel, channels, truncation,
   reference point and phase route). Everything downstream is linear algebra
   on the real moment vector `m`. `predict` and `fit` share one response
   matrix `C`.
2. Assumptions are parametrisations of `m` (`ParameterMap: theta -> m`),
   never switches inside the physics. No independence, isotropy or
   Gaussianity is a default; each is an opt-in object and is recorded in
   every output.
3. Dirac-line kernels are evaluated only after pairing with a smooth,
   compactly supported channel response. The Faraday phase and any screen
   characteristic function multiply each harmonic line at its own
   `tau_m = 2 lambda_m^2` inside the channel sum (`eq: smooth channel kernel`).
4. Every approximation carries an `ErrorTerm(value, kind, note)` with
   `kind in {bound, estimate, measured, unbounded, not_applicable}`. A missing
   input is `unbounded` (`value=None`), never zero. `ErrorBudget.total()` is
   `unbounded` when any term is, otherwise the sum with the weakest kind.
5. Every object names the manuscript equation it implements in `LABEL`.
6. Coordinates are the manuscript's Legendre cosines `(mu, eta)` and the
   dimensionless displacements
   `z = ((gamma - gamma0)/s_gamma, (B - B0)/s_B, (varphi - varphi_ref)/s_varphi)`
   of `eq: local response remainder`. Moments and bases are stored in `z`;
   the raw-displacement form of `eq: explicit joint moments` is
   `M_raw = s_gamma^r s_B^s s_varphi^b M_z`. The manuscript's worked example
   uses `scales = (gamma0, B0, 1/(2 (c/nu_*)^2))`.

Units: Gauss, Hz, rad/m^2, metres for wavelengths, erg/s/sr per electron for
harmonic powers; channel outputs are channel-integrated powers for
`unit_peak` channels and per-Hz averages for `unit_integral` channels.
Importing `syncmoments` enables float64.

## 2. Object model

```
syncmoments/model/
  index.py         Truncation, MomentIndex, Entry, lower_set_margin
  errors.py        ErrorTerm, ErrorBudget, AssumptionRecord, Provenance
  channels.py      Channels
  phase.py         TaylorPhase, GaussianScreen, CumulantScreen, EmpiricalScreen
  screens.py       LaplaceScreen, GammaScreen (eq: illustrated screens)
  moments.py       Reference, Support, PopulationSamples, JointMoments
  bounds.py        RemainderInputs, screen_/azimuth_factorisation_bound, depth_error_bound
  assumptions.py   Factorisation, Closure, ParameterMap, named constructors
  kernels.py       KernelModel protocol, Modes, ContinuumKernel, PolynomialTestKernel, required_m_max
  harmonic.py      HarmonicKernel
  basis.py         build_basis, SpectralBasis, basis_convergence
  predict.py       predict, direct_channel_average, Prediction
  adapters.py      to_joint_faraday, fixed_harmonic_comparison, burn_screen_check
  fit/observation.py  StokesData
  fit/linear.py       fit_linear, fisher
  fit/nonlinear.py    Transform, LogDensity, fit_bfgs, fit_nodal
  fit/diagnostics.py  identifiability, feasibility_checks
  fit/result.py       FitResult
```

Files whose name starts with an underscore are private helpers of the module
they sit next to; their public names are re-exported from the public module.
All classes are immutable `equinox.Module`s. Shape and configuration errors
raise `ValueError` at trace time; value errors (non-finite, negative, out of
range) raise through `equinox.error_if`, so they work under `jax.jit`. Every
traced path supports JIT and automatic differentiation. `scipy` is used only
in the tests.

The data flow is

```
Channels + kernel + Truncation + Reference + phase route
        --build_basis-->  SpectralBasis (I_basis, V_basis, P_basis, kernel_terms)
PopulationSamples | ParameterMap(theta)  -->  JointMoments (m0, m2)
SpectralBasis + JointMoments + amplitude  --predict-->  Prediction (stokes, budget, provenance)
SpectralBasis + StokesData + ParameterMap --fit_linear / fit_bfgs / fit_nodal-->  FitResult
```

## 3. Index layout (`index.py`)

`Truncation(L_mu, L_eta, N, depth_degree=None, *, max_orders=None)` holds
static integers. `N` is the total Taylor degree. `depth_degree=None` uses the
main-text cutoff `r + s + b <= N` for the `P` block; `depth_degree=L` uses the
`app: depth moments` layout `r + s <= N, b <= L` (`app: depth moments`).
`max_orders=(N_gamma, N_B, N_depth)` `[extension]` (each an int or `None`)
also requires `r <= N_gamma`, `s <= N_B`, `b <= N_depth` (Section 3.1).

Rows are tuples `(l, k, r, s, b)` in lexicographic order:

| block | rows | size |
|---|---|---|
| `h0` | `l <= L_mu`, `k <= L_eta`, `r + s <= N`, `b = 0` | `n0 = (L_mu+1)(L_eta+1) C(N+2, 2)` |
| `h2` | `l + k` even (with `parity=True`), `r + s + b <= N` | `n2 = n_+ C(N+3, 3)` (`app: depth moments`: `n_+ C(N+2, 2)(L+1)`) |
| `h0_ext` | real rows with `l + k` even, `b >= 1`, same cutoff as `h2` | not part of `m` |

Here `n_+ = [(L_mu+1)(L_eta+1) + d]/2` with `d = 1` iff both limits are
even. Slot 0 is `M0_{00;000} = 1`. `h0_ext` is used by `ParameterMap`
(azimuth separability needs `M0` with `b > 0`) and by
`JointMoments.from_samples`. `parity=True` (default) drops the `h2` rows with
`l + k` odd because their bases vanish identically (`eq: angular parity`);
`parity=False` keeps them `[extension]`. `components` without `"V"` drops the
odd-parity `h0` rows (continuum kernel).

Flattening (`extra eq: finite fit model`):
`m = concat(M0[h0], Re M2[h2], Im M2[h2])`, `n_real = n0 + 2 n2`. `I` bases
are nonzero only on `h0` rows with `l + k` even, `V` bases only on `l + k`
odd (masks `parity_I`, `parity_V`); both parity classes stay in `h0` because
they are different moments.

Worked enumeration at `(L_mu, L_eta, N) = (2, 2, 2)`: `n0 = 9 * 6 = 54`,
`n_+ = 5`, `n2 = 50`, `n_real = 154`, free parameters 153.

| block | slot formula | index tables |
|---|---|---|
| `h0` | `6 (3 l + k) + i(r, s)` | `i: (0,0)->0, (0,1)->1, (0,2)->2, (1,0)->3, (1,1)->4, (2,0)->5` |
| `h2` real | `54 + 10 p(l, k) + j(r, s, b)` | `p: (0,0)->0, (0,2)->1, (1,1)->2, (2,0)->3, (2,2)->4`; `j` over `(0,0,0), (0,0,1), (0,0,2), (0,1,0), (0,1,1), (0,2,0), (1,0,0), (1,0,1), (1,1,0), (2,0,0)` |
| `h2` imaginary | real slot + 50 | |

Checks: slot of `M0[l=1,k=2;r=1,s=0]` is 33; `Re M2[l=0,k=2;r=0,s=1,b=0]`
is 67 and its imaginary part 117; `h0_ext` has 20 rows. Other sizes
(`n0 / n2 / n_real`): `(1,1,1) -> 12/8/28`; `(0,0,2) -> 6/10/26`;
`(2,2,N=2,depth_degree=3) -> 54/120/294`. `MomentIndex.labels()` prints
`"M0[l=1,k=2;r=1,s=0]"` and `"Im M2[l=0,k=2;r=0,s=1,b=1]"`. Tables are
tuples, so an index is hashable and can be a static field.

### 3.1 Per-variable caps `[extension]`

The caps intersect the cutoffs above with `r <= N_gamma`, `s <= N_B`,
`b <= N_depth`. Caps that remove no row normalise to `None`, so a
non-binding `max_orders` gives a `Truncation` equal to (and hashing like) the
default and a byte-identical `MomentIndex`; `is_capped()` is true only when a
row is removed. In the `app: depth moments` layout `N_depth < depth_degree`
raises `ValueError` (set `depth_degree` instead), because the adapters
enumerate `b` up to `depth_degree`. `caps()` returns the effective limits,
`max_b()` honours `N_depth`, so `build_basis` picks
`TaylorPhase(min(N, N_depth))`. `main_text_sizes` has no closed form for a
capped truncation and raises. Sizes (`n0 / n2 / n_real`):

| truncation | sizes |
|---|---|
| `(2, 2, 2)` | 54 / 50 / 154 |
| `(2, 2, 2, max_orders=(None, 0, None))` | 27 / 30 / 87 |
| `(2, 2, 2, max_orders=(1, 1, None))` | 36 / 40 / 116 |
| `(2, 2, 2, max_orders=(None, None, 0))` | 54 / 30 / 114 |
| `(8, 8, 2)` | 486 / 410 / 1306 |
| `(8, 8, 2, max_orders=(None, 1, None))` | 405 / 369 / 1143 |
| `(8, 8, 2, max_orders=(None, 0, None))` | 243 / 246 / 735 |

The retained `(r, s, b)` set is a lower (downward-closed) set `Lambda`.
`margin_rows()` returns its telescoped margin `S(Lambda)`
(`lower_set_margin`, Section 8.1): for the main-text layout the margin of
the `P` row set (the `I, V` set is its `b = 0` section), for the
`app: depth moments` layout the margin of the `(r, s)` set with `b = 0`
(the depth order is bounded by the depth tail). The basis cost falls with
the caps: `_basis_taylor.needed_orders` collects the `(r, s)` of the
retained rows and `angular_taylor(..., orders=)` chooses, for each nested
forward level, the direction set (`z_gamma`, `z_B` or both) with the fewest
forward tangents (`_harmonic_taylor.level_plan`; at `N = 2`, 9 tangents
uncapped, 6 with `N_B = 1`, 4 with `N_B = 0`). An uncapped truncation takes
the same all-direction plan as before.

The provenance `truncation` record and `JointMoments.to_dict()["index"]`
carry `max_orders` after `depth_degree`: `null` when uncapped, otherwise the
binding `[N_gamma, N_B, N_depth]` with `null` for an uncapped variable.

Rows of the response matrix `C` (`(4 n_ch, n_real)`, channel-major, Stokes
order I, Q, U, V): `C[I_j, h0] = I~_j` masked by `parity_I`;
`C[V_j, h0] = V~_j` masked by `parity_V`; with `P~ = a + i b`:
`C[Q_j, Re] = a`, `C[Q_j, Im] = -b`, `C[U_j, Re] = b`, `C[U_j, Im] = a`.
All other entries are zero.

## 4. Channels (`channels.py`)

`Channels` holds `family`, `centres_hz`, `widths_hz`, `support (n_ch, 2)`,
Gauss-Legendre `nodes`/`weights` on each support (for the continuum route),
`normalisation`, `smoothness` and family parameters. `__call__(nu_hz)`
returns the differentiable response `(n_ch, *nu.shape)`, exactly zero at and
outside the support edges.

| family | response | smoothness |
|---|---|---|
| `bump` (default; the manuscript's channel) | `exp[1 - 1/(1 - t^2)]`, `t = (nu - c)/w` | C-infinity (99) |
| `planck_taper(taper)` | flat centre, Planck tapers | C-infinity |
| `raised_cosine` | | 1 |
| `tophat` | | 0 |
| `gaussian(support_sigma=6)` | truncated at `support_sigma` sigma; edge jump recorded | 0 |
| `from_table(..., smoothness=, argument="nu"\|"omega")` | interpolated table | declared |

`normalisation="unit_peak"` (default) gives band-integrated powers (the
manuscript's example); `"unit_integral"` gives `int R(nu) dnu = 1` per Hz.
`build_basis` requires `smoothness >= N + 1` for a `basis_remainder` of kind
`bound`; otherwise that term is `unbounded` with the reason. No `2 pi`
factor enters anywhere; `from_table(..., argument="omega")` converts
`R_nu(nu) = R_omega(2 pi nu)`. `width_ratio()` is recorded in
`provenance.numerics`; `edge_jump()` returns the largest unit-peak response
at a support edge (0 for continuous families).

## 5. Phase routes (`phase.py`)

A `PhaseWeights` object returns `(n_weights, *tau.shape)` complex weights at
`tau = 2 lambda^2`, `lambda = C_SI_M / nu` in metres, evaluated per harmonic
line or per frequency node.

| route | weights | forced assumptions |
|---|---|---|
| `TaylorPhase(degree)` (default) | `w_b = exp(i tau varphi_ref) (i tau s_varphi)^b / b!`, `b = 0..degree` | none |
| `GaussianScreen(mean, sigma)` | `w_0 = exp(i tau mean - tau^2 sigma^2 / 2)` (`eq: independent gaussian screen`, Burn) | `independent_screen`, `gaussian_screen` |
| `CumulantScreen(kappa (4,), g5_bound=None)` | `w_0 = exp(g_4(tau))` | `independent_screen`, `cumulant_screen`; zero-free interval assumed |
| `EmpiricalScreen(depths, weights)` | `w_0 = sum_i w_i exp(i tau depth_i)` | `independent_screen` (exact for the supplied discrete screen) |
| `LaplaceScreen(mean, sigma)` | `w_0 = exp(i tau mean) / (1 + (tau sigma)^2 / 2)` (`eq: illustrated screens`) | `independent_screen`, `laplace_screen` |
| `GammaScreen(mean, sigma, shape=4, sign=1)` | `w_0 = exp(i tau mean) exp(-i s t sqrt(k)) (1 - i s t / sqrt(k))^(-k)`, `t = tau sigma` | `independent_screen`, `gamma_screen` |

`varphi_ref` and `s_varphi` are read from `Reference` by `build_basis`.
The Taylor degree must equal the index's maximum `b`; screens force
`max b = 0`. Screen hyper-parameters are baked into the basis; a fitted
screen uses the Taylor route with `gaussian_screen(index, mean, sigma,
fit_hyper=True)` (the `gaussian_depth` closure with free hyper-parameters)
or a rebuild.

## 6. Kernels (`kernels.py`, `harmonic.py`)

`Modes(I (n_ch,), V (n_ch,), P (n_weights, n_ch) complex)` are the channel
kernels at one `(gamma, B, mu, eta)`; `P` excludes `exp(2 i phi)`.
`ProjectedModes` are their Legendre projections per `(l, k)` pair, already
multiplied by `(2l+1)(2k+1)/4`, in `index.pairs` order. `build_basis`
takes their `(z_gamma, z_B)` Taylor tensors from
`HarmonicKernel.angular_taylor` for the harmonic kernel with
`derivatives="analytic"` (Section 6.1) and from nested `jax.jacfwd` of
`angular_projection` for every other kernel and for `derivatives="autodiff"`.

### 6.1 `HarmonicKernel`

`HarmonicKernel(m_max, *, n_nodes=None, n_outer=64, n_inner=64, n_mu=48,
n_eta=48, m_chunk=64, chunk_budget=CHUNK_BUDGET, tail_probe=None,
E_phys=None, quadrature="product", derivatives="analytic")`
implements `eq: smooth channel kernel` and `eq: channel derivative
coefficients`: `H_{P,j} = sum_m Q_m R_j(nu_m) w(tau_m)`, `nu_m = m nu_B / D`.
Harmonics are traced floats `m = 1..m_max` batched inside `lax.map` steps.
`describe()` (the provenance kernel record) includes `chunk_budget` and
`derivatives`.

Angular projection, primary route (`quadrature="product"`): for each harmonic,
channel and sign the cell in `t = mu eta` where the line meets the channel
support is integrated with `t = sign v^4`, `v` on Gauss-Legendre nodes, and
the inner variable `log mu` from `log|t|` to `0`; the `mu < 0` half enters
through the parity factors `(1 +/- (-1)^(l+k))`. This resolves the
`1/|mu|` Jacobian of the double projection before parameter differentiation
(main text, Section 4). Because `R_j` vanishes to all orders at its support
edges, the boundary terms of the `(gamma, B)`-dependent cells vanish.
Automatic differentiation is the derivative of the computed values in
cells with `lo = 0` or `lo >= 0.1 hi`, except at an exact coincidence of a
channel edge with a line at `t = 0`: there the rule is not differentiable,
AD freezes the lower edge (zero tangent) and difference quotients grow like
`h^(-3/4)`. In cells with `0 < lo < 0.1 hi` (a
channel edge meeting a line near `t = 0`) the nodes follow the moving bounds
affinely in `t` with frozen node values (`RHO_SWITCH = 0.1` in
`_harmonic_cells.py`), which avoids the `lo^(-3/4)` node velocity of the
`v`-map; AD then gives the rule applied to the derivative of the
pulled-back integrand. Either way a derivative carries its own quadrature
error, estimated by `numerical` and not bounded; at 64
nodes the two node motions differ by 2.7e-6 relative at the switch. The
secondary route (`quadrature="tensor"`, `n_mu x n_eta`
Gauss-Legendre) serves the polynomial test kernel and the opt-in
product-versus-tensor route check (`build_basis(..., cross_route=True)` or
`convergence="full"`), recorded in `provenance.finite_checks`; it converges
slowly for derivative bases.

Bessel functions: `harmonic_lines` needs `J_{m-1}`, `J_{m+1}` and `J_m'`
at `x = m beta sin(theta) / D`. `syncmoments.bessel.bessel_jn_neighbours`
evaluates all three with one periodic trapezoid rule on the order-`m`
contour: the order-`k` integrand is `exp(-k a) E e^{i(phi + (k - m) t)}`, so
the three integrands share `cos(phi)`, `sin(phi)` and `E` (three
transcendentals per node instead of nine). The contour-shift identity is
exact for every integer order because the integrand is entire and
periodic, so no fallback to separate contours is needed; precision is
checked against SciPy `jv`/`jvp` for `m` in {1, 2, 16, 48, 112, 240, 496,
1008}, `x` from 1e-8 to the resolution edge (relative 2e-12), not bounded.
`bessel_jn_and_prime` is unchanged.

Memory `[extension]`: one `lax.map` step of `angular_projection`,
`angular_taylor` and the harmonic tail probe holds at most
`max(chunk_budget, n_nodes)` primal Bessel integrand values, one angular
point of one harmonic being the floor (`CHUNK_BUDGET = 2^20`, defined in
`_harmonic_cells`; `m_chunk` is an upper cap on harmonics per step). `_harmonic_cells.chunk_plan(points, n_nodes, m_chunk,
budget)` returns `(harmonics, block)` with `harmonics * block * n_nodes <=
max(budget, n_nodes)`, and `sum_point_blocks` sums the flattened
`(2, n_outer, n_inner)` product cells or the `(n_mu, n_eta)` tensor nodes in
checkpointed point blocks (padding masked to zero). At the benchmark the
plan is one harmonic and two point blocks per step. Blocking changes only
the summation order. The harmonic tail probe blocks its samples under the
same budget (`_harmonic_tail.sample_plan`). `for_tangents(t)` returns a copy
with `chunk_budget // t`, which the basis build (`_basis_taylor`) uses for
its nested `jacfwd`, where each tangent carries `n_nodes` values per point;
the budget counts primal values only, so a caller that nests `jacfwd`
without it holds a multiple of the budget. The nested-`jacfwd` remainder
probe divides by `4^q` only for `derivatives="autodiff"`; with the
analytic rule the tangents do not pass through the Bessel quadrature, and
the division made that probe 1.8 to 2.5 times slower for a peak lower by
12 to 24 % (Section 8). `channel_modes` plans for one
point, so vmapped over `S` samples it holds `S` times that:
`direct_channel_average` caps its `lax.map` batch by
`_direct.sample_batch`, which applies the kernel's
`samples_per_step(channels, requested) = max(1, min(requested,
chunk_budget // (h n_nodes)))` (`h` harmonics per step of the one-point
plan; 102 samples for `HarmonicKernel(40)` at the default budget).
`batch_size` of `direct_channel_average` is an upper cap. For the benchmark
channels, `S = 4096` and `batch_size = 1024`, peak RSS fell from 0.72 to
0.54 GB (JAX 0.10.0) and from 1.23 to 0.63 GB (JAX 0.10.2); the time rose
from 1.3 to 1.9 s, and a larger `chunk_budget` restores it at the old memory.

Derivatives: `derivatives="analytic"` (default) computes the Taylor tensors
`{(r, s): d^r_{z_gamma} d^s_{z_B}}` of the projected modes
(`angular_taylor(channels, gamma0, B0, *, scales, ..., order, orders=None)`)
without any tangent through the Bessel quadrature. One band of orders
`J_{m+k}`, `|k| <= width`, per point (`_bessel_recurrence.bessel_band`, one
contour) gives the Taylor polynomial of `(J_{m-1}, J_{m+1}, J_m')` in
`x - x0` through the recurrence `J_n' = (J_{n-1} - J_{n+1}) / 2`; the
per-point streams and the Legendre tables of the moving nodes are
differentiated by nested `jacfwd` with that polynomial, and the contraction
`sum_p q P_l P_k` is assembled by the Leibniz rule (`_harmonic_taylor`).
Node motion in `(gamma, B)` through `product_cells` is differentiated as
before, affine cells included. A split in which only `gamma` tangents
reach the Bessel evaluation and the `B` derivatives are assembled from
`B^2`, `R_j(nu_m)` and `w(tau_m)` at fixed Bessel values does not reproduce
the rule derivative: the product-cell nodes depend on `nu_B` (so on `B`),
which moves `(mu, eta)` and therefore `x`, and a frozen-node `B` derivative
differs from the rule derivative by quadrature error. The recurrence route
treats `gamma` and `B` alike.
`derivatives="autodiff"` nests `jacfwd` of `angular_projection` through the
quadrature and the contraction: the 0.2.0 structure, on the shared-contour
rule `bessel_jn_neighbours` instead of one contour per order, so it is not
bit-identical to 0.2.0. The two are the same AD of the same program (node
motion included) and agree within 1e-12 per Stokes block and per
derivative order at `gamma0 in {1.01, 2, 20, 50}` x `B0 in {1e-2, 10, 100}`
G, `N = 0..3`, on both angular routes. The channel centres of the
`gamma0 = 20` and `50` corners moved from 8 and 10 to 8.3 and 10.3, off the
edge-line coincidence, so lower-edge node motion is exercised;
`test_analytic_bases_equal_nested_autodiff_at_an_edge_line_coincidence`
keeps one coincidence case (`gamma0 = 50`, centre 10). Where the rule is
not differentiable (above) neither path is the derivative of the computed
values. Where autodiff of the shifted contour is inaccurate
(`|x| << m` at small `m`; at `m = 1`, `x = 1e-4` the third derivative is off
by more than 1e-3, and at `x = 1e-6` by more than its own size; the
separate contours of 0.2.0 also break down there, with different errors)
the recurrence matches SciPy to 1e-8; at `x = 0` it gives
the analytic derivative where 0.2.0 gave 0 (`harmonic_lines` has
`dx = 0` there, so bases are unchanged).

Peak RSS and time of `build_basis` at benchmark size (`HarmonicKernel(40)`,
the benchmark channels, `Truncation(8, 8, N)`, 64 x 64 product cells,
`convergence=False`, `/usr/bin/time -l`; one shared CPU machine, so times
are noisy; JAX 0.10.0 / 0.10.2):

| `N` | 0.2.0 | 0.3.0 `"analytic"` | 0.3.0 `"autodiff"` |
|---|---|---|---|
| 0 | 1.6 GB / not measured | 0.44 GB, 1.9 s / 0.49 GB, 1.8 s | 0.44 GB, 1.8 s / 0.50 GB, 1.7 s |
| 1 | 4.3 GB / not measured | 0.52 GB, 2.4 s / 0.54 GB, 2.5 s | 0.56 GB, 3.3 s / 0.68 GB, 3.8 s |
| 2 | 10.9 GB, 20.0 s / 45.6 GB, 32.6 s | 0.82 GB, 6.5 s / 0.93 GB, 7.4 s | 0.98 GB, 10.6 s / 1.40 GB, 11.3 s |
| 3 | not measured | 2.82 GB, 32.3 s / 3.44 GB, 25.9 s | 2.05 GB, 59 s / 3.07 GB, 66 s |

The `"autodiff"` entries for `N <= 2` were measured before `for_tangents`
divided its budget; the `N = 3` entry after. At `N = 3` the analytic path's
peak is XLA compile memory (runtime temporaries 70 MB), which the chunk
budget does not reduce; `N = 4` was not measured and may exceed 4 GB with
JAX 0.10.2. XLA
temporaries of the reverse-mode memory test (`m_max = 40`, three channels,
`Truncation(2, 2, 1)`, 64 nodes; `jacfwd` / `grad`): 0.2.0 6.7 / 5.8 GB
(JAX 0.10.0) and 25.3 / 26.0 GB (JAX 0.10.2); 0.3.0 33 / 33 MB and 33 / 40 MB.
Budget scan at `N = 2` (B's chunking before the recurrence route, JAX 0.10.0 /
0.10.2): `2^18` 0.76 / 0.95 GB, `2^20` 0.96 / 1.39 GB, `2^22` 1.86 / 3.38 GB,
with times within noise; the default `2^20` is kept.

`physical_error` is `unbounded` with the note "vacuum helical-orbit,
pure-rotation reference; zero only if declared" unless `E_phys` is supplied.
`truncation_error` is a zero `bound` when `m_max >= required_m_max`, every
channel is compact and no emitting sample (weight > 0, `B > 0`) lies
outside the declared `Support` (`gamma <= gamma_hi`, `B >= B_lo`, relative
slack 1e-12). The support condition is checked only on concrete samples; the
note says whether it was checked, and without samples (as in `build_basis`)
or under tracing it records the hypothesis as unchecked.
`predict(samples=..., kernel=...)` and `direct_channel_average` call
`truncation_error` on their samples, so the check runs on both routes. A sample outside
the support, or `m_max < required_m_max`, gives an `estimate` from
`tail_probe` extra harmonics on supplied samples (or at the reference
point, noted) when `tail_probe` is set, and `unbounded` otherwise; the note
names the violating variable and its range.

### 6.2 `ContinuumKernel`

`ContinuumKernel(*, n_nodes_F=128, tail_cutoff=50.0, x_min=1e-6, E_phys=None,
n_eta=48, chunk_budget=CHUNK_BUDGET)`
implements `eq: directional continuum`: `K_I = A F(x)`, `K_Q = -A G(x)`,
`B_perp = B sqrt(1 - eta^2)`, `x = nu / (a_B gamma^2)`; `channel_modes` sums
`w_n R_j(nu_n) K_S(nu_n) w(2 (c/nu_n)^2)` over the channel's quadrature
nodes. `mu` is ignored: `components = ("I", "Q")`,
`required_closures = ("uniform_mu",)`. `build_basis` raises for `L_mu > 0`,
records `isotropic_pitch [required by ContinuumKernel]` and drops the
odd-parity `h0` rows ("V not modelled"). `x < x_min` is a value error.
`truncation_error` is `not_applicable`; the finite `F`/`G` tail bound at
`(gamma0, B0, eta = 0)` enters the `numerical` slot on the mass column
`(0,0,0,0,0)`, so `predict` multiplies it by `|m_0|`.

Memory `[extension]`: `chunk_budget` bounds the `F`/`G` cosh-integral
values of one `lax.map` step. `angular_projection` runs its `eta` nodes in
blocks of `eta_block(channels) = max(1, min(n_eta, chunk_budget // (2 n_ch
n_nu n_nodes_F)))` through `sum_point_blocks` (checkpointed, padding
masked), with the Legendre weights folded in per block;
`samples_per_step` caps a caller that vmaps `channel_modes`
(`direct_channel_average`), and `for_tangents` divides the budget for
nested `jacfwd` (the basis build at `N = 2` uses `chunk_budget // 9`). One
`eta` node or one sample is the floor, which can exceed a very small
budget. `describe()` records `chunk_budget`. Against the unblocked code the
bases agree within 6.6e-16 per block, the probe envelopes within 7.7e-15,
the direct averages exactly; the refined-rule `numerical` term, a
difference of two bases, changes by 2e-9 of itself (1e-15 of the basis
block). The refined kernel of the convergence rebuild
(`_basis_core.refined`) keeps the caller's `chunk_budget`. The
nested-`jacfwd` remainder probe runs the projection with
`for_tangents(4^q)` at nesting order `q`, and `angular_residual` caps its
sample batch by `samples_per_step` and projects with
`for_tangents(batch)` (Section 8). Peak RSS and wall time of README
example (b), whole example (JAX 0.10.0 / 0.10.2): 6.71 / 27.07 GB without
the budget; 1.92 / 5.06 GB with the continuum budget alone, when the
unbudgeted order-3 `jacfwd` probe held 1.85 / 4.97 GB; 1.16 GB, 14.7 s /
1.79 GB, 14.4 s with the probe budgeted. Its basis build fell from
4.23 / 11.87 GB to 0.65 / 0.88 GB. Example (d): 4.55 / 12.22 GB without
the budget, 2.92 / 3.13 GB with it, when the example script's vmap of
`kernel.kernels` over every ray held 2.5 GB; 0.85 GB, 7.1 s / 1.12 GB,
7.3 s with that map batched by `samples_per_step(channels, 256)`. The
final figures are serial runs on an otherwise idle machine; the probe
outputs agree with the unbudgeted code to 2.2e-15 relative.

### 6.3 `PolynomialTestKernel` `[extension, test oracle]`

A polynomial in `z` of degree `<= N` times Legendre polynomials of degree
`<= L`, times `w(tau)` at fixed line positions. Its finite response is exact,
so it tests index bookkeeping, the `C` layout, factorials and `to_dict`
round trips independently of Bessel numerics.

### 6.4 Regimes and the refusal rule

`required_m_max(support, channels) = ceil((1 + beta_max) nu_hi,max / nu_B,min)`
with `nu_B,min = e B_min / (2 pi gamma_max m_e c)` is the largest harmonic
that can meet any channel on the declared support (the manuscript's
`harmonic_cutoff`). `build_basis` refuses `m_max < required_m_max` unless
`allow_truncated=True`, and names `ContinuumKernel` in the message.

| regime | kernel | notes |
|---|---|---|
| `gamma` up to a few tens, `m` up to about 1e3 | `HarmonicKernel` | the manuscript's benchmark: `gamma0 = 20`, `m <= 40` |
| Galactic (`B ~ 5 uG`, `gamma ~ 1e4`, `nu ~ 1 GHz`, `m ~ 7e11`) | `ContinuumKernel` only | `isotropic_pitch` required; `physical_kernel` stays `unbounded` because the continuum replacement is not bounded against the harmonic reference |

Measured costs of 0.2.0 (CPU, one machine; not remeasured for 0.3.0):
harmonic product quadrature with `m_max = 40`, three bump channels,
`L = 2`: value 0.9 s per call at 64 x 64 nodes; jit-compiled Jacobian at
32 x 32 about 1 s, Hessian 2.2 s, third order 5.8 s (compile 11 s).
Continuum kernel: sub-second. Section 6.1 has the 0.3.0 basis costs.

## 7. Statistics and assumptions (`moments.py`, `assumptions.py`)

`Reference(gamma0, B0, depth_ref=0.0, scales=(1, 1, 1))`: traced leaves,
`gamma0 > 1`, `B0 > 0` (Gauss), `depth_ref` in rad/m^2, positive scales.
`Support(gamma=(lo, hi), B=(lo, hi), depth=(lo, hi), truncated=None)`:
`None` leaves `excluded_tail` unbounded; `False` declares the population
complete (`declared_zero`); `True` makes the tail a required input and the
amplitude counts the retained column only. `PopulationSamples(gamma, B, mu,
eta, phi, depth, weights)` is a discrete measure with weights normalised on
use. `JointMoments(index, m0, m2, reference, m0_ext, assumptions,
discrepancy)` implements `eq: explicit joint moments` with `from_samples`,
`from_vector` (checks `m[0] == 1`), `to_vector`, `get(h, l, k, r, s, b)` and
`to_raw_displacements`.

### 7.1 Factorisations

Variables `VARS = (gamma, B, mu, eta, phi, depth)`. Every row is the average
of the product monomial
`chi = z_gamma^r z_B^s P_l(mu) P_k(eta) e^{i h phi} z_varphi^b`. For a
partition `G` of `VARS` into groups,

```
M^{(h)}_{lk;rsb} = prod_{g in G} < prod_{v in g} f_v(row) >_g
```

with `f_gamma = z_gamma^r`, `f_B = z_B^s`, `f_mu = P_l(mu)`, `f_eta = P_k(eta)`,
`f_phi = e^{i h phi}`, `f_depth = z_varphi^b`. Special cases:

| partition | formula | manuscript |
|---|---|---|
| `{depth} \| rest` | `M^{(h)}_{lk;rsb} = M^{(h)}_{lk;rs0} m_b`, `m_b = <z_varphi^b>` | `eq: independent screen moments` |
| `{B} \| rest` | `<P_l P_k e^{ih phi} z_gamma^r z_varphi^b> <z_B^s>` | |
| `{phi} \| rest` | `M2 = m_phi M0_ext`, `m_phi = <e^{2i phi}>` | `detail-eq: angular factorisation error` (constant conditional circular moment; weaker than independence) |
| `{mu} \| rest` with `uniform_mu` | `<P_l(mu) ...> = 0` for `l >= 1` | isotropic pitch |
| six singletons | product of marginal tables | fully independent |

### 7.2 `ParameterMap`

`ParameterMap.build(index, factorisation, closures=(), *, discrepancy=None,
name=None, label="[extension]", free_hyper=(), nodes=None)` (the named
constructors below call it; the raw dataclass also needs the computed
`tables_spec` and `gathers`): per group,
the distinct nonconstant projections of the rows of `m` onto that group's
variables form its free table (static, built at construction); rows whose
closed factor is exactly zero (`uniform_mu` with `l > 0`, and the rows a
symmetry closure drops group by group, Section 7.4) are removed
before tables are formed; reconstruction is a product of static gathers.
`h0_ext` rows are reconstructed whenever their projections are covered by the
free tables. Methods: `n_free()`, `labels()`, `is_affine()` (at most one
group has a free table, no fitted hyper-parameters, not nodal),
`affine_pieces(reference=...)` giving `m = P theta + c`, `__call__(theta,
reference) -> JointMoments`, `project(full, amplitude=None) -> Parameters`,
`constraint_residual(full)` (signed `m_full - m(project(full))`),
`assume(*others)` and `record() -> AssumptionRecord`. The record does not
depend on `theta`: a closure whose hyper-parameters are fitted (listed in
`free_hyper`) is recorded with `closure_kind` `"<kind> (fitted)"` (for example
`"gaussian_depth (fitted)"`) and one shared `NaN` placeholder per packed
hyper-parameter, so `JointMoments.assumptions`, a static field, is the same
for every `theta`, eagerly and under tracing (`jax.tree.map` across fits,
`jnp.stack` of draws and one `filter_jit` trace all work). Non-fitted
closures keep their constructor values. The fitted values are in `theta`
and in the provenance note "fitted hyper-parameters of '<map>' (the
AssumptionRecord holds NaN placeholders for them): ..." of the `FitResult`
and of `FitResult.prediction`. Closures:
`uniform_mu`, `gaussian_depth`, `cumulant_depth`, `delta`, `fixed_table`,
and the symmetry closures `pitch_symmetric`, `field_reversal_symmetric`
(Section 7.4).

Free parameters at `(L_mu, L_eta, N) = (2, 2, 2)` (the unit moment excluded):

| constructor | groups | free |
|---|---|---|
| `no_assumption` | one group | 153 |
| `independent_screen` | `{depth} \| rest` | 53 + 60 + 2 = 115 |
| `gaussian_screen(mean, sigma)` | `{depth}` closed Gaussian | 113 (+2 if `fit_hyper=True`) |
| `field_independent` | `{B} \| rest` | 26 + 60 + 2 = 88 |
| `azimuth_separable` | `{phi} \| rest` | 73 + 2 = 75 |
| `isotropic_pitch` | `{mu}` closed uniform | 17 + 40 = 57 |
| `independent_screen + azimuth_separable` | | 57 |
| `fully_independent` | six singletons | `N_gamma + N_B + max_b + L_mu + L_eta + 2 = 12` with `(N_gamma, N_B, max_b) = truncation.caps()` (`2N + max_b + ...` uncapped; `max_b = N`, or `depth_degree` in the `app: depth moments` layout) |
| `nodal(index, nodes)` | nonnegative weights on fixed nodes (softmax of logits) | number of nodes |
| `pitch_symmetric` | one group, odd-`l` rows dropped | 115 |
| `field_reversal_symmetric` | one group, odd-`l + k` rows dropped | 129 |
| `pitch_symmetric + field_reversal_symmetric` | | 103 |
| `fully_independent + field_reversal_symmetric` | six singletons, odd `l` and odd `k` dropped | 10 |

`Factorisation(groups=...)` builds a generic partition `[extension]`.

### 7.3 Discrepancy allowance

An assumption's discrepancy has four explicit sources; otherwise it is
`unbounded`:

1. `ParameterMap.discrepancy` at moment level, propagated as `A |C| Delta`.
   `predict` attributes it to the first `AssumptionRecord` of the moments;
   every further record is `unbounded`. Moments without an assumption
   record add it to `statistical_input`. `assume(*others)` keeps an input
   map's discrepancy only when the other maps add no constraint (the
   refined groups and closures equal that map's own); otherwise the
   combined map carries `None` (`unbounded`) and needs its own allowance,
   for example `ParameterMap.build(..., discrepancy=...)` on the combined
   map or `JointMoments.from_samples(parameter_map=combined,
   discrepancy="measured")`.
2. The manuscript bound helpers in `bounds.py`:
   `screen_factorisation_bound(sigma_P_in, Phi_R_abs_min)`
   (`detail-eq: screen factorisation error`) takes per-line arrays
   `(n_ch, n_line)`, `sigma_m` the ray standard deviation of
   `|R_j(nu_m)| P_m` and `Phi_m = |<exp(i tau_m R)>|`, and returns
   `sum_m sigma_m sqrt(1 - |Phi_m|^2)`; the `(n_ch,)` form takes
   `sigma = sum_m |R_j(nu_m)| sigma_{P_m}` (`int |R_j| sigma_{P(nu)} dnu` for
   a continuum) and `Phi = min_m |Phi_R|`. The standard deviation of the
   channel-summed incident polarisation is not a valid input.
   `azimuth_factorisation_bound(sigma_F, m_phi_abs)` (`detail-eq: angular
   factorisation error`; `(n_ch,)` only, `P` column, zeros in `I, V`).
   Both take `amplitude=` (default 1) and return a Stokes-unit
   `ErrorTerm` of kind `bound` for the caller's inputs; pass it to
   `predict` or `direct_channel_average` through `assumption_allowances`
   (item 4). Nothing checks that `sigma` and `Phi` are the population's
   statistics.
3. `JointMoments.from_samples(..., parameter_map=pm, discrepancy="measured")`
   stores `|m_joint - m_fac|` with `kind="measured"` (opt-in).
4. `predict(..., assumption_allowances={name: ErrorTerm})` and the same
   keyword of `direct_channel_average` replace the term of a named
   assumption, forced or moment-level, with a caller-supplied allowance in
   Stokes units (the amplitude included, as
   `screen_factorisation_bound(..., amplitude=N_src)` returns it). The value
   broadcasts to `(n_ch, 4)`; the allowance keeps its kind and value, its
   note is prefixed "allowance supplied by caller", and the provenance gets
   one note per allowance. An unknown name, a non-`ErrorTerm`, a value
   that does not broadcast or the kind `not_applicable` raises `ValueError`. When the first record's
   term is replaced, the moment-level discrepancy is not propagated (a
   provenance note says so). The amplitude cross term divides a Stokes-unit
   allowance by `N_src`. Nothing checks that the allowance holds.

Assumptions forced by the basis (screen routes: `independent_screen` and
`<shape>_screen`; `ContinuumKernel`: `isotropic_pitch`) are `unbounded` in
`predict` and `direct_channel_average` unless the caller supplies an
allowance. They override a
moment-level term of the same name, because the retained moments (`b = 0`
rows on screen routes, `L_mu = 0` rows for the continuum kernel) do not
resolve the variables they constrain; the provenance notes that the
moment-level discrepancy was not propagated.

Without allowances a screen-route or continuum prediction has an
`unbounded` total. With an allowance for every forced record the total is
finite once every other slot is valued. README example (d) replaces both
forced terms of a continuum screen route (`screen_factorisation_bound` for
`independent_screen`, a declared zero for `isotropic_pitch` on a population
uniform in `mu`); its total stays `unbounded` because `basis_remainder`,
`physical_kernel`, `excluded_tail` and `depth_model` are not supplied.

The forward path never projects a full tensor onto a factorised set on its
own.

### 7.4 Symmetry assumptions `[extension]`

`pitch_symmetric(index)` and `field_reversal_symmetric(index)` declare a
symmetry of the population measure. Each is one group over all of `VARS`
with a closure of the same kind and the empty variable group, so no
independence is assumed and no marginal is closed; the closure only drops
rows (`_factor.drops_row`), as `uniform_mu` drops `l > 0`.

Combined with a factorisation (`assume`), the symmetry is applied per group
of the partition. Each symmetry acts one variable at a time: pitch symmetry
flips `mu`; field reversal flips `mu` and `eta` and shifts `phi` by `pi`,
and `e^{i h pi} = 1` for `h in {0, 2}` (`_factor.FLIPPED`). Under the
product measure `p = prod_G p_G` the symmetry maps every group to itself,
so the marginal of `p o g` on `G` is `p_G o g_G`, and the joint invariance
holds exactly when every group marginal is invariant. A group factor whose
flipped exponents have an odd sum is then zero, and so is every row that
contains it. With one group this is the joint rule below (odd `l`; odd
`l + k`). When `mu` and `eta` are in different groups, field reversal drops
the rows with odd `l` or odd `k`, so no free table keeps `<P_odd(mu)>` or
`<P_odd(eta)>`, and the map can only represent populations that satisfy
its own declaration. Free counts: `fully_independent` with field reversal
has 10 at `(2, 2, 2)` and 16 at `(8, 8, 2)` (12 and 24 without; at
each case `ceil(L_mu/2) + ceil(L_eta/2)` fewer); the
partition `{eta} | rest` with field reversal has 52 at `(2, 2, 2)` (79
without). An exact product population with mirrored `mu` and `eta`
marginals is reproduced to 1e-14 relative and the Jacobian rank equals
`n_free` (`test_symmetry_factorisation.py`). A `delta` or `fixed_table`
closure whose own marginal is not symmetric is accepted with a symmetry:
only its even entries enter, so the map represents the symmetrised
marginal. Both assumptions are recorded and nothing checks the
combination.

A `fixed_table` combined with a symmetry always takes the full table: the
shape it has in the same partition without the symmetry closures (for
example `(2,)` for `('mu',)` at `(2, 2, 2)`). `_factor.symmetrised_tables`
returns, per closure, the full length, the kept positions and a note;
`_project.prepare` checks the full shape, `ParameterMap.symmetrised` (a
static field filled by `build`) holds the trims, and `_project.group_tables`
gathers the kept positions with static indices before `closure_table`, so
the map stays traceable under `jit`. Only the kept entries enter the
moments. The ignored entries are those the symmetry sets to zero (odd
flipped exponents on the group) and any entry no kept row uses; their
values do not change the moments. `build` and `assume` accept the same
input and give identical table specifications, gathers, labels, records
and bitwise-equal moments (`test_symmetry_fixed_table.py`). The reduced
table is refused on both routes, because `assume` builds the map without
the symmetry first, and that map needs the full table; the error names the
full shape and the dropped entries, e.g. "fixed_table on ('mu',) needs
shape (2,) (full table, symmetrised: dropped <P_1(mu)>)". The record keeps
the full input table in `hyper`, and `closure_kind` carries the note, e.g.
`fixed_table (full table, symmetrised: dropped <P_1(mu)>)+field_reversal_symmetric`.
Two records whose tables differ only in dropped entries are different,
although their moments are equal, so `_project.same_closure` treats such
closures as conflicting in `assume`. Groups where the symmetry drops no
entry (for example `gamma`) keep their previous table shape.

* Pitch symmetry: the measure is invariant under `mu -> -mu` with
  `(gamma, B, eta, phi, depth)` fixed. Since `P_l(-mu) = (-1)^l P_l(mu)`,
  every moment with odd `l` vanishes; those rows are removed from the free
  table.
* Field reversal: the measure is invariant under `B -> -B` with the
  electrons fixed. The pitch cosine and the viewing cosine change sign and
  the sky azimuth of the field turns by `pi`:
  `(mu, eta, phi) -> (-mu, -eta, phi + pi)` at fixed `(gamma, B, depth)`.
  `e^{2i phi}` is invariant and `P_l(-mu) P_k(-eta) = (-1)^{l+k} P_l P_k`,
  so every moment with odd `l + k` vanishes. With `parity=True` those are
  only `h0` rows, which carry the `V` basis (`parity_V`), so the finite `V`
  response is exactly zero.
* The reversal is local. It acts at fixed Faraday depth. Reversing the field
  along the whole line of sight would also reverse the sign of the Faraday
  depth, which this declaration does not model.

Both combine with each other and with the factorisations through
`assume()` (symmetry closures merge by kind and are never split). The
records carry the label `[extension]` and an `unbounded` discrepancy by
default; `JointMoments.from_samples(..., discrepancy="measured")` stores
`|m_joint - m_fac|`, which is `|m_joint|` on the dropped rows and 0
elsewhere. Free parameters (unit moment excluded): at `(2, 2, 2)` 153
without an assumption, 115 pitch, 129 field reversal, 103 both; at
`(8, 8, 2)` 1305, 769, 1065 and 649. A population with the declared symmetry
gives exact zeros on the dropped rows, and its prediction equals the joint
one to 1e-13 (`test_symmetry_assumptions.py`).

### 7.5 No field-averaged kernel route

A kernel averaged over a declared distribution of `B`
(`FieldAveragedKernel`, with `max_orders=(None, 0, None)` forced) was
designed and not implemented. It would require `B` to be independent of
`(gamma, mu, eta, phi, depth)` under the electron-number measure, with a
known distribution. In the target populations the field strength is
correlated with the electron energy (synchrotron losses scale as
`B^2 gamma^2`), with the Faraday depth (which integrates `n_e B_par`) and with
the field direction, so the route would rest on an assumption that does not
hold there. Per-variable caps (`max_orders` with a small `N_B`, Section
3.1) reduce the cost of the `B` direction and keep the joint moments.

## 8. Error budget (`errors.py`)

`ErrorTerm(value, kind, note, manuscript_term)`: `value` is a nonnegative
envelope `(n_ch, 4)` (or broadcastable) in channel Stokes units, `None` iff
`kind in {unbounded, not_applicable}`. `declared_zero(reason)` records a
zero bound with its reason. `total()` orders kinds `bound < measured <
estimate` and returns the weakest kind present.

`ErrorBudget` (`eq: channel error budget`) slots, in order:

| slot | content | manuscript term |
|---|---|---|
| `basis_remainder` | `eq: local response remainder` from `RemainderInputs` (angular residual and order-`N+1` derivative envelopes times absolute moments; for a capped truncation the lower-set margin sum of Section 8.1); `unbounded` for a line-kernel basis with channel smoothness `< N + 1`, and in `predict(samples=...)` when order `N + 1` is not in `certified_orders`; on a screen route `P` is the intrinsic residual only | `N_src <rho_nu>` |
| `statistical_input` | `A sum_a \|S~_a\| Delta_a`; an explicit `Delta` and a moment-level discrepancy not attributed to an assumption are summed (weaker kind) | `N_src sum_a \|S_a\| Delta_a` |
| `physical_kernel` | kernel model discrepancy; `unbounded` unless declared | `E_phys` |
| `harmonic_truncation` | omitted harmonics beyond `m_max`; the zero `bound` is conditional on the population lying inside the `Support`; `predict(samples=..., kernel=...)` and `direct_channel_average` check concrete samples (Section 6.1), a basis-level zero without samples keeps the unchecked hypothesis in its note | part of `E_num` |
| `excluded_tail` | population outside the declared support; in `predict(samples=...)` and `direct_channel_average` a `Support.truncated=False` declaration contradicted by concrete samples outside `gamma`, `B` or depth is `unbounded` (traced samples are not checked and the declared zero is kept) | `E_tail` |
| `numerical` | per-column envelope `\|C' - C\|` `(4 n_ch, n_real)` of the primary route refined against itself (2x angular nodes; `convergence="full"` also 2x Bessel or `F`/`G` resolution), contracted with `\|m\|` by `predict` (invariant under `Reference.scales`); continuum `F`/`G` tail bound on the mass column; kind `estimate` | `E_num` |
| `screen_exponent` | Taylor route `not_applicable`; Gaussian, empirical, Laplace and Gamma screens a zero `bound`; cumulant screen `unbounded` without `g5_bound`, otherwise the `estimate` `A <H_I>_angles max\|w_0\| (e^{eps5,j} - 1)` at the reference point (basis) or `A <\|P_j\|> (e^{eps5,j} - 1)` with the phased-sum modulus (direct average), `eps5,j = tau_max^5 g5/5!` at the channel's lower edge | `eq: screen exponent error` |
| `depth_model` | `tau <\|K_P\| \|delta varphi\|>` via `depth_error_bound` (`estimate` by default) | `eq: joint screen remainder` |
| `amplitude` | `delta N_src (\|C m\| + e)`, `e` the valued per-electron slots (`depth_model / N_src` and Stokes-unit assumption allowances `/ N_src` included; `excluded_tail` not rescaled), the same in `predict` and `direct_channel_average`; a concrete `delta N_src = 0` is a zero `bound` ("amplitude declared exact"); a traced one keeps the weakest kind of `bound` and the valued slots; a negative or NaN value raises through `equinox.error_if`, a complex one `ValueError`; `None` is `unbounded`; an `ErrorTerm` input is taken as the whole slot | |
| `assumption` | tuple of `(name, ErrorTerm)` pairs, one per declared assumption; an `assumption_allowances` entry replaces the named term (Section 7.3) | `E_phys` |

`terms()`, `total()`, `unbounded()`, `envelope()` and `to_dict()` are
provided; `to_dict()` fills the manuscript term of each slot. A budget with
assumptions crosses a JIT boundary through `equinox.filter_jit` (the names
are string leaves).

`Provenance` is a frozen dataclass of hashables (`package_version`, `kernel`,
`channels`, `truncation` (including `max_orders`), `reference`, `support`, `phase_route`,
`quadrature`, `assumptions`, `units`, `numerics`, `certified_orders`,
`finite_checks`, `notes`) with `to_dict()`, `with_notes()` and
`with_assumptions()`. `certified_orders` keeps its name for API
compatibility; it lists the derivative orders validated by finite-difference
tests (finite checks, not certificates). Notes such as "incident polarisation not modelled" and
"observing response assumed exact" are added when applicable.

`Prediction(stokes (n_ch, 4), amplitude, moments, budget, provenance,
channels)` offers `to_dict()`, `summary()` and
`propagate(R, response_uncertainty=None) -> ErrorTerm (n_data,)`, which
returns `|R| @ envelope + |delta R| @ |stokes|`
(`extra eq: data error propagation`); the second term is `unbounded` when
`delta R` is not supplied.

`RemainderInputs(rho_ang, H, absolute_moments, depth_tail,
intrinsic_residual, kind, *, absolute_depth_coefficients=None,
margin_H=None, margin_moments=None)` collects the
inputs of `eq: local response remainder`. `from_samples(samples, basis, *,
kernel=None, derivative_envelope="probe", angular_residual=None,
phase=None, ...)` estimates the order-`N+1` envelopes on the samples
(`kind="estimate"`) and computes the absolute moments on the
samples it is given, so those samples must be the predicted population (or
its absolute moments must be recomputed there, as README example (a) does).
`phase=None` means `basis.phase`, so the probe describes the route the basis
uses (a basis without a `phase` field falls back to the exact per-emitter
phase); another route makes the probed `P` inputs describe that route. The
`"probe"` envelope is refused (`ValueError`) unless `N + 1` is in
`SpectralBasis.certified_orders` (0..3 on the harmonic kernel, so
`N <= 2`); a supplied `derivative_envelope` array runs no `jacfwd` and is
not refused. `predict(samples=..., kernel=...)` does not raise there: it
leaves `basis_remainder` `unbounded` and its note names the order and
points to `errors=RemainderInputs(...)` or
`from_samples(..., derivative_envelope=<array>)`. `basis_remainder` returns
`unbounded` itself for a line-kernel basis with channel smoothness
`< N + 1`. In the `app: depth moments` layout on a Taylor route the depth
term uses `absolute_depth_coefficients` `(n_ch, n_a)` =
`sum_m |c_{a,m} R_j(nu_m)|` when supplied (kind of the inputs) and
otherwise the modulus of the phased `b = 0` coefficient (kind `estimate`).
On a screen route `P` is `intrinsic_residual` with no depth-Taylor term; the
emitter-depth/screen mismatch belongs to the `independent_screen`
assumption term.

The probe differentiates in `z_3 = (z_gamma, z_B, z_depth)` by one of two
methods (`_remainder_probe.probe_method`). For a kernel with
`angular_taylor` and `derivatives="analytic"` (the default
`HarmonicKernel`) it takes the `(z_gamma, z_B)` partials from
`angular_taylor` at each probe point, one pass per depth order `b` with the
single phase weight `d^b_{z_depth} exp(i tau (depth + s_depth z_depth)) =
exp(i tau depth) (i tau s_depth)^b`; the norm is the symmetric Frobenius
norm `sqrt(sum_beta |beta|!/beta! |d^beta|^2)`. Otherwise it nests
`jax.jacfwd` of `angular_projection` (the 0.2.0 algorithm, on the chunked
0.3.0 kernel). The two agree to 1e-12
relative per channel and Stokes block (`test_probe_taylor.py`). Costs at
the benchmark (`L = 8`, `N = 2`, probe order 3; JAX 0.10.0 / 0.10.2):

| probe | peak RSS | time per probe point |
|---|---|---|
| 0.2.0 (nested `jacfwd`) | 70.7 to 72.2 GB / not measured | 123 to 159 s including compile, two-point runs / not measured |
| 0.3.0 `angular_taylor` (default) | 3.2 / 3.7 GB | 6.7 / 7.2 s per further point; 13.2 s including compile |
| 0.3.0 nested `jacfwd`, `derivatives="analytic"` (`method="jacfwd"`), budget not divided (default) | 2.15 / 2.02 GB | 22.8 / 21.9 s per further point |
| same, `for_tangents(4^q)` (earlier 0.3.0 state) | 1.68 / 1.77 GB | 46 / 43 s per further point; 53 / 55 s including compile |
| 0.3.0 nested `jacfwd`, `derivatives="autodiff"`, `for_tangents(4^q)` (default) | 1.96 / 2.89 GB | 178 / 206 s for one sample point including compile |
| same, budget not divided | 3.97 / 10.09 GB | 103 / 124 s for one sample point including compile |

The nested-`jacfwd` probe carries the primal and three tangents per level,
so an order-`q` nesting holds `4^q` copies of the projection's working
set. It runs the projection with
`kernel.for_tangents(tangent_divisor(kernel, q))`: `4^q`
(`chunk_budget // 64` at `q = 3`) for `ContinuumKernel`,
`HarmonicKernel(derivatives="autodiff")` and unknown kernels, and 1 for the
analytic `HarmonicKernel`, whose tangents do not pass through the Bessel
quadrature. On the analytic kernel the division lowered the peak by 12 to
24 % and made the probe 1.8 to 2.5 times slower; divisors 4 and 16 were
also slower than 1 (32.5 and 34.4 s per further point with JAX 0.10.0).
On the `"autodiff"` kernel the undivided probe needs 10.1 GB with JAX
0.10.2, so it keeps `4^q` and is about 1.7 times slower than undivided
(divisor 4: 121 s and 5.3 GB with JAX 0.10.2, 121 s and 2.4 GB with JAX
0.10.0). On the continuum kernel of README example (b) the undivided
probe held 1.85 / 5.0 GB and divisor 4 held 1.26 / 2.18 GB, against
1.15 / 1.80 GB at `4^q` (Section 6.2). All divisors give the same `H` to
3.3e-15 relative per block, and divisor 1 is bit-identical to the analytic
default. The margin envelope builds one nesting per margin order that
occurs. `angular_residual` caps its sample batch by the kernel's
`samples_per_step` and projects with `for_tangents(batch)`. The
`angular_taylor` row and the `4^q` analytic row are serial runs on an
otherwise idle machine (one and three samples plus the reference point);
the other rows are serial runs at load averages 9 to 30 from concurrent
jobs, so their times are noisy. The analytic undivided row repeats an
earlier measurement (2.2 / 2.0 GB, 25.5 / 17.3 s) that this table had
attributed to `derivatives="autodiff"`.

### 8.1 Lower-set remainder for capped truncations `[extension]`

For a capped truncation the retained multi-indices form a lower set
`Lambda` and the total-degree operator-norm form does not apply. The
remainder of `T_Lambda f = sum_{beta in Lambda} d^beta f(0) z^beta / beta!`
is bounded by a sum over the telescoped margin `S(Lambda)`
(`lower_set_margin`):

```
|f(z) - T_Lambda f(z)| <= sum_{beta in S} |z^beta| / beta! sup_{t in [0,1]} |d^beta f(p_k(t))|
p_k(t) = (z_1, ..., z_{k-1}, t z_k, 0, ..., 0),  k the first nonzero coordinate of beta
```

for `f` of class `C^{|beta|}` on the box spanned by `0` and `z`. Derivation,
by induction on the dimension `d`: write `Lambda = union_{j <= J}
Lambda_j x {j}` over the last coordinate (each section `Lambda_j` is a
lower set). The one-dimensional Taylor step in `z_d` at fixed `z'` gives
`f = sum_{j <= J} z_d^j / j! h_j(z') + R_d` with `h_j = d_d^j f(., 0)` and
`|R_d| <= |z_d|^{J+1} / (J+1)! sup_t |d_d^{J+1} f(z', t z_d)|` (integral
form, valid for complex `f`). Since `T_Lambda f = sum_j z_d^j / j!
T_{Lambda_j} h_j`, the difference is `R_d + sum_j z_d^j / j! (h_j -
T_{Lambda_j} h_j)`, and the induction hypothesis applied to each `h_j`
telescopes the rest. Hence `S = {(0, ..., 0, J+1)} union_j S(Lambda_j) x {j}`
with `S({0..n}) = {n+1}` in one dimension. The path `p_k(t)` lies in the box,
so the supremum over the box also bounds.

* For total degree `|beta| <= N`, `S` is the shell `|beta| = N + 1`, the
  multi-index form of `eq: local response remainder`; uncapped truncations
  keep the manuscript's operator-norm form as the default.
* `S` contains every minimal element of the complement of `Lambda` and can
  contain more. The minimal elements alone do not bound the remainder:
  for `Lambda = {0,1}^2` (for example `Truncation(., ., 2,
  max_orders=(1, 1, None))` in `(z_gamma, z_B)`) and
  `f = (x - x^2)(y - y^2)` at `(1, 1)`, the remainder is 1, while the sum
  over the minimal elements `(2,0)` and `(0,2)` with the box supremum is
  0.5. `S` adds `(2,1)`, whose term covers it
  (`test_minimal_elements_alone_do_not_bound_a_tensor_product_set`).
* Every `beta` in `S` has a predecessor `beta - e_i` in `Lambda`, and
  `|beta| <= N + 1`.

`basis_remainder` of a capped basis is `rho + sum_lk sum_{beta in S}
margin_H margin_moments / beta!` per channel and component, with
`margin_H` `(n_ch, 3, n_lk, n_m)` the path supremum of `|d^beta K_{X;lk}|`
over the `n_m` rows of `truncation.margin_rows()` and `margin_moments`
`(n_lk, n_m)` the absolute moments `<|P_l P_k| |z^beta|>`. The operator-norm
inputs `H`, `absolute_moments` alone leave it `unbounded` with a note naming
the missing inputs. `from_samples` fills `margin_H` from the path points
`p_k(t)`, `t in {0} union segment_points` (an `estimate`; every order
`|beta|` must be in `certified_orders`), `margin_moments` and, in the
`app: depth moments` layout, the margin `intrinsic_residual`, whose `P`
entries are the `(z_gamma, z_B)` derivatives at the emitter depth. A
supplied `derivative_envelope` then has shape `(n_ch, 3, n_lk, n_m)`. The
numerical checks (`test_lower_set_truncation.py`,
`test_lower_set_remainder.py`): the remainder of a polynomial in `Lambda`
and its bound are zero; monomials on the margin attain the bound (relative
1e-13) for minimal elements and are covered otherwise; non-separable test
functions `exp((a + ib).z)` and `1/(1 - c.z)` are covered at random points
with a worst remainder/bound ratio above 1e-3; and `|predict - direct| <=
basis_remainder` holds for capped harmonic-kernel bases in both index
layouts, with a finite error above 1e-3 of the scale.

## 9. Basis and prediction (`basis.py`, `predict.py`)

`build_basis(kernel, channels, truncation, reference, *, support, phase=None,
quadrature=None, allow_truncated=False, convergence="angular", E_phys=None,
allow_nonsmooth=False, cross_route=None)` is an eager Python function
(validation on concrete inputs, provenance, the `required_m_max` refusal)
around a jitted core; `jax.jacfwd` with respect to the reference applies to
the core. `SpectralBasis` stores `I_basis (n_ch, n0)`, `V_basis (n_ch, n0)`,
`P_basis (n_ch, n2)` complex, `kernel_terms` (the kernel's four error
slots), `provenance`, `certified_orders` (set to `(0, 1, 2, 3)` by
`build_basis`; the name is kept for API compatibility: these are the
derivative orders validated by finite-difference tests, finite checks, not
certificates) and `phase`; `response_matrix()` returns `C (4 n_ch, n_real)`.
For a Dirac-line kernel a channel `smoothness < N + 1` is refused unless
`allow_nonsmooth=True` (the derivative columns then miss the edge
contributions of the moving lines, and `basis_remainder` is `unbounded`).

`convergence="angular"` (default) does one extra build: the same route at
doubled angular nodes (harmonic) or doubled `n_nu` and `n_eta` (continuum),
which fills `numerical` with the per-column envelope `|C' - C|`. `"full"`
also doubles the Bessel or `F`/`G` resolution; `False` leaves `numerical`
unbounded. The product-versus-tensor route check of the harmonic kernel is
opt-in: `cross_route=None` runs it only for `convergence="full"`, and
`True` or `False` decides explicitly (`ValueError` for a non-bool value or
for `cross_route=True` with `convergence=False`). When it runs, one more
build on the other angular route records
`max_a max_j |dC_ja| / max_j |C_ja|` as a named `route check` in
`provenance.finite_checks` (not a budget term; a large ratio can reflect
the tensor route's own error on derivative columns), and the `numerical`
term is unchanged. `provenance.quadrature` names the `numerical_route` and
the `check_route` (`None` when the check did not run).
`basis_convergence(basis, kernel, *, factor=2, full=False,
cross_route=False)` computes the envelope directly.

`predict(basis, moments, *, amplitude, errors=None, statistical_input=None,
amplitude_uncertainty=None, excluded_tail=None, depth_model=None,
samples=None, kernel=None, assumption_allowances=None) -> Prediction`
contracts `C m` and assembles the budget; per-electron slots are computed
at unit amplitude and scaled by it. Every missing input is an `unbounded`
slot: `basis_remainder` needs `errors` (a `RemainderInputs`, or `samples`
together with `kernel` for a probe), `statistical_input` needs `Delta_a` as
an `(n_real,)` array, a scalar (broadcast to every moment; `0` declares
exact moments) or an `ErrorTerm` over `m`, unless an unattributed
moment-level discrepancy is present (the two are summed), `depth_model` and
`amplitude` need their inputs, and `excluded_tail` follows
`Support.truncated`. With `amplitude = 0`, `amplitude_uncertainty > 0` and a
valued `depth_model` the amplitude slot raises through `equinox.error_if`.
`assumption_allowances` replaces named assumption terms (Section 7.3).

With `samples` and `kernel`, `predict` (i) probes `RemainderInputs` with
`from_samples(samples, basis, kernel=kernel, angular_residual="probe")` on
the basis phase route when `errors` is not given, and leaves
`basis_remainder` `unbounded` with the reason when `N + 1` is not in
`basis.certified_orders` (for example `N = 3`: pass
`errors=RemainderInputs(...)` with a supplied `derivative_envelope`);
(ii) always replaces `harmonic_truncation` by
`kernel.truncation_error(basis.support, basis.channels, samples=samples,
reference=basis.reference, phase=basis.phase)`, so the Support check runs;
(iii) makes a declared-complete `excluded_tail` `unbounded` when an emitting
sample lies outside the Support. `errors` that is not a `RemainderInputs`
raises `ValueError`.

`direct_channel_average(samples, kernel, channels, *, amplitude,
reference=None, phase=None, support=None, excluded_tail=None,
amplitude_uncertainty=None, depth_model=None, batch_size=256,
assumption_allowances=None)` averages the channel kernels over a discrete
population without truncation (`basis_remainder = not_applicable`,
`numerical` `unbounded`); it checks the samples against the `Support` for
`harmonic_truncation` and `excluded_tail`, and its amplitude slot has the
same cross term as `predict` (`delta N_src` times the valued per-electron
slots, `depth_model / N_src` and the screen term); the omitted unbounded
`numerical` slot is named in the note.

## 10. Fit module (`fit/`)

`StokesData(stokes, noise, mask=None, response=None, discrepancy=None,
response_uncertainty=None)`: `stokes (n_ch, 4)` channel-major, or the data
vector `(n_data,)` with a `response (n_data, 4 n_ch)`; `noise` is `(n_data,)`
variances or an `(n_data, n_data)` covariance; `mask` marks kept rows and
broadcasts from `(4,)` to `(n_ch, 4)` (with a response, a mask over the
`n_data` rows). `whitened()` returns `(L^-1 d, L^-1)` on the kept rows with
`Sigma_kept = L L^T`; `design(C)` returns `R C` on the kept rows;
`dof(n_params) = n_kept - n_params`; `discrepancy_vector()` propagates a
Stokes-space `ErrorTerm` by `|R|`.

`fit_linear(basis, data, parameter_map, *, amplitude="fit",
discrepancy_policy="bias_bound", tol=1e-8, diagnostics=True) -> FitResult`: with `m = P theta + c` and
the unknown `u = (A, A theta)`, the design is `G = R C [c | P]`; the solver
minimises `||L^-1 (d - G u)||` by a rank-revealing SVD of the
column-equilibrated design `G D`, `D = diag(1/||G_i||)`, so the rank does
not depend on the `Reference` scales (minimum equilibrated-norm solution
when rank-deficient); `A = u[0]`, `theta = u[1:]/u[0]`; Fisher
`G^T Sigma^-1 G`, covariance `D V s^-2 V^T D` from the SVD of the whitened
`G D` (equal to `(G^T Sigma^-1 G)^-1` at full rank; `None` when
rank-deficient, rank reported).
`bias_bound = |K| (|R| E + |delta R| (|S_hat| + E))` with
`K = (G^T Sigma^-1 G)^+ G^T Sigma^-1`, `E` the declared discrepancy and
`S_hat` the fitted channel Stokes (`extra eq: data error propagation`). It
has the discrepancy's kind, or `estimate` when the plug-in
`|delta R| |S_hat|` term is nonzero. It is `unbounded` when no valued
discrepancy is declared, when a `response` is set without
`response_uncertainty` (zeros declare `R` exact), and when the design rank is
below `n` (only the identified projection is bounded; its maximum is quoted
in the note). The prediction's `statistical_input` is the one-sigma noise
plus `|dm/du| bias_bound` (kind `estimate`; exact in `m` for a fixed
amplitude, linearised through `theta = u[1:]/u[0]` otherwise) and is
`unbounded` whenever `bias_bound` is, including the default fit without a
declared discrepancy. `"inflate"` adds `diag(delta^2)` (declared discrepancy
only) to `Sigma` and is labelled a heuristic: the Fisher matrix, covariance,
`chi2` and the identifiability report then describe the inflated weights;
a provenance note gives the `chi2` under the declared noise.

Nonlinear: `Parameters(log_amplitude, tables, hyper, logits)`, `Transform`
(`forward`, `inverse`, `log_det`: log for amplitude and sigma, softmax for
nodal weights, tanh-box for support intervals), `LogDensity(basis, data,
parameter_map, prior=None, transform)` callable in `z` (jit, grad, vmap),
`fit_bfgs(logdensity, z0, *, maxiter=500, tol=1e-9, precondition=True,
restarts=2, line_search_maxiter=30, diagnostics=True)` (refuses
`n_params > 2000`; the search runs in Gauss-Newton-diagonal scaled
coordinates with the curvature floored at 1e-6 of the largest diagonal
entry, so the scale ratio is at most 1e3, and restarts after a line-search
failure; a trial point with a non-finite density or a log slot beyond
`|z| = 700` gets objective `+inf`; after a run that did not succeed the
lowest finite objective among the returned point, the recovered pre-step
point and the start is kept), `fit_nodal(basis,
data, nodes, *, iters=500, step=0.1, discretisation=None, tol=1e-8,
diagnostics=True)` (exponentiated-gradient steps on the simplex with the
amplitude profiled; `extra eq: nodal distribution`, `extra eq: nodal
bound`). Both return a `FitResult`; its `converged` flag is the
optimiser's own (BFGS success or the float64 precision floor; the nodal
simplex stationarity residual), not a global-optimum statement.

`fit_bfgs` forms `bias_bound = |K| |delta|` over `z` with
`K = J^+ L^-1`, `J` the Jacobian of the whitened model at the optimum (the
Fisher covariance at full rank; an equilibrated pseudo-inverse otherwise,
used only for the quoted maximum). `delta` and the `unbounded` cases are
those of `fit_linear`: no valued discrepancy, a response without
`response_uncertainty`, or rank below `n`. A valued result is an `estimate`
whose note begins "linearised at the optimum (Gauss-Newton ...; prior and
log-Jacobian curvature excluded)"; the linearisation error is not bounded,
and with an informative prior the MAP response to `delta` is smaller than
`|J^+ L^-1| delta`. The prediction's `statistical_input` is the one-sigma
noise plus `|dm/dz| bias_bound` (`estimate`) when a discrepancy is
declared, and `unbounded` otherwise (the noise maximum is quoted in the
note), as for `fit_linear`. `fit_nodal` has `bias_bound = None` and an
`unbounded` `statistical_input` with or without a discrepancy: the softmax
gauge makes its Fisher matrix in `(log A, logits)` singular, so no
covariance is formed. The same data therefore give the same error kind on
the linear and BFGS routes. From the degenerate start `z0 = 0` on
`gaussian_screen(fit_hyper=True)` BFGS returns but ends away from the truth;
start from a projected or near-truth point.

Diagnostics: `identifiability(basis, data, parameter_map, *, tol=1e-8,
theta=None, reference=None, amplitude=True)` gives the SVD of the whitened,
column-equilibrated design `G D` (the same rank rule as `fit_linear` and
the BFGS/nodal Fisher summaries), a labelled null-space basis,
per-parameter resolution and `weak(threshold)`. Singular values, modes,
`null_basis` and resolution are in the scale-invariant equilibrated
coordinates; `column_scales` holds `D`, and `D @ null_basis` gives the null
directions in `u`. `feasibility_checks(moments, index,
support)` tests necessary conditions only (`eq: joint moment feasible set`):
`m[0] = 1`; PSD of the `(z_gamma, z_B)` block of `M0_{00}`; support bounds;
`|M2_{00;000}| <= 1`; `|M2_{00;2s0}| <= M0_{00;2s0}` and the `z_B^2`
analogue; `|<P_1^2 e^{2i phi}>| <= <P_1^2>`; Cauchy-Schwarz
`|M2_{00;100}|^2 <= M0_{00;200}`, `|M2_{00;010}|^2 <= M0_{00;020}`. Checks
that need `M0` with `b > 0` are listed as not computable from the fitted
vector. `FitResult` carries `theta, moments, amplitude, fisher, covariance,
rank, chi2, dof, converged, n_iter, bias_bound, identifiability,
feasibility, prediction, provenance` and `to_dict()`, which emits strict
JSON (non-finite floats become `null`, so the `NaN` placeholders of a
fitted-hyper record round-trip; the `to_dict()` of `Prediction`,
`JointMoments`, `Provenance` and `AssumptionRecord` share the same
rendering). For a map with `free_hyper` the provenance of the result and of
its prediction carries the note "fitted hyper-parameters of '<map>' (the AssumptionRecord
holds NaN placeholders for them): hyper:<closure>:<name>=<value>, ...".

## 11. Adapters (`adapters.py`)

`to_joint_faraday(basis, moments)` flattens `a = (l, k, r, s)` and returns
`(coefficients, M, absolute_next)` so that `faraday.joint_faraday_average`
reproduces `predict` for the continuum kernel with delta channels and
`depth_degree = L`. `fixed_harmonic_comparison(basis, expansion, samples)`
compares with `expansion.mixed_average` at the two truncation orders (not an
equality). `burn_screen_check` compares `GaussianScreen` with a delta channel
against `rm.burn_depolarisation`.

## 12. Acceptance summary

Worktree `synchro-code-review` (JAX 0.10, Equinox 0.13.7, NumPy 2.3, one
CPU machine). The counts below are the tests collected per file by
`PYTHONPATH=. python -m pytest tests/model --co -q`, rechecked against the
0.3.0 collection of 2026-09-25 (Section 12.7 gives the totals and the
`slow` split). Every oracle is
NumPy/SciPy/mpmath code written independently of the JAX path; since 0.3.0
the mpmath values are literals generated offline by
`scripts/reference_constants.py`, so the tests do not import mpmath.
The manuscript's benchmark implementation and saved results are a
verbatim copy in `tests/model/reference` since 0.3.0, so the tests run
without the manuscript repository. Sections 12.1 to 12.9 describe the
0.2.0 acceptance; Section 12.10 records 0.3.0.

### 12.1 Benchmark (`tests/model/test_benchmark.py`, 45 tests)

Reproduction of the manuscript's smooth-channel example (Section 5.3.1):
`gamma0 = 20`, `B0 = 1` G, three bump channels at `y_j = 2, 4, 8` with 65 %
half-widths, the correlated population of `eq: channel toy population`,
`HarmonicKernel(m_max=40)` (`required_m_max` is 40, so the harmonic-tail
term is a zero bound), `Truncation(L, L, N)`, pinned against the saved
`finite_IQUV` and `direct_IQUV` rows of the manuscript's
`validation/full_response_results.json`, read from the verbatim copy in
`tests/model/reference/validation` (hash-checked, Section 12.10; units
converted by
`e^2 Omega_0^2 / (2 pi c)`). A fast variant (`L in {2, 4}`, 8 x 8 latent and
64 x 64 angular moment nodes) runs by default; the full variant (`L = 8`,
256 x 256 angular nodes, `N in {0, 1, 2}`) reproduces the table rows,
including the mixed-term deletion change at `N = 2, L = 8`. The basis
itself is checked against the manuscript's finite-difference coefficients
in `test_basis.py::test_benchmark_basis_matches_manuscript_coefficients`
(1e-6 of the largest coefficient per Stokes against the Richardson
combination of the two saved step sizes).

The executed README example (`scripts/model_examples.py a`) prints, for
the full-resolution population (1,048,576 samples) at `N = 2, L = 8`:

| quantity | manuscript | package |
|---|---|---|
| max relative difference from the saved `finite_IQUV` (per channel) | | 1.5e-6, 5.1e-7, 1.4e-6 |
| max relative error against `direct_IQUV` at width 1.0 (per channel) | 1.75e-3 (largest) | 1.75e-3, 5.3e-4, 2.2e-4 |
| `required_m_max` | 40 | 40 |
| harmonic lines meeting a channel at the reference | | `m = 1 .. 26` |
| `independent_screen` declared, measured term vs. reduced-minus-joint shift (of `I`) | | 7.7e-2 covers 7.6e-3 |
| declared inputs plus the remainder probe (`H` and `rho_ang` from a 16-atom discretisation, absolute moments on the population): total kind, envelope (of `I`) | | `estimate`, 3.8e2 (covers the 1.75e-3 error; loose by five orders of magnitude at `L = 8`) |

### 12.2 Oracle checks

| file | tests | establishes |
|---|---|---|
| `test_index.py` | 116 | row enumeration and slots against an `itertools`/`math.comb` oracle over 13 truncations; the worked numbers of Section 3; parity and component masks; hashability and one compile per index |
| `test_errors.py` | 45 | kind arithmetic (`bound < measured < estimate`, `unbounded` propagation), broadcasting, `to_dict` JSON round trips, `Provenance` immutability |
| `test_channels.py` | 41 | every family against closed forms; exact zero at and beyond the support; `jacfwd` finite everywhere; unit-peak and unit-integral areas to 1e-12 |
| `test_phase.py` | 20 | Taylor weights, `GaussianScreen` against `rm.burn_depolarisation`, `EmpiricalScreen` against `rm.screen_polarisation`, `CumulantScreen` remainder |
| `test_stokes_nodes.py` | 6 | `stokes_harmonic(n_nodes=)` reproduces the historical output bit for bit at `n_nodes=None` |
| `test_kernels.py`, `test_continuum.py` | 11 + 9 | `PolynomialTestKernel` exactness; `ContinuumKernel` against `ultrarel.F/G` quadrature, mu-independence, `x_min` guard |
| `test_harmonic.py`, `test_harmonic_projection.py` | 20 + 7 | line powers against `scipy.special.jv`; product-coordinate projections against `dblquad` and the manuscript's `full_response_product.projected`; `(gamma, B)` derivatives against Richardson differences to third order; tensor route degrades at narrow channels (recorded) |
| `test_moments.py` | 43 | `from_samples` against explicit NumPy sums on the toy population; `from_vector`, `get`, `to_raw_displacements` |
| `test_bounds.py`, `test_remainder_probe.py` | 15 + 4 | the manuscript bound helpers against closed forms; the probe `H`, absolute moments and angular residual against NumPy oracles on a polynomial kernel; the covering inequality `|predict - direct| <= basis_remainder` |
| `test_basis.py`, `test_basis_build.py`, `test_basis_convergence.py` | 5 + 5 + 8 | column algebra exact on the polynomial kernel; benchmark coefficients; refusal rules on both sides (`required_m_max`, smoothness, `L_mu > 0`, phase degree); the per-column `numerical` envelope `(4 n_ch, n_real)` |
| `test_screens.py` | 42 | `LaplaceScreen` and `GammaScreen` weights against SciPy quadrature of the screen densities (Gamma shapes 0.7, 4, 25, both signs); the closed forms and the residual position angle of `eq: illustrated screens`; forced records; the `direct_channel_average` route and the `build_basis`/`predict` route (`depth_degree = 0`) against a per-atom NumPy channel sum to 1e-12, with `screen_exponent` a zero `bound` and the forced assumptions `unbounded` |
| `test_predict.py`, `test_predict_routes.py` | 8 + 3 | `predict` equals `direct_channel_average` and a NumPy oracle to 1e-12 on the polynomial kernel; every budget slot's kind; `propagate`; `filter_jit` and `grad` |
| `test_adapters.py` | 5 | `to_joint_faraday` reproduces `predict` through `faraday.joint_faraday_average` (< 1e-12 with 1e-13-relative-width channels); Burn sign pin; both measured remainders of `fixed_harmonic_comparison` |
| `test_jit_ad.py` | 6 | one trace per static configuration; `grad` with respect to `m0`, closure hyper-parameters and `gamma0` against Richardson differences; `vmap` over a batch of moments |
| `test_docstrings.py` | 115 | every public module of `syncmoments.model` and `syncmoments.model.fit` states `LABEL`, units, shapes and what it does not certify; every public class and function has its own docstring, a manuscript label (classes) and a limits statement or a pointer to the module docstring that has one; `docs/DESIGN.md` names every keyword of `predict`, `direct_channel_average`, `build_basis`, `basis_convergence` and the three fit routes, neither this file nor the README repeats a statement superseded by the round-2 fixes, and the README shows `assumption_allowances` with `screen_factorisation_bound` (textual checks only) |

Regression tests of the review fixes (each written to fail before its fix).
The last three files and the fitted-hyper and `cross_route` tests of the
assumption and basis-convergence files are from the second review round:

| file | tests | establishes |
|---|---|---|
| `test_budget_regressions.py` | 8 | basis-forced assumptions stay `unbounded` against a measured zero; records after the first are `unbounded`; the amplitude cross term covers `delta N_src` times the per-electron error; an explicit `statistical_input` adds to an unattributed discrepancy; nonsmooth line-kernel `basis_remainder` is `unbounded`; the per-column `numerical` contraction |
| `test_bounds_regressions.py` | 10 | nonsmooth line bases unbounded inside `RemainderInputs`; per-line `screen_factorisation_bound` covers a two-line channel; the phased depth coefficients give an `estimate` and `absolute_depth_coefficients` a covering `bound`; no depth-Taylor term on screen routes; the probe refused outside `certified_orders` |
| `test_linear_regressions.py` | 13 | `bias_bound` unbounded without `response_uncertainty` and when rank-deficient, responsive to a declared `delta R`; rank and solution invariant under column rescaling from 1e-100 to 1e100; `tol` and the inflated weights reach the identifiability report; the moment bias enters `statistical_input` |
| `test_assumptions_regressions.py` | 14 | `assume()` drops an allowance when another map adds a constraint; a fitted closure's record is a `theta`-independent `"(fitted)"` placeholder (eager, jit, `lax.cond`), so two fits share one pytree structure (`tree_map`, `jnp.stack`, one `filter_jit` trace); non-fitted closures keep their constructor values; the `fully_independent` count formula |
| `test_nonlinear_regressions.py` | 13 | `fit_bfgs` from `z0 = 0` on a fitted Gaussian screen returns a finite point; out-of-domain trial points get `+inf` |
| `test_support_check_regressions.py` | 16 | samples outside the `Support` void the zero harmonic-tail bound (both variables, `1e-9` relative outside and far outside) and a contradicted `truncated=False` tail; samples on the edges and zero-weight samples keep it |
| `test_product_cells_regressions.py` | 7 | derivatives stable across a line/channel-edge coincidence (64 vs 128 nodes; `jacfwd` to second order at `m_max = 40`); both node motions and the dispatcher at `lo/hi` from 2e-6 through the switch 0.1 to 0.5; reverse-mode memory within twice forward mode |
| `test_basis_convergence_regressions.py` | 11 | the `numerical` envelope's shape, invariance under `Reference.scales`, covering of the 64-vs-16-node change, the named route check, the continuum tail on the mass column; the default build does exactly one extra build, the route check runs only on request (`cross_route=True` or `"full"`) and leaves `numerical` bitwise unchanged, `cross_route` validation |
| `test_predict_direct_regressions.py` | 16 | `predict(samples=...)` runs the Support checks of the harmonic tail and of the excluded tail (same kinds as `direct_channel_average` outside the Support, zero bounds inside); `N = 3` leaves `basis_remainder` `unbounded` instead of raising; `assumption_allowances` (a `screen_factorisation_bound` closes a two-ray budget and bounds the error, the same input in `direct_channel_average`, a declared `isotropic_pitch` allowance, three invalid inputs); the direct amplitude cross term; a concrete zero `amplitude_uncertainty` is a zero `bound`, negative values raise (also traced); the sample probe uses `basis.phase` |
| `test_remainder_probe_route.py` | 38 | `from_samples(phase=None)` probes the basis route (harmonic kernel with a Gaussian screen; differs from the per-emitter route; the `predict(samples)` path; a stub basis without `phase` keeps the exact phase); no module, class or method docstring outside the listed modules calls a finite check "certified" (30 modules) |
| `test_nonlinear_bias_regressions.py` | 4 | BFGS and linear fits agree that an undeclared bias is `unbounded`; a declared discrepancy gives the linearised BFGS bias against NumPy `|pinv(J_fd)| delta`; when `m` is affine in `z` the bias covers the moment error of data shifted by the discrepancy; nodal `statistical_input` is `unbounded` with or without a discrepancy |

### 12.3 Assumption representation (`test_assumptions.py`, 112 tests; `test_assumptions_jit.py`, 10; `test_predict_assumptions.py`, 3)

Free counts 153, 115, 113, 88, 75, 57, 12 at `(2, 2, 2)`; product
populations reproduce the joint tensor to 1e-13 with `constraint_residual`
zero and measured discrepancy zero; on a correlated 9-atom population with
the harmonic kernel (`m_max = required_m_max = 18`) the reduced prediction
differs from the joint one by more than 1e-3 relative for
`independent_screen` and `field_independent`,
`|predict(m_fac) - predict(m_joint)| <= A |C| Delta` holds to 1e-10 relative,
`|predict(m_fac) - direct| <= assumption + basis_remainder` holds with the
probe `H`, the name appears in `to_dict()`, and without
`discrepancy="measured"` the term and the total are `unbounded`.
`ContinuumKernel` with `L_mu > 0` raises and the provenance lists
`isotropic_pitch`. `test_assumptions_jit.py` checks `ParameterMap.flatten`
and `unflatten` under plain `jax.jit` for maps with complex (`e^{2i phi}`)
free tables against the eager NumPy layout (regression for a traced
boolean index). In README example (b) the `isotropic_pitch` term is
`unbounded`: the continuum kernel forces the assumption, its columns vanish
on every `l >= 1` row, and the measured `|m_joint - m_fac|` on the retained
`L_mu = 0` rows is zero by construction, so `predict` does not propagate it
(Section 7.3). A pitch-dependent population is not seen by any measured
term; that dependence stays in the unbounded `physical_kernel` and this
unbounded assumption term, unless the caller supplies an allowance
(README example (d) declares a zero for a population uniform in `mu` by
construction).

### 12.4 Fit

| file | tests | establishes |
|---|---|---|
| `test_observation.py` | 48 | masking, whitening against `numpy.linalg.cholesky`, response handling, discrepancy propagation by `abs(R)`, `dof`, extremes (variances 1e-300..1e300, correlation 0.999999) |
| `test_linear.py`, `test_linear_policies.py` | 8 + 6 | zero-noise recovery to 1e-10 on the full-rank `isotropic_pitch` configuration; `fisher == G^T Sigma^-1 G`; nonlinear Fisher against central differences; 3-sigma coverage over 200 seeds; rank-deficient designs report `covariance is None` and labelled null vectors; `tol` on both sides of a singular value; the bias bound contains the bias of a violating population; `fit_linear == fit_bfgs` to 1e-8 on an affine map |
| `test_nonlinear.py`, `test_nonlinear_fit.py`, `test_nonlinear_boundaries.py` | 16 + 13 + 5 | `Transform` round trips and `log_det`; `grad(LogDensity)` against finite differences; BFGS recovery at zero noise (14 and 50 parameters); the 2000-parameter cap on both sides; nodal recovery of true weights and feasibility at 4100 nodes; logits +/-40 |
| `test_result.py` | 9 | `FitResult` wrapping of BFGS and nodal outcomes with Gauss-Newton Fisher matrices; BFGS `bias_bound` and `statistical_input` `unbounded` without a declared discrepancy; the fitted-hyper provenance note and the `"(fitted)"` closure kind; strict-JSON round trips (NaN placeholders as `null`) with complex tables |
| `test_diagnostics.py`, `test_feasibility.py` | 12 + 9 | singular values against `numpy.linalg.svd` of the whitened design; null vectors leave `A C m` unchanged; crafted infeasible vectors fail exactly the intended necessary condition; margins equal explicit sample sums |
| `test_fit_integration.py`, `test_fit_integration_nonlinear.py` | 9 + 6 | end-to-end synthetic fits on the harmonic kernel (16 channels, `Truncation(1, 1, 1)`, correlated 12-atom population): rank-deficient without `V` data, full rank with all Stokes, recovery within Fisher errors, Gaussian coverage, discrepancy policies with a violating population; `independent_screen` refused by `fit_linear` and fitted by `fit_bfgs`, `field_independent` by BFGS from the projected least-squares solution, `fit_nodal` recovering the atom weights, and the `FitResult` wrappers with the Fisher matrix in `z` and the nodal discretisation term |

### 12.5 Boundary sweeps (`test_boundaries*.py`, 187 tests, 18 of them `slow`)

Collected at 12:36 on 2026-09-24: `test_boundaries.py` 79 (none `slow`),
`test_boundaries_quadrature.py` 14, `test_boundaries_corners.py` 22 (16
`slow`), `test_boundaries_continuum.py` 13 (1 `slow`),
`test_boundaries_depth.py` 40 (1 `slow`), `test_boundaries_fit.py` 19.
Shared oracles are in `tests/model/_boundary_oracles.py`.

Both sides of every dispatch are evaluated directly, never through the
dispatcher alone. `test_boundaries.py`: Bessel resolution classes at the
class edges and the resolution guard; `required_m_max +/- 1` on
`channel_modes` and through `build_basis`; `m_max` extremes against a SciPy
sum; the Galactic regime refused even with `allow_truncated=True`; parity
on and off; the corner grid `gamma0 x B0 x mu x eta` of the line powers
(finite, `rel_err < 1e-3` between methods, no blow-up beyond 1e6 times the
reference, four corner cells pinned to 32-digit mpmath literals). `test_boundaries_corners.py`:
the product-cell grid `gamma0 in {1.01, 1.1, 2, 5, 20, 50} x B0 in {1e-2, 1,
10, 100}` G, product cells at two resolutions (`rel_err < 1e-3` per live
pair), the tensor route recorded (it degrades to 2e-3..3.5e-3 at
`gamma >= 20`), and the `(B0'/B0)^2` scaling along the `B0` axis. By
default it runs one cell per `gamma0` at the `B0` extremes `{1e-2, 100}`
with the channel at 10 `nu_B`. Only under `-m slow` do the interior
`B0 in {1, 10}` cells (12 tests) and the channel at 90 `nu_B`
(`required_m_max` 170..297, harmonics up to `m = 300`; 4 tests) run.
`test_boundaries_quadrature.py`: product versus tensor quadrature at
channel widths `{0.05, 0.2, 0.65}` (product converged between 48 and 96
outer nodes; tensor degradation recorded), channel support edges through
derivative order `N + 1 = 4` for every family on both sides, the smoothness
dispatch of `build_basis`, derivatives through a line on a support edge and
through an empty-cell transition, `gaussian` channels at `support_sigma in
{4, 6, 8}`. `test_boundaries_continuum.py`: the continuum `x_min` guard on
both sides with mpmath-literal pins of `F` and `G`, `eta -> +/-1`, and the
harmonic-versus-continuum comparison at `gamma in {10, 20, 50}` (`50` under
`-m slow`) and against pitch-averaged SciPy lines at `gamma = 20`. These are
the `E_phys` finite check, not a bound: the signed relative differences
(I, P) are recorded and pinned to 1e-3 absolute as a drift guard (measured
-1.8 %/-5.5 % at `gamma = 10`, -2.4 %/-3.7 % at 20, -0.6 %/-1.9 % at 50,
-2.5 %/-2.4 % for the pitch-averaged cell; the 24-vs-48-cell change is at
most 9e-5).

`test_boundaries_depth.py` sweeps the phase routes: `depth_ref in {0, 1e2,
1e4}` rad/m^2 (reference phases up to 9e10 rad), `tau Delta_depth in
{1e-3, 1, 10, 100}` across `depth_degree in {0, 1, 2, 4}` (the measured
Taylor error grows as `|tau|^{N+1}` and is covered by the `eq: local
response remainder` bound and by the package's `basis_remainder` in both
index layouts where the probe order is validated; `GaussianScreen` stays
exact at `tau sigma = 100` where the Taylor route diverges), and the
`N in {0..3}` x `L in {0, 4, 8}` grid (exact on the polynomial oracle;
shared columns agree across truncations on the harmonic kernel; the
`N = 3`, large-`L` shared-column case runs only under `-m slow`).
`test_boundaries_fit.py` covers the statistics side: the `is_affine`
dispatch (affine maps obey superposition and reproduce `affine_pieces`;
non-affine maps violate it and refuse `affine_pieces`), `noise -> 0` down
to variances of 1e-300 through whitening, the identifiability SVD and the
log density, weights of 1e300, a single sample, and softmax logits
`+/-40` and `+/-700` against normalised weights. The BFGS cap `2000 +/- 1`
is in `test_nonlinear_boundaries.py`.

### 12.6 Executed README examples (`scripts/model_examples.py`)

The README section "Finite joint response and spectral fits" quotes the
printed output of the script (examples (b) and (d) are in
`scripts/model_examples_continuum.py`, (c) in `scripts/model_examples_fit.py`).
Example (a) is the benchmark regime with `independent_screen` declared and
`discrepancy="measured"`; example (b) is the continuum kernel at
`gamma0 = 3000`, `B0 = 5` uG, bump channels between 0.1 and 3 GHz, with the
Burn screen route; example (c) is `fit_linear` on synthetic continuum data
with the identifiability and feasibility reports; example (d) passes
`assumption_allowances` on a continuum `EmpiricalScreen` route with two
correlated rays (`screen_factorisation_bound` for `independent_screen`, a
declared zero for `isotropic_pitch`). Rerun the script after any change to
`syncmoments.model` and paste the new output into the README. The run of
2026-09-24 12:29, after the second review round, took 6 min 20 s wall time
(a: 346.2 s, of which 96.4 s is the basis build with its one convergence
rebuild and 147.2 s the remainder probe; b: 20.7 s; c: 5.7 s; d: 5.4 s);
all 126 quoted output lines of the README section appear verbatim in it.
Against the first-round run, the only changed outputs are the timings, the
`amplitude` slot of example (a) with `amplitude_uncertainty=0` (now
`bound 0.000e+00`, previously `estimate 0.000e+00`) and the added
`statistical_input` kind in example (c). In example (d) the screen bound
covers the measured screen error in every channel (0.094 against 0.093 of
`I` in the top channel).

The first 0.3.0 run of 2026-09-25 with JAX 0.10.0 took
1 min 56 s wall time on a shared machine (a: 78.2 s, of which 18.8 s is the
basis build with its convergence rebuild and 34.5 s the remainder probe;
b: 23.4 s; c: 5.7 s; d: 6.6 s); all 126 quoted output lines appear verbatim
in it. Against the 0.2.0 run the changed outputs are the timings and the
two printed null-space vectors of example (c): the null space has three
dimensions with singular values at roundoff, so the SVD's basis inside it
is not determined, while the projector onto it agrees with 0.2.0 to
1.3e-15 and the response matrix is bitwise unchanged. The same script with
JAX 0.10.2 prints the same numbers except the timings, other null-space
vectors and the finite-response differences of example (a) in the ninth
digit (1.52748223e-06 against 1.52748222e-06). Peak RSS of the whole script
was 10.2 GB with JAX 0.10.0 and 29.6 GB with JAX 0.10.2; example (b) alone
took 6.7 GB and 27.0 GB, as in the 0.2.0 tree (6.4 GB and 26.7 GB).

After the round-2 fixes (weight edges, `transfer_slab` scaling, per-group
symmetry rule, continuum chunk budget) the script was rerun on 2026-09-25 in
both environments, alongside the slow suites, so its timings are not
comparable and the README keeps the earlier ones. Every printed line
equals the earlier run of the same environment except the timings and the
two null-space vectors of example (c), which changed between runs of the
same environment as well. Peak RSS of the whole script was 6.57 GB with
JAX 0.10.0 and 8.42 GB with JAX 0.10.2 (`/usr/bin/time -l`).

After the round-3 fixes (the `transfer_slab` JVP rule and refusals, the
shared weight normaliser in `mixed_moments`, the probe budgets and the
batched example script) the script was rerun on 2026-09-25 in both
environments, one run at a time on an otherwise idle machine; the README
quotes the JAX 0.10.0 run. Wall time 1 min 43 s with JAX 0.10.0 (a:
77.5 s, of which 18.2 s is the basis build with its convergence rebuild
and 35.1 s the remainder probe; b: 14.4 s; c: 5.6 s; d: 3.7 s) and
1 min 50 s with JAX 0.10.2 (a: 85.3 s; b: 13.2 s; c: 6.1 s; d: 4.1 s).
Peak RSS of the whole script 6.07 GB and 6.97 GB; examples (b) and (d)
alone peak lower (Section 6.2). All 126 quoted lines
appear verbatim in the JAX 0.10.0 output. The JAX 0.10.2 output differs
from them only in the timings, the null-space vectors of example (c) and
the eighth and ninth digits of the finite-response differences of
example (a) (5.14411831e-07 against 5.14411821e-07 in channel 1).

After the round-4 fixes the script was rerun once with JAX 0.10.0, while
both fast suites ran on the same machine: 112 s wall time and 5.97 GB peak
RSS. All 120 quoted lines other than the six timing lines appear verbatim
in its output, so the README keeps the round-3 figures. It was not rerun
with JAX 0.10.2.

After the round-5 fixes the script reads the manuscript's saved results
from the verbatim copy `tests/model/reference/validation` instead of the
manuscript repository two directories above the checkout (the two files
are byte-identical). It was rerun once per environment on 2026-09-25,
while both fast suites ran on the same machine: 114 s wall time and
6.11 GB peak RSS with JAX 0.10.0, 123 s and 6.66 GB with JAX 0.10.2. All
120 quoted lines other than the six timing lines appear verbatim in the
JAX 0.10.0 output. The JAX 0.10.2 output differs from them, as after
round 3, in the eighth and ninth digits of example (a)'s finite-response
differences and in the null-space vectors of example (c). The README
keeps the round-3 figures.

### 12.7 Test inventory

Collected on 2026-09-25 from the final 0.3.0 tree with
`PYTHONPATH=. python -m pytest --co -q` (and `-m slow` for the split); both
environments collect the same 5196 test IDs:

| scope | collected | `slow` | fast |
|---|---|---|---|
| `tests/model` | 2091 | 62 | 2029 |
| legacy tests (outside `tests/model`) | 3105 | 1573 | 1532 |
| total | 5196 | 1635 | 3561 |

The same command on the 0.2.0 export (commit `b39cf0c`, JAX 0.10.0)
collects 1400 tests: 1211 in `tests/model`, 42 of them `slow`, and 189
legacy tests. The 3796 added tests are 3790 in the 36 new files listed in
Section 12.10, three in `test_docstrings.py` (112 -> 115) and three in
`test_remainder_probe_route.py` (35 -> 38). The third review round added
222 of them (seven new files with 210 tests, and 12 in
`test_transfer_scaling.py`). The fourth round, with the reference
provenance test added between rounds, added 280: six new files with 269
tests (`test_transfer_higher_order.py` 32,
`test_transfer_companion_boundaries.py` 49,
`test_transfer_extreme_jacobians.py` 53,
`test_weight_normalisation_hessian.py` 70, `test_symmetry_fixed_table.py`
63, `test_reference_provenance.py` 2), 9 in `test_probe_memory_budget.py`
and 2 in `test_transfer_refusals.py`. The fifth round added 2347: seven
new files with 2076 tests (`test_transfer_batched_reverse.py` 134,
`test_transfer_eager.py` 13, `test_transfer_expm_boundaries.py` 83,
`test_ad_transform_matrix.py` 1705, `test_ad_transform_matrix_nested.py`
10, `test_ad_transform_matrix_v020.py` 53,
`test_weight_normalisation_second_order.py` 78), and 271 in the rewritten
`test_weight_normalisation_hessian.py` (70 -> 341). The helper modules
`tests/_ad_matrix.py`, `tests/_ad_matrix_cases.py`,
`tests/_ad_matrix_model_cases.py` and
`tests/test_weight_normalisation_oracle.py` hold no tests. After the fifth
round, `tests/test_transfer_scale_and_budget.py` (30 tests) was added with
the last two fixes (Section 12.10). The per-file
counts quoted in Sections 12.1 to 12.5 and 12.10 match this collection.

The 1635 `slow` tests are opt-in (`-m slow` or `SYNCMOMENTS_RUN_SLOW=1`,
gated by `tests/conftest.py`; a plain run reports them as skipped): 1555
cells of `test_ad_transform_matrix.py`, 18 Hessian pins in
`test_ad_transform_matrix_v020.py`, 24 full-resolution benchmark cases in
`test_benchmark.py` (16 x 16 latent and 256 x 256 angular nodes), 16 in
`test_boundaries_corners.py`, 11 in `test_probe_taylor.py`, 9 in
`test_b_analytic_derivatives.py`, 1 in `test_boundaries_continuum.py` and
1 in `test_boundaries_depth.py`. Section 12.10 gives the suite runs.

### 12.8 Known limitations

* `E_phys` and `E_tail` are never bounded by the package; they are inputs.
* The Galactic harmonic regime (`m ~ 1e12`) is refused, not computed; the
  continuum kernel there does not bound its discrepancy from the harmonic
  reference, and its measured assumption terms cannot see pitch-angle
  dependence (Section 12.3).
* Finite refinement checks and the refined-rule `numerical` envelope are
  `estimate` terms, not certificates; the remainder probe evaluates `H` at finitely
  many points and is an `estimate` unless envelopes are supplied.
* The order-`N+1` probe of `RemainderInputs.from_samples` evaluates the
  derivative tensor at every probe point; at `L = 8`, `N = 2` on the
  harmonic kernel this took about 146 s per point in 0.2.0 and takes about
  26 s for the first two points (compile included) and 7 s per further
  point in 0.3.0 (Section 8), so the README example still probes the
  reference point only, on a 16-atom discretisation.
* `statistical_input` accepts an `(n_real,)` array, a scalar (broadcast to
  every moment; `0` declares exact moments) or an `ErrorTerm` over `m`;
  other shapes raise `ValueError`.
* A screen-route or continuum prediction has an `unbounded` total unless
  every forced assumption gets a caller allowance
  (`assumption_allowances`); nothing checks that an allowance holds, and
  the inputs of `screen_factorisation_bound` (the ray standard deviation of
  the incident polarisation per line and `|Phi_R|`) are the caller's.
* With traced samples the Support checks cannot run: a declared-complete
  `excluded_tail` keeps its declared zero, and its note states that the
  Support hypothesis was not checked (shared by `predict` and
  `direct_channel_average`).
* `predict(samples=...)` repeats the refusal condition of
  `RemainderInputs.from_samples` (`N + 1` not in `certified_orders`) instead
  of catching its `ValueError`; the two must change together.
* The BFGS `bias_bound` is linearised at the optimum and leaves out prior
  and log-Jacobian curvature: an `estimate` whose linearisation error is not
  bounded (conservative with an informative prior). A nodal fit's
  `statistical_input` stays `unbounded` even with a declared discrepancy
  (the softmax gauge; a gauge-fixed pseudo-inverse is possible future work).
* `amplitude_uncertainty=None` stays `unbounded`; only an explicit concrete
  `0` declares the amplitude exact.
* The continuum examples still need more memory with JAX 0.10.2 than with
  0.10.0 (Section 6.2): example (b) peaks at 1.79 GB against 1.16 GB and
  example (d) at 1.12 GB against 0.85 GB.
* `syncmoments.transfer.transfer_slab` checks only its value. A derivative
  whose own value exceeds float64 overflows to inf without an error; this
  needs `eps` within a few orders of 1.8e308. A check on the tangent would
  be non-linear in the tangent and break transposition. For non-diagonal
  `K`, Jacobians in `S` and `eps` were tested up to `max|eps ds| = 1e307`,
  Jacobians in `K` and `ds` up to 1.7e308, second derivatives up to 1e306,
  and the value's derivative route at 1.7e308 in first order; `K = a I` up
  to 1.79e308. Third and higher derivatives were not tested at extreme
  scale.
* The value of `transfer_slab` carries the derivatives of `Phi S + G eps`
  (one extra 8x8 exponential, about twice the primal cost) once the
  exposure `k + max(0, -e(ds)) + max(0, e(max|K ds|))` reaches 896 in
  float64, and always in float32, whose cost was not measured. The
  threshold's margin (at least 54 binary orders below the first failure of
  the plain route) was measured on a grid of `|K|` in 1e-6..1e6, `ds` in
  1e-9..1e6 and `|K ds| <= 2e4`; outside that grid it is not established.
  Under `vmap` the choice is made for the whole batch (at every level of a
  nested `vmap`), and `transfer_los` makes it once per path, so one extreme member or slab doubles the primal
  cost of all. A user-written `lax.scan` over `transfer_slab` gets a
  `lax.cond` per slab, which cost 1.4x in `jit(grad)` in `eps` of a 64-slab path.
  `jit(grad)` in `K` of `transfer_los` measured 1.13 to 1.28x slower than
  before the round-4 fix, at high machine load and with the cause not isolated;
  compile time of `grad` and `hessian` grew about 1.6x, because
  `transfer_los` compiles two scans.
* `transfer_slab` and `transfer_los` raise on a non-finite result (0.2.0
  returned NaN or inf). Eager calls, whose cores are compiled once per
  shape with `equinox.filter_jit`, raise `EquinoxRuntimeError` with nothing
  on stderr; under an outer `jax.jit` the `equinox` error arrives as a
  `JaxRuntimeError` and a callback traceback is printed to stderr.
  The JVP rule makes derivatives with respect to `K` in `transfer_los`
  about 1.8 to 2.2 times slower than 0.2.0 at 64 slabs (`grad`, `jvp`,
  `hessian`; with `jax.scipy.linalg.expm` in the rule, 512 slabs gave
  `jacfwd` in `(eps, K)` 1.28 -> 3.35 s with JAX 0.10.0, not remeasured)
  and derivatives in `eps` alone faster. Reverse modes through a batched
  `vmap` in `K` are faster than 0.2.0 (`grad` of a sum over 16 rays 16 to
  18 ms against 32 to 33 ms).
* The rule's 8x8 exponential (`syncmoments._expm.expm_pade13`) uses
  `ceil(log2(|A|_1 / theta_13))` squarings; the value's 5x5 exponential is
  `jax.scipy.linalg.expm`, whose `floor` count applies Pade 13 up to
  `2 theta_13`. For a Faraday-dominated slab (`aI = 1e-3`, `rV = 1`,
  `rQ = 0.3`, `rU = -0.2`) the value's relative error against mpmath is
  9.6e-12 just below `|K ds| = 10.74` (1.7e-16 just above 5.37), and the
  `floor` method's error grows with the number of squarings (0.13 for a
  rotation-dominated 8x8 matrix at `|A|_1 = 1.15e10`); for the realistic
  rotation-dominated slabs of a `moment_driven_slab_cgs` spectrum
  (`n_e = 0.03 cm^-3`, `B_par = 3 uG`, `L = 1 kpc`, 50 MHz to 20 GHz) it is
  1.2e-10 to 4.4e-10. The value's exponential is unchanged since 0.2.0, so
  values stay bit-identical to earlier 0.3.0 states; computing it with
  `expm_pade13` would change them at the 1e-11 to 1e-10 level. To be
  addressed in 0.3.1.
* `d out / d ds` in ill-conditioned slabs, where the derivative is far
  smaller than the terms that cancel in it, is accurate only to their
  roundoff and is not flagged (relative error 0.2 to 56 against mpmath on 4
  of 120 random slabs; 0.2.0: 1.5e-4 to 110). To be addressed in 0.3.1.
* Third derivatives through `expm_pade13` were checked by probes (1e-16 to
  7e-14 against autodiff of `jax.scipy.linalg.expm` in four nestings) but
  are not in the test suite. Float32 reverse mode for small sources is
  tested (`tests/test_transfer_scale_and_budget.py`); `expm_pade13` uses the
  float64 `theta_13` in float32 too, so its float32 squaring count differs
  from `expm`'s.
* Weight normalisation on XLA CPU flushes to zero every weight whose ratio
  to the largest is below `2^-1022` (at most `n 2^-1022` of the mass).
* Second derivatives with respect to the weights are not refused.
  `w / sum(w)` (`rm._divide_by_sum`) has the custom JVP `rm._sum_tangent`,
  which repeats the plain division JVP's operations (values and first
  derivatives bit-identical to the round-4 state) and whose own JVP is the
  closed form `d2p[dr, y] = -((dr - p dR) Y + (y - p Y) dR) / R^2`, a rank-2
  form evaluated on `dr 2^-k` and `R 2^s`, summed and scaled once by the
  exact `2^(k + 2s)`. Log-weight Hessians are exact in every mode down to
  `max(w)` of about 1e-307; Hessians in `w` for `max(w) >= 2^-958` are exact
  or inf with the exact sign except under `jacfwd` over `jacfwd`. A refusal
  cannot separate the remaining cases from exact results, because a
  primal-only criterion would also refuse the log-weight Hessians at the
  same weights. Remaining behaviour, to be addressed in 0.3.1:
  - `max(w) < 2^-958`: every entry of a Hessian in `w` along unit
    directions exceeds float64; `jax.hessian`, `jacrev` over `jacfwd`,
    `jacfwd` over `jacfwd` and Hessian-vector products give NaN, and
    `jacrev` over `jacrev` gives infinities of the wrong sign in 10 of 16
    entries of `mixed_moments` (1e-300 and `2^-1022`) and 3 of 16 for a
    `JointMoments.from_samples` moment under jit. In the transposed
    division, the tangent of `R` forms `-(Y ct_j) / R^2` and `+2 m Y / R^2`
    as separate products that overflow before they cancel. A gauge
    reformulation (`r' = r K / R`, `K = stop_gradient(R)`) turns them into
    NaN but changes JAX 0.10.2 forward first-derivative rounding in 86 of
    1993 compared arrays.
  - `jacfwd` over `jacfwd` for `sum(w) < 2^-512`: the materialised second
    tangent of `p` has entries `(2 p_i - delta_ia - delta_ib) / W^2`,
    infinities of both signs that sum to zero, so any consumer's
    contraction `sum a_i d2p_i` is NaN, including consumers whose exact
    Hessian is representable only by cancellation (0 for a constant). A fix
    needs a second-order rule per consumer: for `G(p, args)`,
    `2^(k+2s) {G''[(x - pX)/Rb, (y - pY)/Rb] - G'[S/Rb^2]}` under one final
    scaling.
  - The RM variance under `jacfwd` over `jacfwd` with 16 weights at
    `max(w) = 2^-511` (exact entries up to `2^1022`) overflows in its own
    products `4 dmu_a dmu_b`: 31 inf, 2 of the wrong sign, 1 NaN, no finite
    wrong value. At 1e-200 a consumer with constant values (exact Hessian 0)
    gives `-inf` in `jax.hessian`, inside the documented criterion because
    its rounding bound `64 eps sum|terms|` itself exceeds float64.
  - A Hessian-vector product at `max(w) = 2^-1022` along `|v| = 2^-1033`
    loses accuracy (a reverse-mode intermediate is subnormal and flushed).
  The Hessian at `w` is `lambda^2` times that at `lambda w`, so rescaled
  weights avoid all of these.
* JAX 0.10.0 and 0.10.2 fail (`broadcast_in_dim ... operand ndim`) under
  `jax.hessian` of a sum over `vmap` of a jitted function when a constant
  such as `jnp.ones_like(x)` is passed into a `custom_jvp` inside another
  `custom_jvp`'s primal. The weight normaliser avoids the pattern; the
  minimal reproduction is in the T-004 round-5 record, not in the
  repository.
* With several invalid fields at once, the `PopulationSamples` constructor
  reports the first failing check of one compiled call, which can differ
  from the field order of 0.2.0.
* For a capped truncation `predict(samples=...)` gates the probe on `N + 1`
  in `certified_orders`, while `from_samples` checks every margin order
  `|beta| <= N + 1`; the gate is conservative. The channel smoothness
  requirement `N + 1` is also kept for capped truncations, although the
  margin can need fewer derivatives.
* At `N = 3` the analytic basis build peaks at 3.4 GB with JAX 0.10.2, most
  of it XLA compile memory, which the chunk budget does not bound.
* The nested-`jacfwd` probe of a `derivatives="autodiff"` harmonic kernel
  keeps the `4^q` budget division and is about 1.7 times slower than
  undivided (Section 8). `tangent_divisor` recognises the analytic
  harmonic kernel by `kernel.name == "harmonic"` and
  `kernel.derivatives == "analytic"`; a kernel-side hook would be cleaner.
* Two `fixed_table` closures under a symmetry whose tables differ only in
  dropped entries give equal moments but different records, so `assume`
  treats them as conflicting (Section 7.4).
* The AD-transform matrix (Section 12.10) leaves out the private
  `model.basis._build_core` (its derivatives are tested in
  `test_b_analytic_derivatives.py` and `test_basis_convergence.py`) and
  the private Taylor-rule path of `_bessel_recurrence`
  (`derivative_coefficients`, `taylor_neighbours`), and covers the normal
  floating-point range only. Its fast tier takes about 115 s with both
  environments running at once, close to a two-minute target. The script
  that computed its 0.2.0 pins is not in the repository.

### 12.9 Docstring conventions

Every public module docstring names its manuscript labels (`LABEL`),
units, shapes and what it does not certify; every public class carries a
`LABEL` attribute or a label in its docstring (`[extension]` for
additions), and every public docstring states its assumptions and limits
or points to the module docstring that does. `tests/model/test_docstrings.py`
checks these statements textually over 18 modules and 83 public objects
(it does not check that the statements are correct; the numerical tests
do that). It also checks that this document names every keyword of the
documented `predict`, `direct_channel_average`, `build_basis`,
`basis_convergence` and fit signatures, and that neither this document nor
the README repeats a statement superseded in the second review round.
Finite numerical checks are called "validated" (finite checks, not
certificates) in the docstrings; `test_remainder_probe_route.py` scans them
for positive "certif*" claims.

### 12.10 0.3.0 acceptance (task T-004)

Two environments, run from the worktree with `PYTHONPATH=.`:

| name | versions |
|---|---|
| JAX 0.10.0 | JAX 0.10.0, Equinox 0.13.7, NumPy 2.3.5, SciPy 1.16.3 |
| JAX 0.10.2 | JAX 0.10.2, Equinox 0.13.8, NumPy 2.5.3, SciPy 1.18.1, no mpmath |

At 0.2.0 the JAX 0.10.2 environment failed 7 tests and skipped 10: the five
extreme-weight tests (`5e307 / 1e308` evaluates to 0 there; see the
changelog), the exact-zero gap of `required_m_max +/- 1` (3.85e-34),
the reverse-mode memory test (an absolute 16 GB cap against 25 GB), and 10
skips for the absent mpmath. In 0.3.0 the weights are normalised through
an exact power of two, the gap test uses the summation tolerance
`2 gamma_K sum |x_i|` (`K = n_nodes + m_max`, Higham eq. 4.4; an estimate,
since `sum |x_i|` assumes well-conditioned Bessel sums), and the memory test
checks `grad <= 2 jacfwd` together with an absolute cap of 2e8 bytes of XLA
temporaries (measured 3.3e7 to 4.0e7 bytes in both environments). The
fit-integration covariance test compares with `pinv(F)` and
`pinv(G_w) pinv(G_w)^T` in the 2-norm, with tolerance
`2 cond(X) eps ||oracle||` per oracle; the constant 2 is measured (misfit
ratios 0.05 to 0.24), not a strict bound, and a relative perturbation of
1e-6 gives ratios of 10.6 or more.

New test files (tests collected; `slow` in parentheses):

| file | tests | establishes |
|---|---|---|
| `tests/test_weight_normalisation.py` | 48 | `w / max(w)` to one ulp and bit-identical to the 0.2.0 division in the normal range; public averages at maximum weights 4e307, 1e308, 1.79e308, 1e-300 and `2^-1022` (whose subnormal companions keep their ratios), eager and jit; gradients and Hessian against analytic oracles; refusal of invalid weights including `-5e-324` |
| `tests/test_weight_normalisation_edges.py` | 50 | a subnormal largest weight refused through four entry points, eager and jit, and under `grad`, `jacfwd`, `jit(jacfwd)`, `hessian` and `jvp`; forward derivatives at `max(w) = 2^-1022` against the oracle, including a unit tangent on every weight; float16, bfloat16 and float32 weights against a float64 oracle (4e-16), with the samples' dtype kept; float8 refused; `[-5e-324, 1]` refused under jit; the XLA CPU flush thresholds (skipped off CPU) |
| `tests/test_transfer_scaling.py` | 36 | `transfer_slab` at `max|eps ds|` in {4e307, 4.6e307, 1e308, 1.79e308}, eager and jit, against `S + eps ds` (`K = 0`, 4e-16) and the analytic slab (`K = I`, 4e-15); `eps ds` of 1.5e308 reached through `ds`; the documented flush of small entries; `jacrev` and `jacfwd` to 1.79e308 against `(1 - e^-a)/a I`, `a` in {0, 2, 50}; bit-identical to 0.2.0 for `max|eps ds| < 1` |
| `tests/test_transfer_derivatives.py` | 93 | the JVP rule of `transfer_slab`: for a thick and a weak non-diagonal `K` at `max|eps ds|` from 1e297 to 1e307, eager and jit, `jacfwd` and `jacrev` in `eps` equal `G` and in `S` equal `Phi` (SciPy `expm` of the 8x8 block, 1e-13 of the largest entry); `d out/d ds = Phi (eps - K S)`; Hessians in all 25 inputs finite, symmetric, equal across forward-over-reverse, reverse-over-forward and forward-over-forward, with the `eps`-`K` block equal to SciPy `expm_frechet`; plain autodiff of the primal reproduced to 1e-13 in the normal range; the primal bit-identical to the plain implementation (float64 and float32, eager and jit); `transfer_los` in both modes, jit and vmap, at scales 1 and 1e306 |
| `tests/test_transfer_refusals.py` | 26 | eight non-finite cases refused with their cause (`eps ds` overflow, `max_squarings` exhausted, `K = 1e12 I`, gain `-800 I`, accumulation `S = 1.5e308`, `eps = 1e308` through an absorbing `K = 1e-3 I`, NaN in `K`, inf in `S`, NaN `ds`), eager and jit, the overflow cases with the message "the result overflowed float64 (accumulated intensity ..."; `jacfwd` and `jacrev` of refused slabs, including `jit(jacrev)`; vmap with one bad member; `transfer_los` refuses inside the scan, also a 3-slab path with `eps = 1e308` through absorbing slabs; `max|eps ds| = 1.79e308`, optical depth 1e6 and gain `-700 I` are not refused; exception types pinned (`EquinoxRuntimeError` eager, `JaxRuntimeError` under jit) |
| `tests/test_transfer_higher_order.py` | 32 | second derivatives through the value: `jacfwd` of the `jvp` primal for a thick (`ds = 20`) and a weak (`ds = 1`) `K` at scales 1, 1e297, 1e303, 1e306, eager and jit; two chained slabs in all four second-order nestings and their full Hessian's symmetry; `transfer_los` with `jacfwd` over the `vjp` primal, the `linearize` primal and `jacrev` in `K`, eager and jit, on normal, extreme and mixed paths; a 3-slab path's Hessian in four orders against the 8x8 reference, and symmetric; all against plain autodiff of `Phi S + G eps` (1e-13; measured at most 6e-16); the primal bit-identical over 300 random slabs under jit and 60 eager, sources 1e-300 to 3e307, also as a `jvp` primal; `-0.0` kept; a vmap batch mixing normal and extreme rays |
| `tests/test_transfer_companion_boundaries.py` | 49 | boundary validation of the value's derivative dispatcher: threshold pinned (896, and 0 in float32); the plain and companion methods called directly agree with each other and with the reference at `T - 8`, `T - 1`, `T`, `T + 1`, `T + 8` for thick and weak `K` at `ds` 1 and 20; at sources 1, 1e-300, 1e250, 1e297, 1e306 and 1.7e308 the dispatcher's route is exact and finite in forward and reverse mode, and its derivatives equal those of the chosen method bit for bit; the per-path routing of `transfer_los` at `T - 1` and `T`; the plain method fails at 1e306 (regression) |
| `tests/test_transfer_extreme_jacobians.py` | 53 | `jacfwd` and `jacrev` in `K`, eager and jit, for thick and weak `K` at `ds` 1 and 20 with `max|eps ds|` 1e308 and 1.7e308, against SciPy `expm_frechet` (1e-13; measured at most 3e-15); the `dPhi S` term with `|S| = 1.5e308`; `d out/d ds` near 1.7e308 in both modes finite and against `max(|eps|, |K| |out|)`; `_split_exponent` pinned (0 up to 1e154, 512 at 1.7e308) |
| `tests/test_weight_normalisation_mixed.py` | 23 | `mixed_moments` at `max(w) = 2^-1022`: a unit tangent on every weight gives an exact zero tangent, eager and jit; non-uniform tangents, `grad`, `jacfwd` and `jacrev` against the oracle `dE[g]/dw_k = (g_k - E[g]) / W`; the Hessian at `max(w)` in {1e-150, 1, 1e150} against `-(g_k + g_l - 2 E[g]) / W^2` |
| `tests/test_weight_normalisation_hessian.py` | 341 | against the exact `Fraction` oracle of `test_weight_normalisation_oracle.py` (`H = J^T F'' J + sum F'_i Hp_i`, softmax form for log-weights; cross-checked against float64 division and `jax.nn.softmax` at unit scale) for `mixed_moments`, the RM variance and a `JointMoments.from_samples` moment: log-weight Hessians at shifts 0, -300, -690, -700, -708 in five modes (`jax.hessian`, `jacfwd`-`jacfwd`, `jacrev`-`jacrev`, `jacrev`-`jacfwd`, Hessian-vector products), eager and jit; Hessians in `w` at `max(w)` in {1, 1e-100, `2^-511`, 1e-160, 1e-200} exact within `64 eps` times the sum of the term magnitudes, or inf with the exact sign, in every mode but `jacfwd`-`jacfwd`; `jacfwd`-`jacfwd` exact while `sum(w) >= 2^-512` and never finite below; unit-direction Hessians for `max(w) < 2^-958` never finite; 16 weights at `2^-511` and `2^-509` |
| `tests/test_weight_normalisation_second_order.py` | 78 | the closed-form rule of `rm._sum_tangent`: primal, `jvp` in four directions, `vjp`, `jacfwd` and `jacrev` bit-identical to the round-4 formulation for seven weight cases, eager and jit; the forward second tangent of `w / W` exact or signed inf from 1 to `2^-1022`; boundary validation of the two second-order routes (closed form and the transposed division) at `2^-958 2^{-2..2}`, agreeing within 1e-13 and with the oracle, and at extremes `n` in {2, 64}, ratios down to `2^-400`, `log2 max(w)` in {0, -700, -960, -1015}; `n = 1` exactly 0 in every mode; small-direction Hessian-vector products in the headroom region; log-weight Hessians under `vmap` in four modes and of a sum over `vmap` (block diagonal, off-diagonal blocks exactly 0); third derivatives in four nestings against the plain division; float32 weights |
| `tests/test_weight_normalisation_errors.py` | 55 | eager refusals of four kinds of invalid weights through nine entry points and of non-finite samples through six raise `EquinoxRuntimeError` (not a `JaxRuntimeError`) with nothing on stderr and no callback log; `PopulationSamples` value refusals eager and under `filter_jit`; product weights bit-identical to per-marginal normalisation, masses 1e300 and 1e-300 included |
| `tests/test_transfer_batched_reverse.py` | 134 | `grad`, `vjp`, `jacrev`, `hessian` and Hessian-vector products of a sum over `vmap` in `K`, in `ds` and in both, for `transfer_slab` and `transfer_los` (per-ray `K`, per-slab `ds`), eager and jit, at source scales 1 and 1e306, against forward-mode autodiff of the 8x8 `jax.scipy.linalg.expm` reference (1e-13 of the largest entry per block); `grad` of the sum over `vmap` equal to `vmap` of `grad`; projections pinned to 0.2.0 (rtol 1e-12); a spectrum of `moment_driven_slab` over 8 frequencies, `grad` and Hessian in `(M0, L)` against `jacfwd` and 0.2.0 |
| `tests/test_transfer_scale_and_budget.py` | 30 | reverse- and forward-mode derivatives in `K` for small sources against a float64 forward-mode reference: float32 at `ds = 20` (sources 1 to 1e-20) and `ds = 1` (1e-18 to 1e-25), float64 down to 1e-290; `max_squarings` given as numpy integers or integral floats (eager, `jit`, `grad`, `transfer_los`) and refused when non-integral, negative, a string or `None` |
| `tests/test_transfer_eager.py` | 13 | a second eager call with the same shapes compiles nothing (`jax_log_compiles`), for `transfer_slab` at scales 1 and 1e306 and with `max_squarings=40`, `transfer_los` with scalar and per-slab `ds`, eager `grad` in `K`, `moment_driven_slab` and `moment_driven_slab_cgs`; eager refusals raise `EquinoxRuntimeError` with nothing on stderr and no error log |
| `tests/test_transfer_expm_boundaries.py` | 83 | boundary validation of `expm_pade13`: `n` and `n + 1` squarings evaluated directly at `theta_13 2^m (1 - 1e-9, 1, 1 + 1e-9)`, `m` in {0, 1, 3, 10, 20, 31}, for thick, weak and rotation-dominated `K`, against each other and SciPy within `1e-14 2^m`; the Frechet derivative against SciPy `expm_frechet` at the thresholds; `|K ds|` down to 1e-300; the `floor` count's error just below the next threshold pinned (regression); the `max_squarings` NaN threshold equal to `jax.scipy.linalg.expm`'s at budgets 0, 3 and 5; a `vmap` batch needing 0, 5, 20 and 31 squarings equal per member to 1e-15 in value and `grad` of the sum |
| `tests/test_ad_transform_matrix.py` | 1705 (1555) | 19 entry-point families and 63 argument groups under 27 transforms (1701 cells), each against jitted forward-mode autodiff of an independent plain-`jnp` reference (`tests/_ad_matrix_cases.py`, `tests/_ad_matrix_model_cases.py`) within 1e-12 of the largest entry per block, all entries finite; the fast tier (150) runs every transform on five groups and the batched-reverse transforms on `transfer_los`, `moment_driven_slab` and `harmonic_lines`; three self-tests of the harness (a rule wrong only at second order is flagged; a `stop_gradient` tangent raises in reverse mode; the block-scaled comparison) |
| `tests/test_ad_transform_matrix_nested.py` | 10 | the `lax.cond` on `any_member` stays a `cond` in the jaxpr under two and three `vmap` levels; the helpers' values under every batching pattern; `grad` and `hessian` of sums over double and triple `vmap` of `transfer_slab` with members needing 0 to 9 squarings, eager and jit (1e-12); `K` batched at the outer or inner level only; nested `vmap` of `transfer_los` with a member on the companion route at 1e306 (values bit-identical, gradients within 1e-13) |
| `tests/test_ad_transform_matrix_v020.py` | 53 (18) | projections of the matrix references' values, Jacobians and Hessians and of the package's values and Jacobians against 0.2.0 (computed once with the 0.2.0 export aliased as `syncmoments`), within 1e-12 |
| `tests/test_bessel_autodiff_accuracy.py` | 4 | the docstring claim of `bessel_jn_neighbours`: third autodiff derivatives at `n = 1`, `x = 1e-6` off by more than 1, the recurrence within 1e-15 of SciPy through order 3 |
| `tests/model/test_probe_memory_budget.py` | 18 | `tangent_divisor` per kernel (`4^q` for the continuum, unknown and `"autodiff"` harmonic kernels, 1 for the analytic harmonic kernel and for `q = 0`); the nested-`jacfwd` probe and margin envelope run at `chunk_budget // 4^q` on the continuum kernel and at `// 4`, `// 16` on the `"autodiff"` harmonic kernel, and see the full budget on the analytic harmonic kernel, including a capped margin envelope; `angular_residual` caps its batch; blocked and unblocked probes agree to 1e-13 on the continuum and both harmonic routes; the refined continuum kernel keeps a user `chunk_budget`, and the convergence estimate does not depend on it |
| `tests/model/test_example_continuum_budget.py` | 2 | the example script's `incident_polarisation` maps in batches of `samples_per_step` and equals the full vmap to 1e-13 |
| `test_memory_numerics.py` | 26 | `bessel_jn_neighbours` against SciPy `jv`/`jvp` over the resolution classes (rtol 2e-12) and against `bessel_jn_and_prime`; AD derivatives against SciPy and Richardson differences; `chunk_plan` bounds; projections and their `jacfwd` invariant under blocking, at and one below the dispatch threshold; XLA temporaries of nested second-order `jacfwd` at the benchmark below 2e8 bytes and below half of the `derivatives="autodiff"` figure (measured 7.8e7 / 8.6e7 against 2.37e8 / 5.65e8 bytes, JAX 0.10.0 / 0.10.2), so a regression to tangents through the Bessel quadrature fails in both environments |
| `test_kernel_chunking.py` | 9 | `sample_batch` within the kernel budget; harmonic and continuum direct averages independent of the budget (1e-13 per Stokes block) with compiled temporaries at a small budget below 1/4 of an unbounded one; continuum projections, second `jacfwd` and basis columns independent of the budget; budget validation and provenance |
| `test_harmonic_doc_claims.py` | 4 | textual: the derivative claims of `_harmonic_taylor`, `harmonic` and `_harmonic_cells` (affine-follow cells and the edge-line coincidence; the `"autodiff"` rule is not bit-identical to 0.2.0) and the memory statement naming the sample-vmapped path |
| `test_bessel_recurrence.py` | 27 | the recurrence band against SciPy (5.7e-13 of the band maximum), derivatives to order 4 (2.1e-10), Taylor coefficients; a jaxpr check that no array under third-order `jacfwd` of `harmonic_lines` carries both the node axis and a tangent axis |
| `test_b_analytic_derivatives.py` | 21 (9) | `"analytic"` against `"autodiff"` per Stokes block and derivative order at the 12 `(gamma0, B0)` corners (channel centres off the edge-line coincidence), `N = 0..3`, both angular routes, the affine-cell coincidence, one edge-line coincidence case and the benchmark; outer `grad`/`jacfwd` in `gamma0` |
| `test_capped_derivatives.py` | 42 | capped columns equal the uncapped columns on the retained rows (rtol 1e-13); the direction plan has minimal cost for every cap in `{None, 0..3}^2` at `N = 1..4` |
| `test_lower_set_truncation.py` | 560 | retained rows against brute force for `N = 0..3` and every cap in `{None, 0, 1, 2}^3`; the margin identities and the counterexample of Section 8.1 |
| `test_lower_set_remainder.py` | 10 | `margin_H` and `margin_moments` against exact polynomial derivatives; `|predict - direct| <= basis_remainder` for capped bases in both layouts |
| `test_probe_taylor.py` | 20 (11) | the `angular_taylor` probe against the `jacfwd` probe (rtol 1e-12), margin envelopes, the symmetric Frobenius identity |
| `test_symmetry_assumptions.py` | 11 | pinned and derived free counts; exact zeros and equal predictions for symmetric populations; `V = 0` under field reversal; measured discrepancies of asymmetric populations; combinations, jit and `jacfwd` |
| `test_symmetry_factorisation.py` | 30 | the per-group symmetry rule of Section 7.4: pinned and derived counts for `fully_independent` with each symmetry and for `{eta} \| rest`; zero odd-`l` and odd-`k` rows in the map output; exact product populations reproduced to 1e-14 with Jacobian rank `n_free`; measured discrepancy `|m_joint|` on the dropped rows of an asymmetric population; the `fully_independent` count under caps |
| `test_symmetry_fixed_table.py` | 63 | a `fixed_table` under field reversal or pitch symmetry on `mu` and `eta` singletons, a joint `(mu, eta)` group and the partition `{mu} \| rest`, at `(1, 1, 0)`, `(3, 1, 1)`, `(2, 2, 2)` and `(8, 8, 2)`: `build` and `assume` give identical specifications, gathers, labels, records and bitwise-equal moments, which equal an oracle built without the symmetry from symmetrised tables (1e-15); nonzero dropped entries give bitwise the same moments; wrong shapes, including the reduced table, refused on both routes with the full shape named; jit and `affine_pieces`; a `gamma` table unchanged |
| `test_reference_provenance.py` | 2 | the saved benchmark results record the SHA-256 of the copied `full_response.py` and `full_response_product.py`; the copy lies inside the repository |
| `test_truncation_provenance.py` | 29 | `max_orders` in basis provenance, `Prediction.to_dict`, `JointMoments.to_dict` through strict JSON; rebuilt truncations equal the originals; the new exports |

Basis equality with 0.2.0 (every column within 1e-12 of the largest column
of its Stokes block, `Re` and `Im` of `P` separately): the benchmark
`(8, 8, 2)`, `(2, 2, 3)`, the tensor route `(4, 4, 2)`, the affine-cell
coincidence `gamma0 = 3 (1 + 1e-9)`, and the corners `gamma0 = 1.01,
B0 = 10`; `gamma0 = 50, B0 = 100, N = 3`; `gamma0 = 2, B0 = 1e-2, N = 3`;
`m_max = 300` give a worst difference of 5.2e-14 (JAX 0.10.0) and 6.8e-14
(JAX 0.10.2 against the 0.2.0 bases built with JAX 0.10.0). The manuscript
benchmark (`-m slow tests/model/test_benchmark.py`) reproduces the saved
finite predictions to 5.55e-10 of channel `I` and the second-order errors
1.644172e-4 and 1.753102e-3 in both environments.

Suite runs of 2026-09-25 on the finished tree after the round-3 fixes
(2539 tests collected, 62 of them `slow`; the two environments at once on
one machine, after the README example runs):

```
-m "not slow"  JAX 0.10.0: 2477 passed, 62 deselected, 2 warnings in 1392.16s
-m "not slow"  JAX 0.10.2: 2477 passed, 62 deselected, 2 warnings in 1446.20s
-m slow        JAX 0.10.0: 62 passed, 2477 deselected in 926.17s
-m slow        JAX 0.10.2: 62 passed, 2477 deselected in 924.50s
```

After the round-4 fixes, 2819 tests collected (62 `slow`), the two
environments at once on one machine; the fast runs alongside one run of
the README example script, the slow runs after them:

```
-m "not slow"  JAX 0.10.0: 2757 passed, 62 deselected, 2 warnings in 1654.70s
-m "not slow"  JAX 0.10.2: 2757 passed, 62 deselected, 2 warnings in 1713.59s
-m slow        JAX 0.10.0: 62 passed, 2757 deselected in 902.43s
-m slow        JAX 0.10.2: 62 passed, 2757 deselected in 896.42s
```

No test fails or is skipped in either environment. The two warnings are
NumPy overflow warnings inside `numpy.linalg` from
`test_nonlinear_regressions.py::test_fit_bfgs_from_zero_returns_on_fitted_gaussian_screen`
(two parametrisations). The benchmark numbers above are the `record_property`
values of the previous slow run (before round 2); these runs pass the same
pins. Lint with JAX 0.10.0's Python: `ruff check syncmoments tests scripts
docs/conf.py` and `black --check syncmoments tests scripts docs/conf.py`
pass. The documentation builds with `sphinx -b html -W --keep-going` under
Sphinx 8.2.3 and 9.1.0 without warnings (9.1.0 failed on three ambiguous
cross-references to `P` in the `CumulantExpansion` docstring, as in 0.2.0).
`scripts/review_radiation.py` regenerates `review-results/radiation.json`
with all 2401 numerical entries equal to the 0.2.0 record; only the
timestamp, the module names and the file hashes change.

Packaging changes made between the third and fourth review rounds:

* The manuscript's `validation/full_response.py`,
  `validation/full_response_product.py` and
  `validation/full_response_results.json` are copied byte for byte into
  `tests/model/reference/validation`, so the benchmark and projection
  tests run without the manuscript repository (they previously read an
  absolute path to it). `tests/model/reference/README.md` gives the SHA-256
  of each file and the update procedure; `test_reference_provenance.py`
  checks that the saved results record the hashes of the two copied
  scripts and that the copy lies inside the repository. ruff and black
  exclude the directory, and the copy must not be edited.
* `pyproject.toml` uses PEP 639 licence metadata (`license = "MIT"`,
  `license-files = ["LICENSE"]`) with `setuptools>=77`, replacing the
  TOML table that setuptools deprecates.
* `MANIFEST.in` adds `CHANGELOG.md`, `.readthedocs.yaml`, the tests
  (`*.py`, `*.json`, `*.md`, so `tests/model` and the reference copy), the
  scripts and the documentation sources to the sdist, without
  `docs/_build`. The round-3 sdist had only the top-level
  `tests/test_*.py` files.
* The inventory counts of Sections 12.1 to 12.7 and this section were
  rechecked against `pytest --co`.

Changes of the fifth review round (details and measurements in the
changelog, 0.3.0):

* `transfer_slab`'s derivative rule takes its 8x8 propagator exponential
  from `syncmoments/_expm.py::expm_pade13` (Pade 13, `ceil` squaring
  count, a squaring `lax.cond` on `any_member`, per-member selection by
  `batched_only`) instead of `jax.scipy.linalg.expm`. Reverse mode through
  `vmap` and nested `vmap` over batched `K` or `ds` works; it raised
  `NotImplementedError` (transpose of `stop_gradient`) after round 4.
  Values are bit-identical; derivatives moved toward mpmath, by up to
  1.9e-10 relative on random slabs and up to 5e-6 in rotation-dominated
  slabs that need many squarings (`|K ds|` about 80 to 2e4), where 0.2.0's
  derivatives erred by that much and the new ones by at most 9e-13. The
  propagators' Pade coefficients are held at the exact scale `2^-40`, so
  the transposed solve no longer divides cotangents by `b_0 = 6.5e16`
  (float32 reverse mode for small sources); `max_squarings` is coerced to a
  Python int before the compiled cores. A reparametrisation `K_s = sg(K) + (K - sg K) 2^a` was
  rejected because it scales every derivative order by `2^a` while the
  rule removes the factor once (Hessian error 1e3 at `max|eps ds| = 1e160`),
  and a materialised-Jacobian rule because it cost 1.3 to 3.6 times more.
* The public transfer cores are compiled with `equinox.filter_jit`: eager
  calls no longer recompile a `lax.cond`, and eager refusals of
  `transfer_los` raise `EquinoxRuntimeError`.
* The second-derivative refusal of the weight normaliser is removed in
  favour of the closed-form second tangent (Section 12.8); values and first
  derivatives are bit-identical to round 4 (1993 arrays, both
  environments).
* The AD-transform matrix (1701 cells) and its nested-`vmap` regression
  file; the `custom_vmap` rules of `any_member` and `batched_only` re-enter
  themselves so that every enclosing `vmap` level reduces or selects per
  member.
* `scripts/model_examples.py` reads the reference copy in
  `tests/model/reference` (Section 12.6).

After the round-5 fixes, 5166 tests collected (1635 `slow`), the two
environments at once on one machine, the fast runs alongside one run of
the README example script per environment, the slow runs after them:

```
-m "not slow"  JAX 0.10.0: 3531 passed, 1635 deselected, 2 warnings in 1889.05s
-m "not slow"  JAX 0.10.2: 3531 passed, 1635 deselected, 2 warnings in 1953.80s
-m slow        JAX 0.10.0: 1635 passed, 3531 deselected in 2259.91s
-m slow        JAX 0.10.2: 1635 passed, 3531 deselected in 2257.94s
```

No test fails or is skipped in either environment. The two warnings are
the NumPy overflow warnings named above. After the last two fixes (Section
12.10, round 5) the 20 files that reach the transfer code were rerun in both
environments on the final tree (5196 tests collected):

```
tests/test_transfer_*.py, tests/test_ad_transform_matrix*.py and 7 more, -m "not slow"
  JAX 0.10.0: 916 passed, 1573 deselected in 475.93s
  JAX 0.10.2: 916 passed, 1573 deselected in 477.64s
test_ad_transform_matrix*.py -m slow -k "transfer or moment_driven or slab or los"
  JAX 0.10.0: 565 passed, 1203 deselected in 924.98s
  JAX 0.10.2: 565 passed, 1203 deselected in 903.04s
```

Lint (ruff
and black with JAX 0.10.0's Python, as above) passes, and the
documentation builds with `-W` under Sphinx 8.2.3 and 9.1.0 without
warnings.

The repository URL (`github.com/zzhang0123/syncmoments`), the
documentation URL (`syncmoments.readthedocs.io`) and the PyPI project
`syncmoments` named in `pyproject.toml`, the README and `docs/conf.py`
did not resolve when the round-3 review checked them; the repository
rename and the Read the Docs and PyPI projects must exist before release.
