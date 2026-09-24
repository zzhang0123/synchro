# `synchro.model`: finite joint response and spectral fits

This document describes the top-level layer of the package: how a finite set
of joint population moments predicts channel-integrated Stokes spectra with an
explicit error budget, how assumptions about the population are declared, and
how the same response matrix serves prediction and fitting. It follows the
manuscript (Zhang & Chluba, *A statistical framework for synchrotron
emission*), whose equation labels are quoted in backticks; anything the
manuscript does not define is tagged `[extension]`.

Section 12 records the test inventory of the final acceptance run on
2026-09-24, after three rounds of review fixes (test files, what each establishes), and the numbers
the executed README examples printed. The pass/fail record of the suite is
kept outside this document.

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
Importing `synchro` enables float64.

## 2. Object model

```
synchro/model/
  index.py         Truncation, MomentIndex, Entry
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

`Truncation(L_mu, L_eta, N, depth_degree=None)` holds static integers. `N` is
the total Taylor degree. `depth_degree=None` uses the main-text cutoff
`r + s + b <= N` for the `P` block; `depth_degree=L` uses the `app: depth moments`
layout `r + s <= N, b <= L` (`app: depth moments`).

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
multiplied by `(2l+1)(2k+1)/4`, in `index.pairs` order; `build_basis`
differentiates them with nested `jax.jacfwd` in `(z_gamma, z_B)`.

### 6.1 `HarmonicKernel`

`HarmonicKernel(m_max, *, n_nodes=None, n_outer=64, n_inner=64, n_mu=48,
n_eta=48, m_chunk=64, tail_probe=None, E_phys=None, quadrature="product")`
implements `eq: smooth channel kernel` and `eq: channel derivative
coefficients`: `H_{P,j} = sum_m Q_m R_j(nu_m) w(tau_m)`, `nu_m = m nu_B / D`.
Harmonics are traced floats `m = 1..m_max` batched inside `lax.map` chunks.

Angular projection, primary route (`quadrature="product"`): for each harmonic,
channel and sign the cell in `t = mu eta` where the line meets the channel
support is integrated with `t = sign v^4`, `v` on Gauss-Legendre nodes, and
the inner variable `log mu` from `log|t|` to `0`; the `mu < 0` half enters
through the parity factors `(1 +/- (-1)^(l+k))`. This resolves the
`1/|mu|` Jacobian of the double projection before parameter differentiation
(main text, Section 4). Because `R_j` vanishes to all orders at its support
edges, the boundary terms of the `(gamma, B)`-dependent cells vanish.
Automatic differentiation is the derivative of the computed values in
cells with `lo = 0` or `lo >= 0.1 hi`. In cells with `0 < lo < 0.1 hi` (a
channel edge meeting a line near `t = 0`) the nodes follow the moving bounds
affinely in `t` with frozen node values (`RHO_SWITCH = 0.1` in
`_harmonic_cells.py`), which avoids the `lo^(-3/4)` node velocity of the
`v`-map. Either way a derivative carries its own quadrature error; at 64
nodes the two node motions differ by 2.7e-6 relative at the switch. Each
`lax.map` chunk of harmonics is checkpointed: at benchmark size
(`m_max = 40`, three channels, `Truncation(2, 2, 1)`, 64 nodes) `jax.grad`
needs 5.8 GB of XLA temporaries (357.6 GB without the checkpoint) and
`jacfwd` 6.7 GB. The secondary route (`quadrature="tensor"`, `n_mu x n_eta`
Gauss-Legendre) serves the polynomial test kernel and the opt-in
product-versus-tensor route check (`build_basis(..., cross_route=True)` or
`convergence="full"`), recorded in `provenance.finite_checks`; it converges
slowly for derivative bases.

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
n_eta=48)`
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

Measured costs (CPU, one machine): harmonic product quadrature with
`m_max = 40`, three bump channels, `L = 2`: value 0.9 s per call at 64 x 64
nodes; jit-compiled Jacobian at 32 x 32 about 1 s, Hessian 2.2 s, third
order 5.8 s (compile 11 s). Continuum kernel: sub-second.

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
closed factor is exactly zero (only `uniform_mu` with `l > 0`) are removed
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
`uniform_mu`, `gaussian_depth`, `cumulant_depth`, `delta`, `fixed_table`.

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
| `fully_independent` | six singletons | `2N + max_b + L_mu + L_eta + 2 = 12` (`max_b = N`, or `depth_degree` in the `app: depth moments` layout) |
| `nodal(index, nodes)` | nonnegative weights on fixed nodes (softmax of logits) | number of nodes |

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

## 8. Error budget (`errors.py`)

`ErrorTerm(value, kind, note, manuscript_term)`: `value` is a nonnegative
envelope `(n_ch, 4)` (or broadcastable) in channel Stokes units, `None` iff
`kind in {unbounded, not_applicable}`. `declared_zero(reason)` records a
zero bound with its reason. `total()` orders kinds `bound < measured <
estimate` and returns the weakest kind present.

`ErrorBudget` (`eq: channel error budget`) slots, in order:

| slot | content | manuscript term |
|---|---|---|
| `basis_remainder` | `eq: local response remainder` from `RemainderInputs` (angular residual and order-`N+1` derivative envelopes times absolute moments); `unbounded` for a line-kernel basis with channel smoothness `< N + 1`, and in `predict(samples=...)` when order `N + 1` is not in `certified_orders`; on a screen route `P` is the intrinsic residual only | `N_src <rho_nu>` |
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
`channels`, `truncation`, `reference`, `support`, `phase_route`,
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
intrinsic_residual, kind, *, absolute_depth_coefficients=None)` collects the
inputs of `eq: local response remainder`. `from_samples(samples, basis, *,
kernel=None, derivative_envelope="probe", angular_residual=None,
phase=None, ...)` estimates the order-`N+1` envelopes with nested `jacfwd`
on the samples (`kind="estimate"`) and computes the absolute moments on the
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
`PYTHONPATH=. python -m pytest tests/model --co -q` at 12:36 on 2026-09-24
(Section 12.7 gives the totals and the `slow` split). Every oracle is
NumPy/SciPy/mpmath code written independently of the JAX path.

### 12.1 Benchmark (`tests/model/test_benchmark.py`, 45 tests)

Reproduction of the manuscript's smooth-channel example (Section 5.3.1):
`gamma0 = 20`, `B0 = 1` G, three bump channels at `y_j = 2, 4, 8` with 65 %
half-widths, the correlated population of `eq: channel toy population`,
`HarmonicKernel(m_max=40)` (`required_m_max` is 40, so the harmonic-tail
term is a zero bound), `Truncation(L, L, N)`, pinned against the saved
`finite_IQUV` and `direct_IQUV` rows of
`validation/full_response_results.json` (units converted by
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
| `test_docstrings.py` | 112 | every public module of `synchro.model` and `synchro.model.fit` states `LABEL`, units, shapes and what it does not certify; every public class and function has its own docstring, a manuscript label (classes) and a limits statement or a pointer to the module docstring that has one; `docs/DESIGN.md` names every keyword of `predict`, `direct_channel_average`, `build_basis`, `basis_convergence` and the three fit routes, neither this file nor the README repeats a statement superseded by the round-2 fixes, and the README shows `assumption_allowances` with `screen_factorisation_bound` (textual checks only) |

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
| `test_remainder_probe_route.py` | 35 | `from_samples(phase=None)` probes the basis route (harmonic kernel with a Gaussian screen; differs from the per-emitter route; the `predict(samples)` path; a stub basis without `phase` keeps the exact phase); no module, class or method docstring outside the listed modules calls a finite check "certified" (30 modules) |
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
reference, four corner cells pinned to mpmath). `test_boundaries_corners.py`:
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
both sides with mpmath pins of `F` and `G`, `eta -> +/-1`, and the
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
`synchro.model` and paste the new output into the README. The run of
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

### 12.7 Test inventory

Final acceptance run on 2026-09-24 (after the third round of review fixes):

```
PYTHONPATH=. python -m pytest tests/model -m "not slow" -q   # 1168 passed, 42 deselected
PYTHONPATH=. python -m pytest tests/model -m slow -q         # 42 passed, 1168 deselected
PYTHONPATH=. python -m pytest -m "not slow" -q --ignore=tests/model   # 187 passed (legacy suite)
```

The 42 `slow` tests are opt-in (`-m slow` or `SYNCHRO_RUN_SLOW=1`, gated by
`tests/conftest.py`; a plain run reports them as skipped): 24 full-resolution
benchmark cases in `test_benchmark.py` (16 x 16 latent and 256 x 256
angular nodes), 16 in `test_boundaries_corners.py`, 1 in
`test_boundaries_continuum.py` and 1 in `test_boundaries_depth.py`. Earlier counts
(1176 collected at 12:36) are superseded; the third round added strict-JSON,
certification-wording and prediction-serialisation regressions
(`test_json_strict_regressions.py`, `test_certif_wording_fit.py`) and the
Laplace/Gamma screen tests (`test_screens.py`).

### 12.8 Known limitations

* `E_phys` and `E_tail` are never bounded by the package; they are inputs.
* The Galactic harmonic regime (`m ~ 1e12`) is refused, not computed; the
  continuum kernel there does not bound its discrepancy from the harmonic
  reference, and its measured assumption terms cannot see pitch-angle
  dependence (Section 12.3).
* Finite refinement checks and the refined-rule `numerical` envelope are
  `estimate` terms, not certificates; the remainder probe evaluates `H` at finitely
  many points and is an `estimate` unless envelopes are supplied.
* The order-`N+1` probe of `RemainderInputs.from_samples` costs one nested
  `jacfwd` of the angular projection per probe point; at `L = 8`,
  `N = 2` on the harmonic kernel one point takes about 2.5 minutes
  (146 s, compile included), so the README example probes the reference
  point only, on a 16-atom discretisation.
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
