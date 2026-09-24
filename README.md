# synchro
<!-- docs:intro:start -->

Documentation: <https://synchro.readthedocs.io>. Releases:
<https://github.com/zzhang0123/synchro/releases>.

Differentiable JAX/Equinox synchrotron kernels, finite statistical contractions,
and polarised radiative transfer for *A statistical framework for synchrotron
emission* (Zhang & Chluba).

This research implementation keeps four operations explicit: physical kernel
approximation, finite kernel Taylor expansion, statistical closure, and numerical
solution. Passing a kernel or regression test does not certify a Galactic
foreground model at 21-cm precision. All relevant discrepancies and uncertainties
must be propagated to the chosen scientific quantity.

The positive finite-cumulant PDF reconstruction and the independent 21-cm
assessment live in the manuscript repository's `validation/` directory. They do
not import this package. `CumulantExpansion` here is the historical name of a
quadratic kernel average; it is not that PDF reconstruction.

The main paper covers radiation kernels, population averages, moment/cumulant
representations and their error control, including internally mixed emission
and pure Faraday rotation. External screens are a special case. General ordered
transfer, Magnus, absorption/conversion and the 21-cm examples remain exploratory
modules beyond this main model.

<!-- docs:intro:end -->
## Statistical interfaces
<!-- docs:screens:start -->

`QuadraticTaylorExpansion` is the descriptive alias for `CumulantExpansion`;
existing imports and Equinox trees retain their identity. Its low-level call
continues to contract supplied tensors. For a validated entry point, use
`model.checked_average(mean, covariance, absolute_error=...)`: it checks finite
real inputs, shapes, symmetry and positive semidefiniteness and returns
`(prediction, envelope)`. The check uses correlation coordinates so a large
independent variance cannot hide a small invalid block. Its numerical tolerance
is `32 * P * eps`; no covariance projection or eigenvalue clipping is performed.

The required `absolute_error` is a caller-established componentwise envelope in
the prediction's units, broadcastable to its shape. The routine validates and
returns it; it does **not** derive a remainder from two moments, certify physical
support or bound floating-point error. For a downstream linear response `A`,
propagate it as `abs(A) @ envelope` (flattening the Stokes/harmonic axes as needed).
For example, zero is justified for this exactly quadratic synthetic kernel:

```python
import synchro
import jax.numpy as jnp

model = synchro.QuadraticTaylorExpansion(
    harmonics=(1,), S0=jnp.zeros((1, 3)),
    dS=jnp.zeros((1, 3, 1)), ddS=2*jnp.ones((1, 3, 1, 1)),
)
prediction, error = model.checked_average([0.3], [[0.25]], absolute_error=0.)
# Each synthetic output is E[x**2] = 0.3**2 + 0.25 = 0.34.
```

`rm_moments` is the descriptive alias for `gaussian_rm_cumulants`. Neither name
establishes Gaussianity. Use `burn_depolarisation` for a declared Gaussian screen,
or `screen_polarisation` for a discrete weighted screen without that closure:

```python
from synchro.rm import screen_polarisation, rm_moments

mean_rm, var_rm = rm_moments([-1., 1.])
P = screen_polarisation([1., -1.], [-1., 1.], jnp.sqrt(jnp.pi / 4))
# P is -1j: incident polarisation and RM are correlated across the two rays.
```

RM is in rad/m² and wavelength in metres. Incident complex `P0 = Q + iU` is a
scalar or a vector matching the 1D RM samples; wavelengths have any shape, which
is also the output shape. Optional weights are nonnegative relative masses.
Forward evaluation scans rays, avoiding a samples-by-wavelengths phase array;
reverse-mode differentiation can still retain per-ray intermediates. JIT and
gradients are supported. Invalid inputs and arithmetic overflow raise errors.
Finite phase values alone do not guarantee accurate argument reduction for
arbitrarily large phases; floating-point and sampling accuracy need separate
checks in such regimes.

For fixed normalized weights, perturbing incident polarisation and RM gives
`|delta P| <= sum(w*|delta P0|) + 2*lambda²*sum(w*|P0|*|delta RM|)`.
Use consistent intermediate values when also changing weights; their additional
bound is `max(|P0|)*sum(|delta w|)`. Sampling/quadrature uncertainty remains an
external input. For internally distributed emission and frequency-dependent emitter spectra,
use `synchro.faraday.emission_polarisation` below. Absorption and conversion
require the separate transfer interfaces.

<!-- docs:screens:end -->
## Independent symbolic derivation
<!-- docs:derivation:start -->

[`derivation/`](https://github.com/zzhang0123/synchro/blob/main/derivation/README.md) contains local Wolfram scripts that derive
and check the helical orbit, retarded radiation, absolute harmonic powers and
Stokes basis conventions from the classical equations. Run
`wolframscript -file derivation/verify.wls` to regenerate the report. It also
compares the complex harmonic amplitudes with direct Fourier integration of the
original electric field, independently of this package's Python implementation.

<!-- docs:derivation:end -->
## Install and verify
<!-- docs:install:start -->

Python 3.12 is the tested runtime. From this checkout:

```bash
python -m pip install -e '.[validation]'
python -m pytest tests -q
```

The tested environment uses JAX 0.10.0, Equinox 0.13.7, NumPy 2.3.5,
SciPy 1.16.3, Matplotlib 3.10.8 and pytest 9.0.2. The dependency ranges in
`pyproject.toml` do not imply every version combination has been tested.
SciPy is confined to validation; it supplies independent special-function,
integration and matrix-exponential references.

Importing `synchro` enables JAX float64 globally for scientific accuracy. Import
it before creating arrays; this is a documented compatibility side effect.
For a source checkout without installation, `PYTHONPATH=.` remains supported.

```bash
PYTHONPATH=. python scripts/run_all.py                 # pytest, diagnostics, figures
PYTHONPATH=. python scripts/run_all.py --no-figures    # pytest and diagnostics
```

Printed diagnostics alone are not pass/fail physics tests. The runner first
executes pytest, then reports script completion. Generated companion figures
are package demonstrations; the manuscript uses its separate independent figure
generators and captions. Do not substitute one set without checking assumptions.

<!-- docs:install:end -->
## Harmonic radiation and derivatives
<!-- docs:radiation:start -->

```python
from synchro.stokes import stokes_harmonic
from synchro.derivatives import derivative_spectra

I, Q, V = stokes_harmonic(10, 5.0, 0.785, 1.047, B=5e-6)
value, gradient, hessian = derivative_spectra(10, 5.0, 0.785, 1.047, B=5e-6)
```

`B` is in Gauss; harmonic power is per source time in erg/s/sr. The reference is
vacuum radiation from a prescribed helical orbit, not a plasma-corrected or
self-consistent particle trajectory. Without `B`, the API preserves its legacy
dimensionless normalization. Differentiating that normalized response is a
different operation: physical fixed-B energy derivatives must include the
Lorentz-factor dependence of the gyrofrequency. These are fixed-harmonic
quantities. Fixed-frequency predictions also need the moving harmonic line
positions and a declared frequency/channel response.

The natural basis uses `Q=parallel-perpendicular` to the projected magnetic
field and `V=-2 Im(E_parallel E_perpendicular*)`. A fixed sky azimuth `phi` gives
`Q_sky=Q*cos(2phi)`, `U_sky=Q*sin(2phi)`. The recurrence implementation includes
the viewing-axis limit. Isotropic pitch angles alone do not force V to vanish.

Integer Bessel J uses a periodic integral with a contour shift to resolve
exponentially small values. A concrete harmonic sets its static resolution;
for traced orders the default is 2048 nodes. `bessel_jn(..., n_nodes=...)`
exposes that static choice. Unresolved `n+abs(x)>n_nodes/2` returns NaN instead
of an aliased answer. This guard is not an accuracy certificate: validate
resolution and derivatives over the actual domain. Modified K and continuum
F/G use continuous hyperbolic integrals. Their public tail-bound helpers bound
only the finite integral tail, excluding quadrature and roundoff. F(0)=G(0)=0;
the continuum slopes diverge there, so derivative tests concern positive x.

`sed.power_law_emissivity_abs(..., theta=...)` gives directional ordered-field
continuum emissivity. Its `theta=None` default additionally averages viewing
angles/random field axes with the normalized sin(theta)/2 measure. Isotropic
electron pitch angles alone do not perform that viewing-direction average.

<!-- docs:radiation:end -->
## Finite statistical response
<!-- docs:fixed:start -->

```python
import equinox as eqx
import jax.numpy as jnp
from synchro.expansion import build_expansion

response = build_expansion([1, 2, 5, 10], 5.0, 0.785, 1.047, B=5e-6)
mean_deviation = jnp.zeros(3)                    # gamma, alpha, theta
covariance = jnp.diag(jnp.array([0.01, 1e-4, 0.0]))
stokes = eqx.filter_jit(response)(mean_deviation, covariance)
```

`build_expansion` precomputes values, gradients and Hessians; repeated calls
contract the same arrays with the supplied mean and covariance. Construction
choices are fixed harmonic orders/reference/units; model arrays are ordinary
Equinox leaves, so callers decide which parameters to differentiate. Runtime
array shapes should remain stable when reusing a compiled response.

At the same Taylor degree and with the same statistics, moment and cumulant
contractions are equal. A Gaussian has no cumulants beyond second order, but
still has fourth and higher moments: a quadratic kernel average leaves those
kernel terms out. The general scalar Bell helper can construct additional
moments by setting unprovided cumulants to zero; that is an explicit closure,
not a guarantee of a legal PDF. Vector contractions support at most four orders
and reject higher orders. No universal pitch or energy width is safe: assess
the kernel remainder at the actual reference, frequency and science precision.

`apply_B` uses the exact B-squared factor at fixed harmonic only when the
remaining population parameters are fixed or B is independent of them.
Correlated populations require conditional or mixed statistics. It does not
supply physical units to a normalized spectrum or account for frequency shifts.

### Correlated field, energy and orientation

An exact conditional B average leaves a function of the other parameters.
`apply_B` does not infer that function: multiplying an already averaged spectrum
by a global field factor assumes independence. The correlated alternative is
`response.mixed_average(field_moments, phase_moments, absolute_error=...)`.
It expands only the known natural-basis kernel at fixed B0 and contracts its
derivatives with unnormalised mixed moments. No unknown conditional function is
differentiated. `mixed_moments` computes these statistics from paired samples:

```python
from synchro import mixed_moments

def average_joint_population(response, offsets, B_over_B0, phi, weights, envelope):
    # response was built at B0>0; offsets use the SAME (gamma, alpha, theta)
    # reference. Angles are in radians; alpha is not the pitch cosine mu.
    W = B_over_B0**2
    field = mixed_moments(offsets, W, weights)
    phase = mixed_moments(offsets, W*jnp.exp(2j*phi), weights)
    return response.mixed_average(field, phase, absolute_error=envelope)
```

`offsets` has shape `(samples,P)`, while B ratios, phi and optional relative
electron-number weights have shape `(samples,)`. Keep the samples paired to
retain correlations. The helper returns `(E[W], E[W*dq], E[W*dq*dq])`, with
shapes `()`, `(P,)`, `(P,P)`. Weights are normalized, **W is not**. The moments
can instead come from a specified joint model. A complex zeroth moment may be
zero while its higher mixed moments still contribute. Ordinary covariance of q
and marginal B moments cannot in general determine these inputs.

`mixed_average` returns sky `(I,Q,U,V)` and a supplied componentwise envelope,
both `(harmonics,4)`; the older call and `checked_average` still return natural
`(I,Q,V)`. It keeps B-squared and the sky rotation exact, but truncates the q
kernel at degree two. Thus `E[B²*dq_i*dq_j]` is retained even though it has total
degree four in B and q. Moments of q must use alpha for this builder; converting
only the mean/covariance from mu=cos(alpha) is not generally sufficient.

For a pointwise kernel remainder `abs(R_s(q)) <= rho_s(q)`, use the weighted
bound `E[(B/B0)²*rho_s]`; the natural Q bound controls each sky Q and U component.
Uncertainty in the three supplied moment tensors adds
`abs(S0)*epsilon0 + abs(dS)@epsilon1 + abs(ddS):epsilon2/2` componentwise.
These are mathematical input contracts. The method requires and returns an
externally justified `absolute_error` in output units; it does not derive it,
validate joint-moment realizability or physical support, or bound numerical and
sample/quadrature errors. Zero is appropriate for an exactly quadratic kernel,
not automatically for synchrotron radiation. For a subsequent linear observation
map propagate the envelope with the elementwise absolute response matrix.

The new contraction is for fixed harmonic numbers. A frequency channel must
also include B-dependent line positions and cannot use this simple B-squared
factorisation. The power-law SED functions remain restricted continuum examples;
they are not assumptions of the general kernel or mixed-moment method.

### Finite statistics for a spectral fit

At a fixed expansion reference the response contracts a finite vector of joint
statistics with known spectral coefficients. The same statistics apply to all
harmonics; correlations do not require a separate unknown function for each
harmonic. In the current three-coordinate quadratic builder, each moment block
contains ten distinct monomials (one constant, three linear and six symmetric
quadratic terms). The real field block and complex phase block thus contain
thirty real statistical components before constraints and degeneracies.

These are not thirty unrestricted fit parameters. They must be consistent with
one nonnegative joint electron-number distribution on the declared support.
For example, `abs(E[W*exp(2j*phi)]) <= E[W]` for nonnegative W; analogous
cross-moment constraints couple the two blocks. `mixed_average` checks neither
their full realizability nor whether the spectral response identifies them.
The method also does not establish the required remainder envelope from the
retained moments. If higher moments are unknown, they must not silently be set
to independent or Gaussian values to supply that envelope.

A frequency-channel fit needs coefficients built from the channel kernel,
including moving line positions, with a single fixed population measure.
Chromatic observing weights belong in those coefficients. The fitting layer
must additionally specify the admissible joint statistics, number amplitude,
measurement likelihood and propagated model discrepancy. An absolute error
envelope is not automatically an independent Gaussian noise covariance.
Weak or degenerate spectral responses constrain combinations of moments rather
than each physical statistic separately; a small residual is not a remainder
certificate or a reconstruction of the full population PDF.

The Faraday extension uses the joint distribution of emitting parameters and
the intervening depth from each emitter to the observer. It includes mixed
emission/rotation and retains spectrum-depth correlations. `joint_faraday_average`
contracts a finite, frequency-independent moment matrix with intrinsic response
coefficients and a finite phase series. The response and its truncation envelope
are implemented; the application still supplies the intrinsic/channel basis,
physical support, admissible moment set and likelihood. The Gaussian
`burn_depolarisation` helper remains a declared factorised foreground model.

<!-- docs:fixed:end -->
### Finite joint response and spectral fits
<!-- docs:model:start -->

`synchro.model` predicts channel-integrated Stokes spectra from a finite set of
joint population moments (`eq: finite joint response`, `eq: channel derivative
coefficients`) and fits those moments to channel data (`extra eq: finite fit
model`). The population is never factorised unless an assumption object is
declared, and every output carries an `ErrorBudget` whose terms are `bound`,
`estimate`, `measured`, `unbounded` or `not_applicable`; a missing input is
`unbounded`, never zero. `docs/DESIGN.md` describes the object model, the
index layout, the free-parameter counts and the acceptance run. The four
examples below are the printed output of `scripts/model_examples.py`
(`PYTHONPATH=. python scripts/model_examples.py`, 6 min 20 s wall time on one
machine, most of it the basis build and the order-3 derivative probe of
example (a); examples (b) to (d) live in `scripts/model_examples_continuum.py`
and `scripts/model_examples_fit.py`); rerun it after any change and replace
the quoted numbers.

#### Example (a): harmonic kernel, the manuscript's smooth-channel benchmark

The setting of main-text Section 5.3.1: `gamma0 = 20`, `B0 = 1` G, three bump
channels at `y_j = 2, 4, 8` (`y = nu / nu_*`, `nu_* = e B0 / (2 pi gamma0 m_e c)`)
with 65 % support half-widths, the correlated population of
`eq: channel toy population` on its full 16 x 16 latent and 64 x 64 angular
grid (1,048,576 samples), harmonics through `m = 40`, and the truncation
`N = 2`, `L = 8` (1306 real moments).

```python
from synchro.model.assumptions import independent_screen
from synchro.model.basis import build_basis
from synchro.model.channels import Channels
from synchro.model.harmonic import HarmonicKernel
from synchro.model.index import Truncation
from synchro.model.moments import JointMoments, Reference, Support
from synchro.model.predict import predict

y = np.array([2.0, 4.0, 8.0])
channels = Channels.bump(centres_hz=y * NU_STAR, widths_hz=0.65 * y * NU_STAR)
reference = Reference(20.0, 1.0, depth_ref=4 * S_DEPTH, scales=(20.0, 1.0, S_DEPTH))
support = Support(gamma=(16, 24), B=(0.8, 1.2), depth=(0, 10 * S_DEPTH), truncated=False)
basis = build_basis(HarmonicKernel(m_max=40), channels, Truncation(8, 8, 2), reference,
                    support=support)                       # 96 s, convergence="angular"
joint = JointMoments.from_samples(samples, basis.index, reference)
screened = JointMoments.from_samples(samples, basis.index, reference,
                                     parameter_map=independent_screen(basis.index),
                                     discrepancy="measured")
pred = predict(basis, screened, amplitude=1.0)
print(pred.summary()); print(pred.budget.total().kind); print(pred.budget.unbounded())
```

`S_DEPTH = 1 / (2 (c / nu_*)^2)` converts the manuscript's depth coordinate
`zeta` to rad/m^2. With the joint moments the prediction reproduces the
manuscript's saved finite response and its direct average:

```text
required_m_max(support, channels) = 40
index: n0=486 n2=410 n_real=1306  build time 96.4 s
numerics: {'n_nodes': 256, 'm_range': (1, 26), 'width_ratio_min': 1.2999999999999998, 'cells': 64, 'route': 'product'}
population size: 1048576
joint moments:  max |pred - manuscript finite| / I_direct per channel: [1.52748222e-06 5.14411821e-07 1.39817674e-06]
joint moments:  max |pred - manuscript direct| / I_direct per channel: [0.00175259 0.00053129 0.00021887]
```

The second line is the manuscript's 1.75e-3 (Figure `fig: full channel
response`, width 1.0, `N = 2`, `L = 8`). The build time includes the one
`convergence="angular"` rebuild (the same route at doubled nodes for the
`numerical` envelope). The product-versus-tensor route check is opt-in
(`build_basis(..., cross_route=True)` or `convergence="full"`) and is
recorded as a named finite check, not a budget term.
With `independent_screen` declared and `discrepancy="measured"` the moments
are the factorised tensor and the budget carries the measured
`|m_joint - m_fac|` propagated as `N_src |C| Delta`:

```text
Prediction (eq: finite joint response): 3 channels, units per-electron channel Stokes per unit z-moment: erg/s/sr (unit_peak) or erg/s/sr/Hz (unit_integral); z = ((gamma-gamma0)/s_gamma, (B-B0)/s_B, (depth-depth_ref)/s_depth), Gauss, Hz, rad/m^2, amplitude 1
  channel 0: I=3.586096e-19 Q=1.882597e-20 U=-9.944719e-20 V=8.295205e-20
  channel 1: I=1.199577e-18 Q=-1.814062e-19 U=-3.997255e-19 V=3.017935e-19
  channel 2: I=3.461261e-18 Q=-7.788530e-19 U=-1.138940e-18 V=7.987727e-19
  error envelope unbounded; unbounded terms: basis_remainder, statistical_input, physical_kernel, depth_model, amplitude
    basis_remainder: unbounded -
    statistical_input: unbounded -
    physical_kernel: unbounded -
    harmonic_truncation: bound 0.000e+00
    excluded_tail: bound 0.000e+00
    numerical: estimate 7.826e-24
    screen_exponent: not_applicable -
    depth_model: unbounded -
    amplitude: unbounded -
    assumption:independent_screen: measured 2.776e-20
  assumptions: independent_screen
  note: incident polarisation not modelled
  note: moment-level discrepancy of kind 'measured' propagated as N_src |C| Delta (assumption)
budget.total().kind = unbounded
budget.unbounded() = ('basis_remainder', 'statistical_input', 'physical_kernel', 'depth_model', 'amplitude')
assumption 'independent_screen': kind measured; max |pred_screen - pred_joint| / I = 7.566e-03; max term / I = 7.731e-02; covered: True
```

The screen assumption moves the prediction by 7.6e-3 of channel `I` (the
toy population correlates depth with energy and field); the measured term
covers that shift. The `numerical` term is the per-column envelope
`|C_2x - C|` of the product route against itself at doubled nodes,
contracted with `|m|`. The total is `unbounded` because five inputs are
missing. Declaring them for this synthetic population (`E_phys` zero
because the harmonic reference is the truth, `statistical_input` zero
because the moments are exact sums over the measure,
`amplitude_uncertainty=0`, `depth_model` zero because the depths are the
sample values) and probing the basis remainder gives the output below. The
remainder inputs come from two measures. The order-3 derivative envelope `H`
(at the reference point only) and the angular residual `rho_ang` are
probed with `RemainderInputs.from_samples` on a separate 2-node (16-atom)
Gauss-Legendre discretisation of the same toy model, which is not a subset
of the population; this takes 147 s, dominated by compiling the nested
`jacfwd` of the `L = 8` projection. The absolute moments
`<|P_l P_k| ||z||^3>` of `eq: local response remainder` are recomputed on
the predicted population itself (weighted sums, under a second); the
16-atom probe underestimates their row sums by the factors printed:

```text
remainder probe (H at the reference, rho_ang on the 16-atom discretisation, absolute moments on the 1048576 population samples): 147.2 s, kind estimate
absolute-moment row sums, population / 16-atom probe (z_2, z_3): [1.732 1.589]
Prediction (eq: finite joint response): 3 channels, units per-electron channel Stokes per unit z-moment: erg/s/sr (unit_peak) or erg/s/sr/Hz (unit_integral); z = ((gamma-gamma0)/s_gamma, (B-B0)/s_B, (depth-depth_ref)/s_depth), Gauss, Hz, rad/m^2, amplitude 1
  channel 0: I=3.586096e-19 Q=1.882597e-20 U=-9.944719e-20 V=8.295205e-20
  channel 1: I=1.199577e-18 Q=-1.814062e-19 U=-3.997255e-19 V=3.017935e-19
  channel 2: I=3.461261e-18 Q=-7.788530e-19 U=-1.138940e-18 V=7.987727e-19
  error envelope (estimate): max 1.330e-15, 3.843e+02 of the largest channel I
    basis_remainder: estimate 1.330e-15
    statistical_input: bound 0.000e+00
    physical_kernel: bound 0.000e+00
    harmonic_truncation: bound 0.000e+00
    excluded_tail: bound 0.000e+00
    numerical: estimate 7.826e-24
    screen_exponent: not_applicable -
    depth_model: bound 0.000e+00
    amplitude: bound 0.000e+00
    assumption:independent_screen: measured 2.776e-20
  assumptions: independent_screen
  note: incident polarisation not modelled
  note: moment-level discrepancy of kind 'measured' propagated as N_src |C| Delta (assumption)
budget.total().kind = estimate
budget.unbounded() = ()
envelope covers |pred_screen - manuscript direct|: True
```

Every slot is now valued and the total is an `estimate`, but the envelope
is 384 times channel `I` while the measured finite-vs-direct error is
1.75e-3 of `I`: the `eq: local response remainder` envelope sums absolute
contributions over the 81 Legendre pairs with the Frobenius norm of the
third-order tensor and the absolute moments (the depth displacement reaches
`|z_depth| = 3` in the manuscript's scale), so at `L = 8` it exceeds the
actual error by five orders of magnitude. `H` is evaluated at one point
and `rho_ang` on the 16-atom measure, not the predicted population; neither
is a supremum over the support. The `amplitude` slot is a zero `bound`:
`amplitude_uncertainty=0` declares the amplitude exact. `E_phys` and the
excluded tail are declared by the caller. `predict(samples=population,
kernel=kernel)` would run the probe itself, on the basis phase route and
with the Support checks of the harmonic tail and of the declared-complete
excluded tail, but it would probe `H` at every sample; at `N >= 3` it
leaves `basis_remainder` `unbounded` (order `N + 1` is not among the
validated orders) and the caller passes `errors=RemainderInputs(...)`
with a supplied `derivative_envelope`.

#### Example (b): continuum kernel in the Galactic regime

At `B0 = 5` uG, `gamma0 = 3000` and channels between 0.1 and 3 GHz the
harmonic index needed to reach the channels is of order 1e12, so the
harmonic kernel is refused and `ContinuumKernel` (`eq: directional
continuum`) is the only route. It ignores the pitch cosine `mu`, which
forces `isotropic_pitch` (`L_mu = 0`), and it does not model `V`. The
Faraday screen is a Burn screen (`GaussianScreen(mean=30, sigma=5)` rad/m^2,
exact in `tau` for that screen, `depth_degree=0`), the population is a
product measure local to the reference (energies 2700..3300 with
`N(gamma) ~ gamma^-2.5`, field within 10 %, uniform pitch, one viewing
angle of 60 degrees, one sky azimuth), and the remainder is probed on
eight corner samples.

```python
from synchro.model.kernels import ContinuumKernel, required_m_max
from synchro.model.phase import GaussianScreen

centres = np.geomspace(0.1e9, 3.0e9, 8)
channels = Channels.bump(centres_hz=centres, widths_hz=0.3 * centres)
reference = Reference(gamma0=3000.0, B0=5e-6, depth_ref=30.0, scales=(300.0, 5e-7, 5.0))
support = Support(gamma=(2700.0, 3300.0), B=(4.5e-6, 5.5e-6), depth=(0.0, 60.0))
screen = GaussianScreen(mean=30.0, sigma=5.0)
basis = build_basis(ContinuumKernel(), channels, Truncation(0, 2, 2, depth_degree=0),
                    reference, support=support, phase=screen)
moments = JointMoments.from_samples(samples, basis.index, reference,
                                    parameter_map=isotropic_pitch(basis.index),
                                    discrepancy="measured")
remainder = RemainderInputs.from_samples(probe, basis, kernel=ContinuumKernel(), phase=screen,
                                         segment_points=(1.0,), angular_residual="probe")
pred = predict(basis, moments, amplitude=1e20, errors=remainder,
               statistical_input=np.zeros(basis.index.n_real))
```

```text
channels (GHz): [0.1   0.163 0.264 0.43  0.698 1.135 1.845 3.   ]
required_m_max(support, channels) = 2043405178518
critical frequency a_B gamma^2 (GHz) at gamma = 2700, 3000, 3300: [0.133 0.164 0.198]
Burn factor exp(-2 sigma^2 lambda^4) at the channel centres:
  [0.0000e+000 6.7602e-252 1.0801e-036 7.0752e-006 1.8302e-001 7.8414e-001
 9.6578e-001 9.9503e-001]
HarmonicKernel refused: required_m_max=2043405178518 lies outside the harmonic design regime (m <~ 16368); use ContinuumKernel with isotropic_pitch
index: components=('I', 'Q') n0=12 n2=12 n_real=36  build time 3.0 s
forced assumptions: ['independent_screen', 'gaussian_screen', 'isotropic_pitch']
notes: ('V not modelled [ContinuumKernel]', 'incident polarisation not modelled')
L_mu > 0 refused: ContinuumKernel requires the uniform_mu closure (isotropic pitch): use L_mu = 0
population size: 4096
remainder probe on 8 samples: 15.3 s, kind estimate
Prediction (eq: finite joint response): 8 channels, units per-electron channel Stokes per unit z-moment: erg/s/sr (unit_peak) or erg/s/sr/Hz (unit_integral); z = ((gamma-gamma0)/s_gamma, (B-B0)/s_B, (depth-depth_ref)/s_depth), Gauss, Hz, rad/m^2, amplitude 1e+20
  channel 0: I=2.321612e-01 Q=0.000000e+00 U=0.000000e+00 V=0.000000e+00
  channel 1: I=2.990974e-01 Q=-7.074508e-100 U=2.470894e-99 V=0.000000e+00
  channel 2: I=3.120780e-01 Q=-1.633457e-18 U=1.954271e-18 V=0.000000e+00
  channel 3: I=2.349223e-01 Q=8.577717e-06 U=-1.834397e-05 V=0.000000e+00
  channel 4: I=1.075459e-01 Q=2.940888e-03 U=1.145815e-03 V=0.000000e+00
  channel 5: I=2.343829e-02 Q=-5.012653e-03 U=5.761419e-03 V=0.000000e+00
  channel 6: I=1.700234e-03 Q=1.211610e-03 U=-6.457802e-04 V=0.000000e+00
  channel 7: I=2.302571e-05 Q=-2.624099e-06 U=-2.138403e-05 V=0.000000e+00
  error envelope unbounded; unbounded terms: physical_kernel, excluded_tail, depth_model, amplitude, assumption:isotropic_pitch, assumption:independent_screen, assumption:gaussian_screen
    basis_remainder: estimate 1.374e-02
    statistical_input: bound 0.000e+00
    physical_kernel: unbounded -
    harmonic_truncation: not_applicable -
    excluded_tail: unbounded -
    numerical: estimate 2.314e-08
    screen_exponent: bound 0.000e+00
    depth_model: unbounded -
    amplitude: unbounded -
    assumption:isotropic_pitch: unbounded -
    assumption:independent_screen: unbounded -
    assumption:gaussian_screen: unbounded -
  assumptions: independent_screen, gaussian_screen, isotropic_pitch
  note: V not modelled [ContinuumKernel]
  note: incident polarisation not modelled
  note: moment-level discrepancy of kind 'measured' not propagated: assumption 'isotropic_pitch' is forced by the basis and stays unbounded
budget.total().kind = unbounded
budget.unbounded() = ('physical_kernel', 'excluded_tail', 'depth_model', 'amplitude', 'assumption:isotropic_pitch', 'assumption:independent_screen', 'assumption:gaussian_screen')
assumption 'isotropic_pitch': kind unbounded, max value -
assumption 'independent_screen': kind unbounded, max value -
assumption 'gaussian_screen': kind unbounded, max value -
polarisation fraction |P|/I per channel: [0.000e+00 0.000e+00 0.000e+00 1.000e-04 2.930e-02 3.258e-01 8.075e-01
 9.357e-01]
max |pred - direct| / I per channel: [0.03 0.02 0.01 0.04 0.12 0.25 0.42 0.59]
sum of the valued terms / I per channel: [0.03 0.03 0.01 0.04 0.13 0.32 1.   4.37]
valued terms cover |pred - direct|: True
```

Four things to read off. `physical_kernel` stays `unbounded`: the continuum
replacement is not bounded against the harmonic reference, and no channel
comparison can be run at `m ~ 1e12`. All three assumptions are `unbounded`
because no allowance is passed (example (d) shows `assumption_allowances`).
The screen route's `independent_screen` and `gaussian_screen` have no
declared discrepancy. `isotropic_pitch` is forced by the continuum kernel,
and the retained `L_mu = 0` moments cannot test it: the measured
`|m_joint - m_fac|` on those rows is zero by construction, so `predict`
keeps the term `unbounded` and notes that the moment-level discrepancy was
not propagated. The Burn factor `exp(-2 sigma^2 lambda^4)` removes the
polarisation below 0.4 GHz and is 0.995 at 3 GHz; the 94 % `|P|/I` printed
for the top channel is the continuum kernel's intrinsic polarisation
fraction on its tail. The upper channels sit on the exponential tail of the
kernel (the critical frequency `a_B gamma^2` of `eq: directional continuum`
is 0.164 GHz at `gamma0`, 0.133 to 0.198 GHz over the `gamma` support),
where the quadratic Taylor expansion in `z_gamma` is poor: the direct
average differs by up to 59 % of `I` there, and the probed
`basis_remainder` says so (the valued terms cover the difference in every
channel, at the price of an envelope larger than the signal in the top two
channels). Reducing that difference needs a narrower `gamma` support or a
higher `N`.

#### Non-Gaussian independent screens

`LaplaceScreen(mean, sigma)` and `GammaScreen(mean, sigma, shape=4.0, sign=1)`
(`synchro.model.screens`) are the illustrated screens of the manuscript
(`eq: illustrated screens`). Each is an exact characteristic function with the
same depth mean and standard deviation as `GaussianScreen`, applied line by
line inside the channel sum, with `depth_degree=0`. At `mean=0`, `sigma=1` and
`t = tau*sigma`, the three weights are `exp(-t**2/2)`, `1/(1 + t**2/2)` and
`exp(-2j*t)*(1 - 1j*t/2)**-4`. The Gamma screen adds the residual position
angle `-t + 2*arctan(t/2)`. Each screen records `independent_screen` plus
`laplace_screen` or `gamma_screen`, all with unbounded discrepancy: the screen
shape is a declared input, and zero covariance between depth and emission does
not justify the factorisation. In `predict` these forced assumptions stay
`unbounded` even when the moments carry a measured discrepancy, unless the
caller passes an allowance through `assumption_allowances` (example (d));
without one a screen-route total is `unbounded`. `tests/model/test_screens.py` checks the
weights against SciPy integrals of the screen densities, and the
`build_basis`/`predict` route and `direct_channel_average` against a
per-atom NumPy channel sum.

#### Example (c): `fit_linear` on synthetic continuum data

Synthetic data from the configuration of example (b) with twelve channels:
`d = A C m* + eta` with Gaussian noise at 0.1 % of channel `I`, `V` masked
(not modelled), `isotropic_pitch` declared (affine, so `fit_linear`
applies), the amplitude fitted. `StokesData` takes the channel Stokes
`(n_ch, 4)`, per-row variances (channel-major) and a mask of rows to keep.

```python
from synchro.model.fit.diagnostics import feasibility_checks, identifiability
from synchro.model.fit.linear import fit_linear
from synchro.model.fit.observation import StokesData

data = StokesData(stokes=stokes, noise=sigma.ravel() ** 2,
                  mask=jnp.array([True, True, True, False]))
result = fit_linear(basis, data, isotropic_pitch(basis.index))
report = identifiability(basis, data, isotropic_pitch(basis.index))
feasibility = feasibility_checks(result.moments, basis.index, support)
```

```text
-- Truncation(L_mu=0, L_eta=0, N=0, depth_degree=0): n_real=3, free=2, data rows kept 36 of 48
rank=3 chi2=22.099 dof=33 converged=True amplitude=9.995439e+19
max |m_fit - m_true| = 6.164e-04; covariance is None: False
identifiability: rank=3 null_dim=0 weak(0.5)=()
feasibility: Necessary conditions only (eq: joint moment feasible set): 5 passed, 0 failed; 4 entries not computable from the supplied moments. Passing does not certify membership of conv{psi(x): x in D}.
  failed: ()
bias_bound kind: unbounded | prediction statistical_input kind: unbounded
-- Truncation(L_mu=0, L_eta=0, N=1, depth_degree=0): n_real=9, free=8, data rows kept 36 of 48
rank=6 chi2=28.914 dof=30 converged=True amplitude=9.974895e+19
max |m_fit - m_true| = 3.131e-02; covariance is None: True
identifiability: rank=6 null_dim=3 weak(0.5)=('<z_B>', 'Re <z_B e^{2i phi}>', 'Im <z_B e^{2i phi}>')
  null direction 0: (('amplitude', -0.0013448963207402355), ('Re <e^{2i phi}>', -0.021408406149540354), ('<z_B>', 0.00979391940529722))
  null direction 1: (('amplitude', 0.003401318989344688), ('Re <e^{2i phi}>', 0.0577494405682197), ('<z_B>', -0.024769377043885172))
feasibility: Necessary conditions only (eq: joint moment feasible set): 4 passed, 3 failed: hermitian_psd[1,e^{2i phi}], support_bound[M2], abs_bound[f=1]; 6 entries not computable from the supplied moments. Passing does not certify membership of conv{psi(x): x in D}.
  failed: ('hermitian_psd[1,e^{2i phi}]', 'support_bound[M2]', 'abs_bound[f=1]')
bias_bound kind: unbounded | prediction statistical_input kind: unbounded
total time 5.7 s
```

At `N = 0` (three unknowns: the amplitude and the complex `<e^{2i phi}>`)
the design has full rank, the moments are recovered to 6e-4 in `z` units
and every computable necessary condition passes. At `N = 1` the design is
rank-deficient by three: the continuum kernel depends on `(gamma, B)` only
through `B_perp` and `B_perp gamma^2`, so the energy and field
displacements enter through one combination each in `I` and in `P`. The
rank is decided on the column-equilibrated design `G D`,
`D = diag(1/||G_i||)`, so it does not depend on the `Reference` scales;
`weak(0.5)` names `<z_B>` and `<z_B e^{2i phi}>`, and the null-space
vectors (their first three components are printed) are in those
equilibrated coordinates (`report.column_scales` holds `D`). The minimum
equilibrated-norm solution then fails three necessary conditions
(`|<e^{2i phi}>| <= 1` among them), `covariance` is `None`, and the report
says which combinations the data do constrain. `bias_bound` is `unbounded`
at both orders because the data carry no declared discrepancy (at `N = 1`
the rank deficiency alone would also make it `unbounded`), and so is the
fitted prediction's `statistical_input`.

#### Example (d): a caller-supplied screen allowance

`assumption_allowances={name: ErrorTerm}` (in `predict` and
`direct_channel_average`) replaces the budget term of a named assumption,
including one forced by the basis, with an allowance in Stokes units. The
setting is that of example (b) with two rays of equal mass: the field is
4.7 uG with Faraday depth 25 rad/m^2 on one ray and 5.3 uG with 35 rad/m^2
on the other, so emission and screen are correlated. The screen is the
exact `EmpiricalScreen` of the two depths, which leaves `independent_screen`
as the only screen assumption. Its allowance is
`bounds.screen_factorisation_bound` (`detail-eq: screen factorisation
error`) in its per-line form, with the channel quadrature nodes `nu_n` as
the lines: `sigma_{j,n} = w_n |R_j(nu_n)| sigma_P(nu_n)`, where
`sigma_P(nu_n)` is the ray standard deviation of the incident polarisation
`<K_Q e^{2i phi}>` of each ray, and `Phi_{j,n} = |<exp(i tau_n depth)>|` over
the rays. Each ray is uniform in `mu` by construction, so the continuum
kernel's forced `isotropic_pitch` gets a declared zero.

```python
from synchro.model.bounds import screen_factorisation_bound
from synchro.model.errors import ErrorTerm
from synchro.model.phase import EmpiricalScreen

screen = EmpiricalScreen(jnp.asarray([25.0, 35.0]), jnp.asarray([0.5, 0.5]))
basis = build_basis(ContinuumKernel(), channels, Truncation(0, 2, 2, depth_degree=0),
                    reference, support=support, phase=screen)
moments = JointMoments.from_samples(samples, basis.index, reference)
allowances = {
    "independent_screen": screen_factorisation_bound(sigma, phi, amplitude=1e20),
    "isotropic_pitch": ErrorTerm.declared_zero("each ray is uniform in mu by construction",
                                               shape=(channels.n_ch, 4)),
}
pred = predict(basis, moments, amplitude=1e20, statistical_input=0.0,
               amplitude_uncertainty=0.0, assumption_allowances=allowances)
```

```text
forced assumptions: ['independent_screen', 'isotropic_pitch']
without allowances: budget.unbounded() = ('basis_remainder', 'physical_kernel', 'excluded_tail', 'depth_model', 'assumption:independent_screen', 'assumption:isotropic_pitch')
with allowances:    budget.unbounded() = ('basis_remainder', 'physical_kernel', 'excluded_tail', 'depth_model')
  assumption:independent_screen: bound; note: allowance supplied by caller: detail-eq: screen factorisatio...
  assumption:isotropic_pitch: bound; note: allowance supplied by caller: declared zero: each ray is uni...
screen error |direct exact - direct screen| / I: [0.007 0.011 0.    0.001 0.015 0.115 0.141 0.093]
screen_factorisation_bound / I: [0.03  0.045 0.064 0.104 0.181 0.252 0.165 0.094]
bound covers the screen error: True
direct screen route: assumption:independent_screen is bound
total time 5.4 s
```

Both assumption terms are now `bound`s with the note "allowance supplied
by caller", and they leave the `unbounded` list. The total stays
`unbounded` because `basis_remainder` (no `RemainderInputs`; example (b)
shows the probe), `physical_kernel`, `excluded_tail` (`Support.truncated`
is `None`) and `depth_model` are not supplied. The screen error is measured
as the difference of two direct averages, the exact per-emitter phase
against the factorised screen route; the bound covers it in every channel
and is nearly attained in the top channel, since two rays with equal
masses attain the Cauchy-Schwarz inequality at each node. The allowance
holds for the stated `sigma_P` and `Phi` only; `predict` does not check
them.

#### What the layer does not certify

* `physical_kernel` (`E_phys`) and `excluded_tail` (`E_tail`) are inputs.
* The `numerical` term is the per-column envelope `|C' - C|` of the basis
  against the same angular route at doubled nodes, contracted with `|m|`;
  it is an `estimate`, not a bound. The product-versus-tensor comparison is
  opt-in (`cross_route=True`, or `convergence="full"`) and is a named
  finite check in `provenance.finite_checks`, not a budget term.
* The remainder probe evaluates envelopes at finitely many points
  (`estimate`) and computes absolute moments on the samples it is given;
  those must be the predicted population. `from_samples(phase=None)`
  probes the basis phase route. The probe is refused for `N + 1` outside
  the orders validated against finite differences (`N <= 2`; the field
  `certified_orders` keeps its name for compatibility, and these are finite
  checks, not certificates); `predict(samples=...)` then leaves
  `basis_remainder` `unbounded` instead of raising.
* The zero `harmonic_truncation` bound assumes the population lies inside
  the declared `Support`. `HarmonicKernel.truncation_error`,
  `predict(samples=..., kernel=...)` and `direct_channel_average` check
  concrete samples: the term, and a declared-complete `excluded_tail`,
  become `unbounded` when an emitting sample lies outside. Without samples
  (or with traced samples) the hypothesis stays unchecked; the
  harmonic-tail note says so, and with traced samples so does the note of
  a declared-zero `excluded_tail`.
* A measured assumption term holds for the supplied population only.
  Assumptions forced by the basis (screen routes, the continuum kernel's
  `isotropic_pitch`) stay `unbounded` whatever the moments carry, because
  the retained moments cannot test them, unless the caller supplies an
  allowance through `assumption_allowances`. An allowance keeps its kind
  and value (`not_applicable` is refused); nothing checks that it holds.
  A moment-level discrepancy is attributed to the first assumption record
  only; `assume()` keeps it only when the other maps add no constraint.
* The amplitude slot is `delta N_src (|C m| + e)` with `e` the valued
  per-electron error slots (allowances divided by `N_src`), in `predict`
  and in `direct_channel_average`. A missing `amplitude_uncertainty` leaves
  it `unbounded`; a concrete `0` is a zero `bound` (amplitude declared
  exact); a negative value raises.
* A fit's `bias_bound` is `|K| |delta|` for the discrepancy the caller
  supplied, with `delta = |R| E + |delta R| (|S_hat| + E)`: over the
  unknowns `u` for `fit_linear`, and over `z` for `fit_bfgs`, where it is an
  `estimate` linearised at the optimum (Gauss-Newton `K = J^+ L^-1`, prior
  and log-Jacobian curvature excluded, linearisation error not bounded).
  It is `unbounded` without a declared discrepancy, with an observing
  response but no `response_uncertainty`, and for a rank-deficient design;
  a linear fit's is an `estimate` when the plug-in `|delta R| |S_hat|` term
  is nonzero. The fit's `statistical_input` adds `|dm/dx| bias_bound` to
  the one-sigma noise and is `unbounded` when the bias bound is. A nodal
  fit has no bias bound and an `unbounded` `statistical_input`: the softmax
  gauge leaves its Fisher matrix singular. Rank decisions use the
  column-equilibrated design, so they do not depend on the `Reference`
  scales. Feasibility checks are necessary conditions, not a proof that the
  fitted moments come from a nonnegative population.
* A map with fitted hyper-parameters records NaN placeholders (closure kind
  `"<kind> (fitted)"`), so the moments' pytree structure does not depend
  on `theta`; the fitted values are in `theta` and in a provenance note of
  the `FitResult` and of its `prediction`. Every `to_dict()` (`FitResult`,
  `Prediction`, `JointMoments`, `Provenance`) writes non-finite floats as
  `null`, so its output is strict JSON.

<!-- docs:model:end -->
### Mixed emission and Faraday rotation
<!-- docs:mixed:start -->

In the pure-rotation model, with no incident background, absorption, scattering
or conversion, the observed polarisation is
`P(nu) = N_src * E[K_P(nu, p) * exp(2j*lambda**2*depth)]`.
The positive measure counts source electrons. Depth is in rad/m² and wavelength
in metres; K_P is complex Q+iU in the observer's sky basis. Depth, source energy,
field and orientation may all be correlated. Different emitters can have
different spectral shapes; no common spectral factor is imposed.

`emission_polarisation(emission, depths, lam, weights=None, source_column=1.)`
computes this discrete average. Depths have shape `(n,)`. Emission is scalar,
`(n,)` for wavelength-independent per-emitter values, or exactly
`(n, *lam.shape)` for per-emitter spectra. The result has `lam.shape`. Relative
nonnegative weights are normalized internally; the physical `source_column`
is a separate finite nonnegative scalar. Complex emissivity is never normalized
as a probability weight. A zero source column gives zero emission; the supplied
measure must still be well defined. The original `screen_polarisation` retains
its existing scalar/per-ray input contract and foreground interpretation.

`faraday_depth_practical(n_e_cm3, B_par_uG, s_pc)` in `synchro.rm` integrates
from **each node to the last node**, using the rounded 0.812 coefficient and
trapezoids. Positions increase towards the observer. Field reversals are
allowed; depth need not be monotone. Add any exterior foreground depth to all
nodes. For spatial quadrature, use weights proportional to source density times
path-quadrature weights and set N_src to their total. Across unresolved rays,
include each ray's column as well as its fixed nonnegative ray weight before
normalizing. Chromatic instrumental weights belong in the observing response.
This NumPy preprocessing routine is not differentiable; gradients with respect
to supplied depths in the JAX average are supported. Path quadrature error is
not inferred from the grid spacing.

For a finite spectral response, represent the intrinsic per-source kernel as
`sum_a c_a(lam)*psi_a(p) + r`. The following **synthetic exact-basis** example
keeps its intrinsic source error zero and bounds only the phase truncation:

```python
import jax.numpy as jnp
from synchro.faraday import (
    emission_polarisation, joint_faraday_moments, joint_faraday_average,
)

lam = jnp.linspace(0.0, 0.8, 9)
x = jnp.array([-0.8, 0.2, 1.0])
depth = jnp.array([-0.4, 0.1, 0.6])
weights = jnp.array([1.0, 2.0, 4.0])
psi = jnp.stack([jnp.ones_like(x), x, jnp.exp(2j*x)], axis=-1)
c = jnp.stack([1 + lam, 0.3j*lam, -0.2 + lam**2], axis=-1)
emission = psi @ c.T
reference = emission_polarisation(emission, depth, lam, weights, source_column=5.)
M, absolute_next = joint_faraday_moments(
    psi, depth, 8, weights, reference_depth=0.1,
)
prediction, envelope = joint_faraday_average(
    c, M, lam, reference_depth=0.1, source_column=5.,
    absolute_next=absolute_next, source_error=0.,
)
assert jnp.all(jnp.abs(prediction-reference) <= envelope + 1e-13)
```

`joint_faraday_moments` takes basis values `(n,n_basis)` and a static nonnegative
phase degree L, and returns `M[a,b]=E[psi_a*(depth-reference_depth)**b]` for
`b=0..L`, plus `A[a]=E[abs(psi_a)*abs(depth-reference_depth)**(L+1)]`.
The moment matrix can instead be fitted directly; its shape fixes the degree
in `joint_faraday_average`. With coefficients `(*lam.shape,n_basis)`, that
routine returns the prediction and absolute envelope, both `lam.shape`:

```text
prediction = N_src*exp(it*reference_depth) * sum_ab c_a*(it)^b/b!*M_ab
error      = N_src*(source_error + |t|^(L+1)/(L+1)! * sum_a |c_a|*A_a)
t          = 2*lam^2
```

`source_error` bounds E|r| per source and is required, as is `absolute_next`.
For continuous populations they must have independent support/derivative or
distributional justification; a sampled estimate alone is not a certificate.
Retained moments do not determine these additional inputs. A complex zeroth
moment can vanish while mixed moments still contribute. Joint cumulants can
parametrise the same finite statistics; marginal depth cumulants alone cannot.
Moment realizability and uncertainty, excluded tails, numerical error and
physical-model discrepancy remain separate requirements. Large phase ranges
can need more modes or direct quadrature; no universal low-order accuracy is
claimed. Strong phases also require control of floating-point argument reduction.

The real-space source/depth pairing is sufficient for pure rotation; no recovery
of a unique spatial geometry is implied. At fixed source measure and column,
paired perturbations give
`|delta P| <= N_src*(E|delta K_P| + 2*lambda²*E[|K_P|*|delta depth|])`, using
the reference K_P in the second term. Different columns or weights add their
normalization errors. For a frequency channel, integrate the **combined**
emitted/rotated spectrum and propagate the envelope with the absolute observing
weights; rotating an already integrated channel generally gives a different
answer. These functions support JIT and gradients (phase degree is static when
collecting moments). Test cases include the sinc slab, an independent transfer
matrix exponential, swapped layers, correlated spectra and analytic gradients.

<!-- docs:mixed:end -->
## Physical transfer and reduced demonstrations
<!-- docs:transfer:start -->

```python
from synchro.los_moments import moment_driven_slab_cgs

# Local electron-number moments M_k=int N(gamma)(gamma-gamma0)^k dgamma.
# This narrow-population illustration does not certify an energy PDF closure.
gamma0, number_density = 2500.0, 1e-12
S = moment_driven_slab_cgs(
    1e8, gamma0, 5e-6,
    number_density, 0.0, number_density*50.0**2,
    number_density/gamma0,                 # illustrative inverse moment approximation
    3.0856775814913673e21,
    n_e=0.03, B_par=2e-6, phi=0.2,
)
```

The physical interface requires frequency Hz, fields Gauss, length cm and
thermal/electron densities cm^-3. Output is erg/s/cm²/Hz/sr. It restores the
separate emission and absorption prefactors; they do not cancel in a physical
source function. Supply a measured or independently computed inverse moment
when precision requires it. For finite support, pass the three endpoint terms
`[N(gamma)*(gamma-gamma0)^k]_lower^upper` through `boundary_terms`; zero defaults
assert that these vanish or are separately represented. Distributional jumps at
physical hard cutoffs must be treated consistently.

The wrapper assumes isotropic ultra-relativistic electrons and a second-order
local energy-kernel Taylor approximation. It uses h=gamma² for absorption;
`derivative_weighted(..., relativistic=True)` evaluates the exact radial
h=gamma*sqrt(gamma²-1) on a NumPy grid, with its own discretization error. The
wrapper omits intrinsic V and alphaV; their propagated error is not bounded by
this illustration. Generic `mueller_matrix` retains all Stokes absorption terms.

Cold-plasma `mueller_rotation` is the signed Stokes rV, twice the position-angle
rate returned by `rotation_coefficient`. `mueller_conversion` and
`conversion_coefficient` both return signed natural rQ, with no further factor
two. Natural conversion mixes U and V; sky azimuth rotates rQ/rU. These leading
high-frequency cold-dielectric coefficients need distribution-dependent
replacements for hot or non-thermal plasma. The Gaussian Burn formula is an
external-screen average, not internal emission and rotation; a uniform emitting
slab has sinc depolarisation at finite Faraday depth.

`moment_driven_slab` remains a reduced-unit teaching interface and rejects
physical Faraday arguments. Use the CGS wrapper to combine those effects.
A uniform slab uses one augmented exponential. `transfer_los` handles ordered,
nonuniform slabs; varying-medium discretization must be checked separately.
A source-column coordinate rescaling preserves zero-emissivity derivatives.
`max_squarings=32` is the static exponential budget; tests include scalar optical
depth 10^6, but do not guarantee arbitrary matrix depth, conditioning or gain.
Exceeding the budget may produce NaN; extreme gain may overflow.

Magnus order two uses a running-prefix commutator, O(N) matrix work instead of
an explicit O(N²) pair sum. It supports unequal slab widths and rejects orders
other than 1 or 2. The commutator is deterministic ordering information, not a
statistical cumulant. Ensemble transfer also needs source–propagator dependence.

<!-- docs:scope:start -->
<!-- docs:transfer:end -->
## Module boundaries

| Layer | Modules | Output / responsibility |
|---|---|---|
| Shared constants | `constants` | Common SI-to-CGS constants aligned with the manuscript |
| Numerical reference functions | `bessel`, `ultrarel` | Values, derivatives and finite-tail controls; physical continuum error remains separate |
| Physical response | `stokes`, `sed`, `derivatives` | Declared normalization and coordinates; power-law/curvature hypotheses explicit |
| Statistical contraction | `expansion`, `cumulants` | Finite kernel average; no automatic positive-PDF closure |
| Population precompute | Grid functions in `kirchhoff`, practical RM/depth integrals | NumPy integrations; not traced or differentiable |
| Online coefficients / propagation | Moment contractions in `kirchhoff`, `conversion`, `rm`, `transfer`, `los_moments` | JAX scalars/arrays; CGS and reduced paths explicit |
| Ordered approximation / limits | `magnus`, `solutions` | Declared finite Magnus order and analytic limits |
| Finite joint response and spectral fits | `model` (`index`, `errors`, `channels`, `phase`, `screens`, `moments`, `bounds`, `assumptions`, `kernels`, `harmonic`, `basis`, `predict`, `adapters`, `fit.observation`, `fit.linear`, `fit.nonlinear`, `fit.diagnostics`, `fit.result`) | Channel-integrated Stokes from joint moments with an `ErrorBudget` (`bound`/`estimate`/`measured`/`unbounded`); opt-in assumptions as `ParameterMap`s; linear, BFGS and nodal fits with identifiability and feasibility reports; `E_phys` and `E_tail` remain inputs |
| Independent checks | `tests/`, diagnostic scripts | SciPy/analytic/finite-difference oracles and explicit finite test ranges |

## Compatibility and scientific scope

This review corrects signed Q, the conversion sign/factor/axis, fixed-B physical
derivatives, and absolute isotropically averaged emissivity (an omitted angular
probability factor made the legacy result twice too large). Physical predictions
using those APIs will change. Dimensionless harmonic defaults are preserved.
Invalid PDF/moment assumptions are not made valid by the corrected numerics.

The accompanying manuscript supplies a total-error propagation interface and
bounded independent PDF benchmarks. A complete Galactic angular/field/energy-tail,
plasma and instrument error budget is still an input to scientific sufficiency.
Population identifiability remains future work.

## Citation and license

Please cite Zhang & Chluba, *A statistical framework for synchrotron emission*.
MIT license; see [LICENSE](https://github.com/zzhang0123/synchro/blob/main/LICENSE).

<!-- docs:scope:end -->