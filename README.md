# synchro

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

## Statistical interfaces

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

## Independent symbolic derivation

[`derivation/`](derivation/README.md) contains local Wolfram scripts that derive
and check the helical orbit, retarded radiation, absolute harmonic powers and
Stokes basis conventions from the classical equations. Run
`wolframscript -file derivation/verify.wls` to regenerate the report. It also
compares the complex harmonic amplitudes with direct Fourier integration of the
original electric field, independently of this package's Python implementation.

## Install and verify

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

## Harmonic radiation and derivatives

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

## Finite statistical response

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

### Mixed emission and Faraday rotation

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

## Physical transfer and reduced demonstrations

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
MIT license; see [LICENSE](LICENSE).
