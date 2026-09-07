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
| Population precompute | Grid functions in `kirchhoff`, practical RM integral | NumPy integrations; not traced or differentiable |
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
