# synchro

A differentiable JAX implementation of a perturbative statistical model of
synchrotron emission — the reference implementation for *A perturbative
statistical model for synchrotron emission* (Zhang & Chluba).

The idea it implements: the **deterministic** half of the model (the radiation
from one electron of given energy, pitch angle and local field) is treated
exactly, from the Schott harmonic decomposition; the **statistical** half (the
distribution of those conditions) is never modelled at all, only summarised by
a truncated hierarchy of statistics. Physics lives in the *derivative spectra*, which are
precomputed once by automatic differentiation; statistics enter afterwards as a
cheap tensor contraction.

## Requirements

Python 3.12 (developed against 3.12.9). The package itself needs only:

| package | version used |
|---|---|
| `jax` | 0.10.0 |
| `equinox` | 0.13.7 |
| `numpy` | 2.3.5 |

`scipy` (1.16.3) and `matplotlib` (3.10.8) are needed only for the validation
layer and the figure scripts. `scipy` is deliberately kept out of `synchro/`:
it serves as an **independent** reference against which the differentiable
kernels are checked, so no benchmark shares a code path with the quantity it
validates.

## Install

There is no packaging step — put the repository root on `PYTHONPATH`:

```bash
export PYTHONPATH=/path/to/this/repo
```

or prefix each command with `PYTHONPATH=.` as below.

Importing `synchro` enables `jax_enable_x64` as a global side effect. The
Bessel quadrature and the likelihood-style sums need float64, so import it
before creating any array.

## Quick start

### Exact single-particle Stokes at one harmonic

Valid for arbitrary β, not just the ultra-relativistic limit.

```python
from synchro.stokes import stokes_harmonic, larmor_power

I, Q, V = stokes_harmonic(n=10, gamma=5.0, alpha=0.785, theta=1.047)
# n=10: I=1.4359  Q=-0.7514  V=-1.2236

abs(I**2 - Q**2 - V**2)      # 2.2e-16 -- a single harmonic is fully polarised
larmor_power(5.0, 0.785, 1e-6)   # 1.903e-26 erg/s
```

Without `B` the Stokes parameters are normalised by `e^2 w_B^2 / (2 pi c)`;
pass `B` in Gauss for physical `dP_n/dOmega` in erg/s/sr.

### Moment and cumulant expansion

`build_expansion` does the expensive part once (autodiff of the Bessel-based
Stokes parameters at the reference point). The returned `CumulantExpansion` is
an `eqx.Module`: jittable, and differentiable with respect to the moments.

The two names are both deliberate. The **interface is raw moments** —
`__call__(mu, cov)` takes the mean and covariance, which is what an observer
measures — but the expansion is **organised in cumulants**, which is what makes
the truncation principled: for a Gaussian the `K=2` cumulant truncation is
exact to all Taylor orders, while the raw-moment series is not.

```python
import jax.numpy as jnp
from synchro.expansion import build_expansion

exp = build_expansion([1, 2, 5, 10, 20], gamma0=5.0, alpha0=0.785, theta0=1.047)

mu  = jnp.zeros(3)                                # <dgamma>, <dalpha>, <dtheta>
cov = jnp.diag(jnp.array([0.09, 0.0025, 0.0]))    # sigma_gamma=0.3, sigma_alpha=0.05
S = exp(mu, cov)                                  # (5, 3) -> I, Q, V per harmonic
# I at n=1,2,5,10,20: 0.2426 0.4623 0.9379 1.3845 1.6383

# the field magnitude is exact, not expanded: S scales by <(B/B0)^2>
S_B = exp.apply_B(S, B0=5e-6, mu_B=0.0, var_B=(2.5e-6)**2)   # ratio 1.2500
```

The reference point is baked into the precomputed spectra and is *not* a field
of the module, so it never enters the jit cache or the gradient.

**Domain of validity** (Sec. 4.7 of the paper): for a Gaussian the error is
fourth order in the fractional width, and the *angular* width is what binds —
`sigma_alpha <= 0.05 rad` keeps the second-order expansion under 1% out to
`n ~ 20`, while `sigma_gamma/gamma0` may be an order of magnitude larger at the
same accuracy. A genuinely isotropic pitch-angle distribution is far outside
this range and must be integrated exactly.

### Polarised radiative transfer

Moments in, observed Stokes out. With `polarised_absorption=True` the
emissivity and the absorption both come from the same moments through the
Kirchhoff closure, so the slab reproduces the correct self-absorbed
polarisation fraction.

```python
import numpy as np
from synchro.los_moments import moment_driven_slab

g  = np.linspace(1.0, 200.0, 2000)
N  = np.exp(-0.5 * (g - 30.0)**2 / 3.0**2)
dg = g - 30.0
M0, M1, M2 = np.trapezoid(N, g), np.trapezoid(N*dg, g), np.trapezoid(N*dg**2, g)
inv_gamma  = np.trapezoid(N/g, g)

S = moment_driven_slab(100.0, 30.0, 1.0, M0, M1, M2, inv_gamma,
                       L=1e4, polarised_absorption=True)
# I=5.0498e+04  Q/I=+0.5275
```

Pass `n_e`, `B_par`, `B_perp` to add cold-plasma Faraday rotation and
conversion. Circular absorption `alpha_V` is omitted throughout — it is
suppressed by O(1/gamma) ~ 1e-4 for the diffuse foreground.

## Running the checks

```bash
PYTHONPATH=. python3 -m pytest tests/ -q
```

50 tests, ~2 minutes. Every analytic identity quoted in the paper is asserted
here, together with boundary tests at each dispatch limit (optically thin and
thick, the Magnus convergence orders, the fourth-order convergence law).

```bash
PYTHONPATH=. python3 scripts/run_all.py
```

Runs the seven validation scripts with printed diagnostics *and* regenerates
all four figures, so `figures/*.pdf` cannot drift out of step with the physics.
Add `--no-figures` to skip the plotting. Individual scripts run standalone,
e.g. `PYTHONPATH=. python3 scripts/validate_kirchhoff.py`.

Coverage:

```bash
PYTHONPATH=. python3 -m pytest tests/ -q --cov=synchro --cov-report=term-missing
```

## Module map

| module | contents | paper |
|---|---|---|
| `bessel.py` | differentiable `J_n`, `J_n'`, `K_nu` via integral representations | App. C |
| `stokes.py` | exact Schott harmonic Stokes `(I, Q, V)`; Larmor power | Sec. 2.3–2.4 |
| `ultrarel.py` | ultra-relativistic `F(x)`, `G(x)` | Sec. 2.5 |
| `derivatives.py` | derivative spectra by autodiff | Sec. 4.4 |
| `expansion.py` | `CumulantExpansion`, `build_expansion`, exact `B` moments | Sec. 4.1–4.3 |
| `cumulants.py` | general-order cumulant expansion (Bell / Faà di Bruno) | Sec. 4.2 |
| `sed.py` | spectral index, curvature, LOS cumulants, absolute emissivity | Sec. 4.6 |
| `kirchhoff.py` | emissivity/absorption moments + Kirchhoff closure | Sec. 6.1 |
| `rm.py` | rotation measure, Burn depolarisation | Sec. 6.2 |
| `conversion.py` | cold-plasma Faraday rotation/conversion coefficients | App. D |
| `transfer.py` | Mueller matrix, exact slab and LOS chain | Sec. 6 |
| `magnus.py` | Magnus `Omega_1`, `Omega_2` | Sec. 6.3 |
| `solutions.py` | thin / self-absorbed / rotation limits | Sec. 6 |
| `los_moments.py` | end-to-end: moments → coefficients → transfer | Sec. 6 |

## Citation

If you use this package, please cite the accompanying paper (Zhang & Chluba,
*A moment expansion for synchrotron emission*).

## License

MIT — see [LICENSE](LICENSE).

## Performance notes

The Bessel functions are evaluated eagerly from a 512-point Gauss–Legendre
quadrature on every call, which is the right trade for the precompute path but
slow in a loop. The fast path for repeated evaluation is
`build_expansion` once, then `CumulantExpansion.__call__` (a single `einsum`,
jittable) — not repeated `stokes_harmonic`.

`transfer_slab` uses an augmented 5×5 matrix exponential with a normalised
source column; it is accurate up to roughly `tau ~ 1e3` per slab. Beyond that
the scaling-and-squaring in `expm` loses precision — discretise more finely.
