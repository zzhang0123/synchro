# Independent Wolfram derivation of the single-electron kernel

Run from the package checkout with a licensed local Wolfram installation:

```sh
wolframscript -file derivation/verify.wls
```

The script exits nonzero on a failed check and writes `verification.json` beside
itself. The saved report records the Wolfram version, script SHA-256 hashes,
individual residuals, derived expressions and finite numerical cases.
`verification.txt` is the console output of the recorded run. No Python package,
JAX implementation, manuscript validation function or saved numerical target is
loaded. The scripts use core Wolfram Language; no xAct installation is needed.

## Conventions and physical inputs

The starting laws are the relativistic Lorentz force, the retarded far-zone
acceleration electric field and Gaussian-cgs Poynting flux. Maxwell's equations
and the retarded Green function are not rederived. The model is a prescribed
classical helical orbit in a uniform magnetic field, with no radiation reaction,
plasma propagation or finite-distance corrections.

- The electron has charge `-charge`, where `charge > 0`.
- `n = {0,0,1}` points from the source to the observer.
- `bp` and `bl` are beta perpendicular and parallel to the magnetic field;
  `omegaB > 0` is the relativistic gyrofrequency.
- The Fourier transform uses `exp(+i omega t)`, so a positive-frequency real-field
  reconstruction uses `exp(-i omega t)`.
- The ordered natural sky basis is parallel to the projected magnetic field,
  then `eperp = Cross[n, eparallel]`. Its cross product is `n`.
- `Q` is parallel power minus perpendicular power, `U = 2 Re(Apar Aperp*)`,
  and `V = -2 Im(Apar Aperp*)`, all with the same power prefactor.
- The harmonic index in the manuscript and derivation is `m`; the public Python
  API retains its existing argument name `n`.
- Harmonic powers are per source time, per solid angle, integrated over a line.
  Dividing by arrival time instead adds `1/D`.

For the optional North/East sky alignment, the axes agree with the
[NASA LAMBDA description of IAU coordinates](https://lambda.gsfc.nasa.gov/product/about/pol_convention.html).
[van Straten et al. (2009), section 2.1](https://arxiv.org/pdf/0912.1662)
uses `exp(+i omega t)` reconstruction and IAU `V = +2 Im(Ex Ey*)`.
Conjugating both amplitudes gives the convention above for the same real field.
The script checks this conjugation and the circular state `(1,i)/sqrt(2)`, which
has `V=+1` here. No handedness or phase correction is fitted to a numerical result.

## What the scripts derive

`verify.wls` follows the physical calculation in six steps:

1. Construct the bases, solve the Lorentz ODE with `DSolveValue`, integrate the
   velocity, and derive the retarded arrival time and observed harmonic period.
2. Establish the radiation integration-by-parts identity for an arbitrary
   trajectory, then project the original helical radiation field onto the sky.
3. Expand the two exponentials in `exp(i x sin(psi))`. Fourier orthogonality
   selects the terms with `s=r+m`; `Sum` evaluates their factorial series to a
   Bessel function. This derives the coefficient instead of installing the
   manuscript's Bessel answer as an integration rule. Periodicity and
   differentiation give the cosine and sine moments and both complex amplitudes.
4. Derive absolute power from Poynting flux, positive-frequency Parseval, the
   finite-time line kernel and the delta-function Jacobian. The line kernel is
   nonnegative, has mass `2 pi`, and has outside-epsilon mass bounded by
   `8/(T epsilon)`; its delta limit is distributional. No total-power calibration
   enters the harmonic prefactor.
5. Form the Jones coherency matrix and derive all four Stokes parameters, full
   single-harmonic polarisation, and rotation to a fixed sky frame. The general
   rotation is checked for arbitrary complex amplitudes, including nonzero U:
   `Qsky = Q cos(2 phi) - U sin(2 phi)`,
   `Usky = Q sin(2 phi) + U cos(2 phi)`, with I and V unchanged.
6. Independently integrate the source-time Poynting flux of the original
   acceleration field over solid angle. This gives the total radiated power and
   the cooling-to-orbit ratio without using the harmonic sum.

The main symbolic domain is `m >= 1` integer, `bp > 0`, `bp^2+bl^2 < 1`,
`0 < theta < pi` and positive physical scales. The field-axis sky basis is a
coordinate choice, because the magnetic projection vanishes there. The exact
Bessel recurrence removes the apparent `1/sin(theta)` amplitude singularity;
finite direct-field checks include both viewing axes.

`numerical_checks.wls` supplies a second route: integrate the **original retarded
vector electric field** in source time, including `dt/dt'`, and compare its
complex Fourier coefficients with `i omega_m exp(-i m phase0) A_m` rotated to the
fixed sky frame. It checks the absolute complex phase as well as I, Q, U and V.
Eight cases cover harmonics 1–25, Lorentz factors 1.2–5, both signs of parallel
velocity, both viewing hemispheres, nonzero field azimuth and initial phase,
and axial nonzero/zero harmonics. The analytic expressions are never used to
construct the direct-field integrand.

A separate harmonic sum and angular quadrature at gamma=6/5 and alpha=pi/4 uses
cutoffs 32 and 64 and working precisions 25 and 35. It compares with the total
power derived in step 6. Reversing V or deleting the delta Jacobian must produce
a detectable failure against the field integral; these are sensitivity controls.

## Interpretation of the saved checks

The symbolic identities establish the stated formulas within the declared
classical far-zone model and assumptions. The high-precision integrations and
harmonic-tail refinement are finite numerical checks, not uniform error bounds.
Residuals reported as arbitrary-precision zero reflect numerical resolution,
not an exact identity; the direct-field comparisons require scaled residuals
below `1e-20`. They do not certify arbitrary Lorentz factors, package floating-point accuracy,
population averaging, propagation, an instrument model or a Galactic foreground.
This is a formula derivation; it does not execute or validate the Python API.

## Comparison with Pandya et al. (2016)

Run `wolframscript -file derivation/compare_pandya.wls` for a separate algebraic
comparison with the printed Eqs. (3)–(11) of arXiv:1602.08749v1. It records the
frequency/resonance mapping, I/Q/U agreement, opposite V sign, and absolute
line-power normalization in `pandya_comparison.json`. The original 44 checks
validate the internal derivation; they did not compare it with Pandya's paper.
The authors explicitly corrected the negative kernel to positive in
[symphony commit f63fc87 (2017-09-10)](https://github.com/AFD-Illinois/symphony/commit/f63fc87ce2e6a3403a767a3eb113a1904698f343),
`src/integrator/integrands.c`, to restore their stated IEEE/IAU convention.
The comparison also checks that all four corrected kernels match ours. This is
an algebraic source comparison, not a run of the external C implementation.

Run `wolframscript -file derivation/check_handedness.wls` for an independent
physical sign check. It solves the signed Lorentz equations for an axial
circular orbit, evaluates the real retarded electric field, and measures its
rotation around the propagation direction before introducing a Fourier
amplitude or a Stokes definition. It then checks circular intensities, opposite
viewing direction, conjugate Fourier convention, reversed charge, and the axial
limit of the manuscript's Bessel formula. Results are in `handedness.json`.
For an electron with B along propagation, the real field rotates in the
right-handed sense and V/I=+1. Reversing propagation with a corresponding
right-handed sky basis gives V/I=-1. A proper rotation of observer coordinates
does not change physical V; a reflection of one transverse axis changes the
component formula's sign and must be included when translating conventions.
