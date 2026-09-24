# Changelog

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
