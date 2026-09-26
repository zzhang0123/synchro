# SED fits of identifiable combinations

`syncmoments.model.fit.reduce_response` and
`syncmoments.model.fit.fit_combinations` fit the linear combinations of the
joint moments that a finite-band channel SED constrains, and report the rest
as unconstrained. The design, the formulas and the test record are in
Section 10.1 of [the design document](../DESIGN.md). This page describes the
worked example of manuscript Section 5.3 (Appendix C.1) and what its numbers
mean.

## Running the example

```
PYTHONPATH=. python scripts/sed_reconstruction_example.py
PYTHONPATH=. python scripts/sed_reconstruction_example.py --quick --out /tmp/sed
```

The example needs SciPy; Matplotlib is optional and TeX is not used. The
full run takes about 25 s on a laptop. `--quick` uses reduced grids (18
centres, 6 fitted) and its numbers are not comparable with the full run.
`--historical PATH` prints a side-by-side table against the research
reference `validation/sed_reconstruction_results.json` of the manuscript
repository, which the script reads and never modifies.

The scripts are:

| File | Content |
|---|---|
| `scripts/sed_reconstruction_example.py` | command line and the steps below |
| `scripts/sed_reconstruction_setup.py` | configuration, basis, truth, mock data |
| `scripts/sed_reconstruction_direct.py` | independent NumPy/SciPy population integration |
| `scripts/sed_reconstruction_checks.py` | validation checks |
| `scripts/sed_reconstruction_report.py` | JSON, NPZ and CSV outputs, provenance |
| `scripts/sed_reconstruction_summary.py` | printed summary and historical table |
| `scripts/sed_reconstruction_figure.py` | the figure |

Outputs, written to `figures/` by default:

- `sed_moment_reconstruction.json`: configuration, the reduction and fit
  records, per-slot arrays over the 60 full coordinates (truth, projected
  truth, representative, unresolved truth, measured bias, retained-subspace
  noise, slot status), the retained combinations, the SED arrays, residuals
  on every centre with the withheld mask, the validation checks, the
  illustrative prior and the provenance (package version, git revision and
  status, SHA-256 of every package module and example script, library
  versions, command line).
- `sed_moment_reconstruction_arrays.npz`: `C`, `T`, `H`, `L`, `Q`, the
  estimators `K` and `K_beta`, `Pi`, the covariance, the combination rows
  `B`, the four direction sets, the singular values, the noise realisation
  and the noise standard deviations.
- `sed_moment_reconstruction_combinations.csv`: one row per retained
  combination.
- `sed_moment_reconstruction.pdf` and `.png`: the figure.

The repository keeps the figure and the CSV. The JSON and NPZ of the 0.4.0
run are attached to the
[v0.4.0 release](https://github.com/zzhang0123/syncmoments/releases/tag/v0.4.0);
the script regenerates all five files in about 20 s.

## Steps

```python
reduction = reduce_response(
    basis, relations="continuum", representatives=MANUSCRIPT_REPRESENTATIVES
)
a_true = reduction.layout.full_vector(moments, amplitude=1.0)
fit = fit_combinations(basis, data, reduction=reduction, max_sigma=0.1)
comparison = fit.against_truth(a_true, discrepancy=direct - forward, noise=noise)
```

1. Before any data, `reduce_response` groups the 60 full coordinates (52
   active response columns and 8 real depth rows of `M0` whose response is
   zero) into 30 groups `q = T a` with `C = H T`. The relations come from
   the continuum scaling identity and are exact for `ContinuumKernel`
   (Section 10.1 of the design document). `MANUSCRIPT_REPRESENTATIVES`
   (defined in the example) picks the representatives `q_0, q_g, q_BB, q_d,
   q_gd, q_dd` of the manuscript; any other choice changes only `H`, `T` and
   the labels.
2. The truth is computed twice: through `PopulationSamples` and
   `JointMoments.from_samples` on a product grid, and in closed form from
   the conditional angular densities. The two agree to 1e-14.
3. The mock data come from an independent full-population integration in
   NumPy and SciPy (Bessel integral for `F`, direct `eta` quadrature, the
   exact `phi` average), not from the truncated response that is fitted.
   I, Q and U of 24 of the 72 centres carry Gaussian noise with standard
   deviation 1 % of the direct I (seed 20260925); the other 48 centres are
   withheld.
4. `fit_combinations` keeps the modes with noise standard deviation
   `1/s_i <= 0.1` in the default metric (Euclidean in the full coordinates,
   amplitude fitted as `a_0`). The threshold is an illustration, not a
   recommended value.

## Three separate claims

- **Analytic redundancy.** The 30 relations of step 1 hold for the continuum
  kernel `K = B Phi(B gamma^2)` at any reference scales, channels and phase
  route. They are not claimed for `HarmonicKernel` or
  `PolynomialTestKernel`, where `relations="auto"` keeps only structural
  zeros. In the example the analytic null space has 30 dimensions.
- **Numerical rank.** 26 singular values of the whitened grouped design
  exceed `rank_tol s_max = 1e-8 s_max`. This holds for the stated channels,
  mask, noise, metric and reference scales only. The 26th value is 1.058
  times the cutoff and the 27th is 0.245 times it, so a small change of
  quadrature or scales could move the rank to 24. The rank is recorded, not
  asserted.
- **Practical recoverability.** 9 modes meet `1/s_i <= 0.1`; 17 are
  numerically nonzero but weaker, and 4 are below the rank tolerance. The
  9th singular value is 19.50 and the 10th is 7.61, against the cutoff 10.

## What a good SED fit determines

The fit determines the 9 combinations `beta = B a` and nothing else. The
60-entry representative `a_hat` is `Pi a` plus noise and bias; it carries 9
independent numbers and is not a recovered tensor. The other 51 directions
are unconstrained by the data: the fit reports their contribution as
`unbounded`, and a zero projection or a zero projected covariance in such a
slot is not a physical zero or a bound. Nothing in the fit checks that
`a_hat` is realisable by a positive population.

The continuum rescaling `B -> kappa B`, `gamma -> gamma/sqrt(kappa)` with the
source column divided by `kappa` leaves the direct continuum SED unchanged
(5.2e-16 of I at `kappa = 1.15` on every ninth centre). It moves the supports
(gamma to 7460-11190, B to 4.6-6.9 microgauss) and changes the moments; the
full coordinates differ by up to 0.65 (the amplitude slot `a_0 = 1/kappa`).
The two populations give the same data but their true `beta` differ by up to
2.56 noise standard deviations (in `beta_1`), because `beta` is defined
through the truncated response and the truncation error of the two
populations differs. A fixed-support external prior may remove this
ambiguity; the SED does not.

## Metric

The singular values, the retained set and `beta` depend on the declared
coefficient metric. The default is Euclidean in the full coordinates `a`,
which are `z`-moments at the basis `Reference` scales times the
source-column ratio `A = N_src/N_*`. The fit factors `T = L Q` with `Q Q^T =
I`, so `C = (H L) Q` and the whitened design `H L` has the singular values of
`C` in that metric. An SVD of `H` alone would change the metric and the
meaning of the cutoff. Changing the reference scales, `N_*` or the `metric`
argument changes `beta`, the singular values and the retained set.

## Error terms

With data `d = C a + delta + n`, the estimator satisfies `a_hat - Pi a = K
(delta + delta_C a) + K n` exactly (`delta_C = C - H T` is zero to rounding
for an exact reduction). The fit reports four terms separately:

| Term | In the example |
|---|---|
| declared discrepancy `bias = abs(K) delta` | `unbounded`: no discrepancy is declared |
| unresolved directions `(I - Pi) a` | `unbounded` without a coefficient bound |
| approximate reduction | `not_applicable`: the reduction is exact |
| measured bias `K delta` | from the synthetic truth only, through `against_truth` |

The omitted terms (basis remainder, kernel physics, excluded tail,
absorption, transfer and calibration) enter the fit only through a declared
`data.discrepancy`. The JSON also records an illustrative refit with
`coefficient_bounds(layout, support, amplitude_bound=2.0)`: the unresolved
term then has kind `bound` and covers `abs((I - Pi) a_true)` elementwise.
That bound rests on the declared Support and amplitude bound and does not
cover the excluded tail; it is not a data constraint.

## Results of the full run

Run with the DEV environment (Python 3.12, JAX 0.10.0, NumPy 2.3.5, SciPy
1.16.3) and the NEW environment (JAX 0.10.2, NumPy 2.5.3, SciPy 1.18.1) on
package revision e7e22af with uncommitted T-006 and T-007 changes (recorded
in the provenance). The historical values come from the research reference
at package b39cf0c.

| Quantity | Historical | DEV | NEW |
|---|---|---|---|
| full / active / grouped / retained | 60/52/30/9 | 60/52/30/9 | 60/52/30/9 |
| numerical rank (margin) | 26 | 26 (1.058) | 26 (1.058) |
| chi2 / dof | 55.2361 / 63 | 55.2361 / 63 | 55.2361 / 63 |
| withheld RMS / max over I | 0.4764 % / 2.0211 % | same to 2e-15 | same to 1e-15 |
| forward/direct max over I | 0.4899 % | same to 1e-14 | same to 2e-15 |
| top 16 singular values | | 2.0e-14 relative | 2.7e-14 relative |
| grouped vs ungrouped coefficients | 1.4e-15 | 1.7e-15 | 1.7e-15 |
| decomposition residual | | 1.4e-15 | 9.6e-16 |
| Monte Carlo covariance diagonal (10000 draws) | 0.0254 | 0.0254 | 0.0254 |
| direct quadrature change over I | 1.5e-7 | 1.5e-7 | 1.5e-7 |
| basis refinement change over I | 1.0e-6 | 1.0e-6 | 1.0e-6 |
| fit refinement coefficient change | 5.9e-6 | 5.9e-6 | 5.9e-6 |
| rescaling invariance over I | 5.2e-16 | 5.2e-16 | 5.2e-16 |

The grouped-ungrouped difference stays at rounding level, below the
standard SVD perturbation estimate for the retained subspace, `eps s_max /
(s_9 - s_10) = 1.5e-14`. The ungrouped path decomposes a 72 x 52 design (the
8 structural zeros removed) where the historical run used 72 x 60. Under
basis refinement
(`n_eta = 96`, `n_nu = 32`, `n_nodes_F = 192`) `T` is unchanged, 9 modes are
retained, the retained subspaces turn by at most 1.8e-5 rad and `beta_hat`
moves by at most 4e-4 of its noise standard deviation.

The retained combinations (DEV; NEW agrees to 3e-15):

| i | true | fitted | 1/s_i | measured bias | noise part / sigma |
|---|---|---|---|---|---|
| 1 | 0.61217 | 0.61426 | 0.00127 | 7.7e-4 | 1.04 |
| 2 | 0.15647 | 0.15571 | 0.00178 | 2.2e-4 | -0.54 |
| 3 | 0.08615 | 0.08880 | 0.00178 | 1.8e-4 | 1.38 |
| 4 | 0.52981 | 0.50222 | 0.00689 | -5.7e-3 | -3.18 |
| 5 | -0.12088 | -0.12554 | 0.00895 | 1.5e-3 | -0.69 |
| 6 | -0.06351 | -0.07054 | 0.00895 | 6.8e-4 | -0.86 |
| 7 | 0.00496 | 0.03153 | 0.03279 | 3.3e-4 | 0.80 |
| 8 | 0.01397 | 0.07291 | 0.03279 | -1.9e-3 | 1.86 |
| 9 | -0.56907 | -0.60254 | 0.05129 | -6.1e-4 | -0.64 |

`beta_4` lies 4.0 standard deviations from its true value: a measured
truncation bias of -0.83 sigma plus a noise part of -3.18 sigma in this seed.
For 9 independent standard normals the largest absolute value reaches 3.18
with probability 1.3 %. The Monte Carlo checks confirm the variances (0.0254
on the coefficient diagonal, 0.020 on `beta`), and the seed is the historical
one; it was not changed.

The pairs (2, 3), (5, 6) and (7, 8) share a singular value (the Q/U
symmetry of the design). Only their two-dimensional subspace is determined;
the package fixes a canonical basis inside each pair from the observation,
not from the data. The individual values of these six combinations therefore
differ from the manuscript's historical values, which used arbitrary SVD
vectors; the subspaces agree to 1e-14 rad and the norms of `beta_hat` within
each pair agree to the printed digits. `beta_1`, `beta_4` and `beta_9` agree
with the historical values directly. The members of each pair are equal to
rounding, so the `beta` correlations inside a pair are at rounding level
(at most 1.6e-15, recorded in the notes), and `K`, `Pi` and the covariance
do not depend on the chosen basis. Pivot norms within `max(1e-8, 1e3 u s_max
/ sep)` of the largest count as tied, so DEV and NEW also choose the same
basis for the weak pairs (agreement 3.6e-10); the numerical-null directions
keep the SVD basis and differ between environments.

## Known limitations

- Analytic grouping covers `ContinuumKernel` only.
- The numerical rank sits 5.8 % above its cutoff; the retained set depends
  on `max_sigma`, `rank_tol`, the noise model, the masks, the metric and the
  reference scales.
- Inside a cluster only the subspace is determined; the canonical basis is
  a convention. The `beta` of a cluster are correlated by at most
  `(rho^2 - 1)/2`, `rho` the cluster's `s_max/s_min` (noted per cluster).
- `a_hat` is not a moment vector; no positivity or realisability test is
  applied.
- Valued unresolved terms need a declared coefficient bound, which does not
  cover the excluded tail.
- The example's agreement with history is a finite check at one seed and
  configuration. It does not show physical adequacy, unique moment recovery
  or an exact global rank. Isotropic pitch, the finite angular family, the
  compact energy segment and the absence of absorption, transfer,
  calibration and excluded tails are declared assumptions of the example.
