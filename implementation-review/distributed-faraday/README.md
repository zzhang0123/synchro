# Distributed pure-Faraday implementation check

Base revision: f421c5a959c0842b08261e48e9cec7552f515954. This local extension adds the joint emission-depth average and finite-statistic response documented in the package README, with a downstream-depth preprocessing helper and preserved external-screen compatibility.

Run from this checkout with its dependencies installed:

```bash
PYTHONPATH=. python -m pytest tests/test_distributed_faraday.py tests/test_rm_regressions.py tests/test_package_improvements.py tests/test_transfer_regressions.py -q
python -m black --check synchro/faraday.py synchro/rm.py synchro/__init__.py tests/test_distributed_faraday.py
python -m ruff check synchro/faraday.py synchro/rm.py synchro/__init__.py tests/test_distributed_faraday.py
```

Actual result: 78 related tests passed in 20.72 s. Four affected tests were rerun after the final complex-input rejection change and passed in 1.52 s. Black/Ruff/whitespace checks and the README example passed. The active checkout import was verified; runtime JAX 0.10.0, Equinox 0.13.7, NumPy 2.3.5, Python 3.12.

The tests include the analytic uniform-slab sinc law, an independent transfer-matrix exponential, exchanged emitting layers, per-emitter frequency dependence, ray-column normalization, signed-channel error propagation, finite-moment remainder checks, a normalized positive-mixture spectral fit, analytic gradients, JIT and invalid/overflow inputs. The fit is a synthetic same-family check, not validation on real data.

Pure Faraday rotation is assumed; the new API does not solve absorption/conversion or construct a physical 3D medium. Source error and next absolute moments are required inputs for a truncation envelope. Feasibility, support/tail uncertainty, numerical/path errors and physical discrepancy remain separate. Full-package/GPU tests were not run. No commit, push or release was performed.
