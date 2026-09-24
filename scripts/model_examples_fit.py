"""Example (c) of ``scripts/model_examples.py``: ``fit_linear`` on synthetic
continuum data (the configuration of example (b), twelve channels) with the
identifiability and feasibility reports.

Run through ``scripts/model_examples.py``; nothing printed is a certificate.
"""

from __future__ import annotations

import time

import numpy as np

import synchro  # noqa: F401  (enables float64)
from synchro.model.assumptions import isotropic_pitch
from synchro.model.basis import build_basis
from synchro.model.index import Truncation
from synchro.model.kernels import ContinuumKernel
from synchro.model.moments import JointMoments
from synchro.model.predict import predict

from model_examples_continuum import banner, galactic_population, galactic_setup


def example_c():
    banner(
        "(c) fit_linear on synthetic continuum data with identifiability and feasibility"
    )
    import jax.numpy as jnp

    from synchro.model.fit.diagnostics import feasibility_checks, identifiability
    from synchro.model.fit.linear import fit_linear
    from synchro.model.fit.observation import StokesData

    t0 = time.time()
    channels, reference, support, screen = galactic_setup(n_ch=12)
    samples = galactic_population()
    amplitude = 1e20
    rng = np.random.default_rng(0)

    for truncation in (
        Truncation(0, 0, 0, depth_degree=0),
        Truncation(0, 0, 1, depth_degree=0),
    ):
        basis = build_basis(
            ContinuumKernel(),
            channels,
            truncation,
            reference,
            support=support,
            phase=screen,
        )
        index = basis.index
        pmap = isotropic_pitch(index)
        truth = JointMoments.from_samples(samples, index, reference, parameter_map=pmap)
        clean = np.asarray(predict(basis, truth, amplitude=amplitude).stokes)
        sigma = 1e-3 * np.abs(clean[:, :1]) * np.ones((channels.n_ch, 4))  # 0.1 % of I
        stokes = clean + sigma * rng.standard_normal(sigma.shape)
        data = StokesData(
            stokes=jnp.asarray(stokes),
            noise=jnp.asarray(sigma.ravel() ** 2),  # (n_data,) variances, channel-major
            mask=jnp.array([True, True, True, False]),  # V is not modelled
        )
        print(
            f"-- {truncation}: n_real={index.n_real}, free={pmap.n_free()}, "
            f"data rows kept {data.n_kept()} of {data.n_data()}"
        )
        result = fit_linear(basis, data, pmap)
        print(
            f"rank={result.rank} chi2={float(result.chi2):.3f} dof={result.dof} "
            f"converged={result.converged} amplitude={float(result.amplitude):.6e}"
        )
        gap = np.abs(
            np.asarray(result.moments.to_vector()) - np.asarray(truth.to_vector())
        )
        print(
            f"max |m_fit - m_true| = {gap.max():.3e}; covariance is None: {result.covariance is None}"
        )
        report = identifiability(basis, data, pmap)
        print(
            f"identifiability: rank={report.rank} null_dim={report.null_dim} "
            f"weak(0.5)={report.weak(0.5)[:4]}"
        )
        for j in range(min(report.null_dim, 2)):
            print(f"  null direction {j}:", report.null_components(j)[:3])
        feas = feasibility_checks(result.moments, index, support)
        print("feasibility:", feas.statement)
        print("  failed:", feas.failed())
        print(
            "bias_bound kind:",
            result.bias_bound.kind if result.bias_bound is not None else None,
            "| prediction statistical_input kind:",
            result.prediction.budget.statistical_input.kind,
        )
    print(f"total time {time.time() - t0:.1f} s")
