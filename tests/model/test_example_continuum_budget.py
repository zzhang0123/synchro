"""Working set of README example (d) (``scripts/model_examples_continuum.py``).

``incident_polarisation`` evaluates ``ContinuumKernel.kernels`` at every ray
sample. It maps over the samples with ``jax.lax.map`` in batches of
``kernel.samples_per_step(channels, 256)`` (the kernel's ``chunk_budget``
over the ``2 n_ch n_nu n_nodes_F`` ``F``/``G`` values of one sample) instead
of one ``vmap`` over the whole ray. Batching changes only the summation
order: the result equals the full ``vmap`` to ``1e-13`` relative.
"""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

import syncmoments  # noqa: F401  (float64)
from syncmoments.model.kernels import ContinuumKernel

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "scripts")
)
import model_examples_continuum as mec  # noqa: E402

RTOL = 1e-13  # batching changes only the summation order


def _ray_and_kernel(chunk_budget):
    channels, _, _, _ = mec.galactic_setup(n_ch=2)
    rays, _ = mec.two_ray_population()
    kernel = ContinuumKernel(n_nodes_F=16, chunk_budget=chunk_budget)
    return channels, rays[0], kernel


def _vmapped(kernel, channels, ray):
    """The full-ray ``vmap`` reference (the example's previous evaluation)."""
    nodes = jnp.asarray(channels.nodes)

    def one(gamma, B, eta, phi):
        return kernel.kernels(nodes, gamma, B, eta)[1] * jnp.exp(2j * phi)

    values = jax.vmap(one)(ray.gamma, ray.B, ray.eta, ray.phi)
    return jnp.einsum("s,scn->cn", ray.normalised_weights(), values)


def test_incident_polarisation_maps_in_budgeted_batches(monkeypatch):
    per_sample = 2 * 2 * 64 * 16  # F and G, n_ch n_nu n_nodes_F
    channels, ray, kernel = _ray_and_kernel(10 * per_sample)
    batches = []
    original = jax.lax.map

    def spy(f, xs, *args, **kwargs):
        batches.append(kwargs.get("batch_size"))
        return original(f, xs, *args, **kwargs)

    monkeypatch.setattr(jax.lax, "map", spy)
    mec.incident_polarisation(kernel, channels, ray)
    assert kernel.samples_per_step(channels, 256) == 10
    assert batches == [10]


def test_batched_incident_polarisation_equals_the_full_vmap():
    for budget in (1, 10 * 2 * 2 * 64 * 16, 1 << 20):
        channels, ray, kernel = _ray_and_kernel(budget)
        new = np.asarray(mec.incident_polarisation(kernel, channels, ray))
        ref = np.asarray(_vmapped(kernel, channels, ray))
        assert new.shape == ref.shape == (channels.n_ch, channels.n_nu)
        scale = np.max(np.abs(ref))
        assert scale > 0
        assert np.max(np.abs(new - ref)) <= RTOL * scale
