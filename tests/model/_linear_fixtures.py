"""Fixtures and NumPy oracles shared by ``test_linear*.py`` (not collected).

``setup`` builds a polynomial basis (``_nonlinear_oracles.polynomial_basis``)
on a given truncation; ``oracle_design`` assembles the whitened design
``L^-1 R C [c | P]`` and target with NumPy; ``with_amplitude`` attaches a
``log_amplitude`` to a ``Parameters``.
"""

import jax.numpy as jnp
import numpy as np

from _nonlinear_oracles import polynomial_basis
from syncmoments.model.assumptions import Parameters
from syncmoments.model.index import MomentIndex, Truncation

# -- helpers ---------------------------------------------------------------------


def setup(t=(1, 1, 1), n_ch=16, seed=3):
    index = MomentIndex.build(Truncation(*t))
    basis, C = polynomial_basis(index, n_ch=n_ch, seed=seed)
    return index, basis, C


def truth_of(pm, seed, amplitude=2.3):
    rng = np.random.default_rng(seed)
    flat = 0.5 * rng.standard_normal(pm.n_free())
    theta = pm.unflatten(jnp.asarray(flat))
    return flat, theta, amplitude


def stokes_of(C, pm, theta, amplitude, reference):
    m = np.asarray(pm(theta, reference).to_vector())
    return amplitude * C @ m, m


def oracle_design(C, pm, reference, data, *, amplitude=True):
    P, c = (np.asarray(x) for x in pm.affine_pieces(reference=reference))
    cols = np.column_stack([c, P]) if amplitude else P
    R = np.asarray(data.response) if data.response is not None else np.eye(C.shape[0])
    kept = list(data.kept_rows())
    noise = np.asarray(data.noise)
    cov = np.diag(noise) if noise.ndim == 1 else noise
    cov = cov[np.ix_(kept, kept)]
    L_inv = np.linalg.inv(np.linalg.cholesky(cov))
    G = L_inv @ (R @ C @ cols)[kept]
    d = L_inv @ np.asarray(data.data_vector())[kept]
    return G, d, L_inv, np.column_stack([c, P]) if not amplitude else cols


def with_amplitude(theta, A):
    return Parameters(
        log_amplitude=jnp.log(A),
        tables=theta.tables,
        hyper=theta.hyper,
        logits=theta.logits,
    )


__all__ = ["setup", "truth_of", "stokes_of", "oracle_design", "with_amplitude"]
