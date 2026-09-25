"""Hand-built bases, data and populations for the diagnostics tests.

Not collected by pytest (leading underscore). The real ``SpectralBasis`` and
``StokesData`` are used when their modules exist; otherwise private stubs
with the same attributes (``index``, ``reference``, ``channels``,
``response_matrix()``; ``stokes``, ``noise``, ``mask``, ``response``,
``whitened()``) stand in. The response-matrix layout of the stub follows
FINAL_DESIGN Section 3 (channel-major rows I, Q, U, V).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from syncmoments.model.channels import Channels
from syncmoments.model.index import MomentIndex
from syncmoments.model.moments import PopulationSamples, Reference, Support

try:  # real classes when the phase-2 modules have landed
    from syncmoments.model.basis import SpectralBasis as _RealBasis
except ImportError:  # pragma: no cover - depends on the phase-2 schedule
    _RealBasis = None
try:
    from syncmoments.model.fit.observation import StokesData as _RealData
except ImportError:  # pragma: no cover
    _RealData = None


class StubBasis(eqx.Module):
    """Minimal stand-in for ``SpectralBasis`` with random bases."""

    index: MomentIndex = eqx.field(static=True)
    channels: Channels
    reference: Reference
    I_basis: jax.Array
    V_basis: jax.Array
    P_basis: jax.Array

    @property
    def n_ch(self) -> int:
        return self.channels.n_ch

    def response_matrix(self) -> jax.Array:
        index, n_ch = self.index, self.n_ch
        pI = jnp.asarray(index.parity_I(), dtype=float)
        pV = jnp.asarray(index.parity_V(), dtype=float)
        C = jnp.zeros((4 * n_ch, index.n_real))
        n0, n2 = index.n0, index.n2
        a, b = jnp.real(self.P_basis), jnp.imag(self.P_basis)
        for j in range(n_ch):
            C = C.at[4 * j, :n0].set(self.I_basis[j] * pI)
            C = C.at[4 * j + 3, :n0].set(self.V_basis[j] * pV)
            C = C.at[4 * j + 1, n0 : n0 + n2].set(a[j])
            C = C.at[4 * j + 1, n0 + n2 :].set(-b[j])
            C = C.at[4 * j + 2, n0 : n0 + n2].set(b[j])
            C = C.at[4 * j + 2, n0 + n2 :].set(a[j])
        return C


class StubData(eqx.Module):
    """Minimal stand-in for ``StokesData`` (variances or covariance)."""

    stokes: jax.Array
    noise: jax.Array
    mask: jax.Array | None = None
    response: jax.Array | None = None
    discrepancy: object | None = None
    response_uncertainty: jax.Array | None = None

    def kept(self):
        n = self.stokes.size if self.response is None else self.response.shape[0]
        if self.mask is None:
            return np.arange(n)
        mask = np.broadcast_to(np.asarray(self.mask, bool), self.stokes.shape)
        return np.flatnonzero(mask.ravel())

    def n_data(self) -> int:
        return int(self.kept().size)

    def whitened(self):
        keep = self.kept()
        d = jnp.asarray(self.stokes).ravel()[keep]
        noise = jnp.asarray(self.noise)
        if noise.ndim == 1:
            L_inv = jnp.diag(1.0 / jnp.sqrt(noise[keep]))
        else:
            L = jnp.linalg.cholesky(noise[np.ix_(keep, keep)])
            L_inv = jnp.linalg.inv(L)
        return L_inv @ d, L_inv


def bump_channels(n_ch):
    centres = 1e9 * (1.0 + np.arange(n_ch))
    return Channels.bump(centres, 0.3 * centres)


def make_basis(index, rng, n_ch=3, reference=None):
    """Basis with random ``I/V/P`` arrays (real class when available)."""
    if reference is None:
        reference = Reference(5.0, 1.2, 0.3, (2.0, 0.7, 1.5))
    channels = bump_channels(n_ch)
    I = jnp.asarray(rng.standard_normal((n_ch, index.n0)))
    V = jnp.asarray(rng.standard_normal((n_ch, index.n0)))
    P = jnp.asarray(
        rng.standard_normal((n_ch, index.n2))
        + 1j * rng.standard_normal((n_ch, index.n2))
    )
    if _RealBasis is not None:
        return _real_basis(index, channels, reference, I, V, P)
    return StubBasis(index, channels, reference, I, V, P)


def _real_basis(index, channels, reference, I, V, P):
    from syncmoments.model.basis import KernelTerms, SpectralBasis
    from syncmoments.model.errors import ErrorTerm, Provenance

    support = Support((2.0, 9.0), (0.3, 3.0), (-3.0, 3.0))
    terms = KernelTerms(*(ErrorTerm.unbounded("test") for _ in range(4)))
    provenance = Provenance(
        "test", (), (), (), (), (), (), (), (), "test", (), (0, 1, 2), (), ()
    )
    return SpectralBasis(
        index=index,
        truncation=index.truncation,
        channels=channels,
        reference=reference,
        support=support,
        I_basis=I,
        V_basis=V,
        P_basis=P,
        kernel_terms=terms,
        provenance=provenance,
    )


def make_data(n_ch, rng, *, mask=None, response=None, covariance=False):
    """Random Stokes data with variances (or a covariance) and optional mask."""
    n = 4 * n_ch if response is None else response.shape[0]
    stokes = rng.standard_normal((n_ch, 4)) if response is None else None
    if covariance:
        A = rng.standard_normal((n, n))
        noise = A @ A.T + n * np.eye(n)
    else:
        noise = rng.uniform(0.5, 2.0, n)
    if response is not None:
        stokes = rng.standard_normal(n)
    cls = _RealData if _RealData is not None else StubData
    kwargs = {}
    if mask is not None:
        kwargs["mask"] = jnp.asarray(mask)
    if response is not None:
        kwargs["response"] = jnp.asarray(response)
    return cls(jnp.asarray(stokes), jnp.asarray(noise), **kwargs)


def random_population(rng, size=40):
    """Correlated atoms inside ``gamma in [3, 8], B in [0.5, 2], depth in [-2, 2]``."""
    t = rng.uniform(-1, 1, size)
    return PopulationSamples(
        gamma=5.5 + 2.0 * t + 0.3 * rng.uniform(-1, 1, size),
        B=1.25 + 0.5 * t**2 + 0.2 * rng.uniform(-1, 1, size),
        mu=np.clip(0.7 * t + 0.25 * rng.standard_normal(size), -1, 1),
        eta=np.clip(-0.5 * t + 0.3 * rng.standard_normal(size), -1, 1),
        phi=0.4 + 1.5 * t + 0.3 * rng.standard_normal(size),
        depth=1.4 * t + 0.3 * rng.uniform(-1, 1, size),
        weights=rng.uniform(0.2, 3.0, size),
    )


def population_support():
    return Support((3.0, 8.0), (0.5, 2.0), (-2.0, 2.0))


def reference_222():
    return Reference(5.0, 1.2, 0.3, (2.0, 0.7, 1.5))
