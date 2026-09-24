"""Reference point, declared support and discrete population samples.

Private helper of ``synchro.model.moments`` (split to keep files short); the
public names are re-exported there. LABEL: the displacement coordinates of
``eq: local response remainder`` (``Reference.z``), the declared support of
``eq: channel error budget`` (``Support``) and the discrete electron-number
measure of ``eq: joint emission depth`` (``PopulationSamples``).

Units: ``gamma`` dimensionless (``> 1``), ``B`` Gauss (``> 0``), ``depth``
rad/m^2, ``mu, eta`` in ``[-1, 1]``, ``phi`` radians; scales carry the units
of their variable. Weights are relative electron-number masses normalised as
in ``synchro.rm._screen_samples``. Shape errors raise ``ValueError`` at trace
time; value errors use ``equinox.error_if``. Nothing here certifies that a
population lies inside its declared support or that a discrete measure
approximates a continuous one.
"""

from __future__ import annotations

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp

from ..rm import _screen_samples

VARIABLES = ("gamma", "B", "mu", "eta", "phi", "depth")


def _as_float(value, name):
    value = jnp.asarray(value)
    if jnp.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    return value.astype(jnp.result_type(value, 1.0))


def _finite(value, name):
    return eqx.error_if(value, jnp.any(~jnp.isfinite(value)), f"{name} must be finite")


def _scalar(value, name):
    value = _as_float(value, name)
    if value.ndim != 0:
        raise ValueError(f"{name} must be a scalar")
    return _finite(value, name)


def _interval(pair, name, *, above=None):
    if len(pair) != 2:
        raise ValueError(f"{name} support must be a (lo, hi) pair")
    lo, hi = (_scalar(v, f"{name} support") for v in pair)
    lo = eqx.error_if(lo, lo > hi, f"{name} support needs lo <= hi")
    if above is not None:
        lo = eqx.error_if(lo, lo <= above, f"{name} support must lie above {above}")
    return (lo, hi)


class Reference(eqx.Module):
    """Reference point and scales of the displacement coordinates ``z``.

    ``gamma0 > 1``, ``B0 > 0`` (Gauss), ``depth_ref`` (rad/m^2) and positive
    ``scales = (s_gamma, s_B, s_depth)``; all traced scalar leaves. Value
    checks use ``equinox.error_if``. ``describe`` needs concrete values.
    ``z(gamma, B, depth)`` returns ``(..., 3)`` dimensionless displacements
    (``eq: local response remainder``); ``gamma`` dimensionless, ``B`` in
    Gauss, ``depth`` in rad/m^2. Assumes nothing about the population;
    the choice of reference and scales is the caller's, and nothing here
    certifies that the population is local to it.
    """

    LABEL: ClassVar[str] = "eq: local response remainder"
    gamma0: jax.Array
    B0: jax.Array
    depth_ref: jax.Array
    scales: tuple[jax.Array, jax.Array, jax.Array]

    def __init__(self, gamma0, B0, depth_ref=0.0, scales=(1.0, 1.0, 1.0)):
        if len(scales) != 3:
            raise ValueError("scales must be (s_gamma, s_B, s_depth)")
        gamma0 = _scalar(gamma0, "gamma0")
        self.gamma0 = eqx.error_if(gamma0, gamma0 <= 1, "gamma0 must exceed 1")
        B0 = _scalar(B0, "B0")
        self.B0 = eqx.error_if(B0, B0 <= 0, "B0 must be positive")
        self.depth_ref = _scalar(depth_ref, "depth_ref")
        checked = []
        for value, name in zip(scales, ("s_gamma", "s_B", "s_depth")):
            value = _scalar(value, name)
            checked.append(eqx.error_if(value, value <= 0, f"{name} must be positive"))
        self.scales = tuple(checked)

    def z(self, gamma, B, depth) -> jax.Array:
        """Dimensionless displacements ``(..., 3)`` of broadcast inputs."""
        gamma, B, depth = jnp.broadcast_arrays(
            jnp.asarray(gamma), jnp.asarray(B), jnp.asarray(depth)
        )
        s_gamma, s_B, s_depth = self.scales
        return jnp.stack(
            [
                (gamma - self.gamma0) / s_gamma,
                (B - self.B0) / s_B,
                (depth - self.depth_ref) / s_depth,
            ],
            axis=-1,
        )

    def describe(self) -> tuple:
        """Hashable provenance tuple of concrete floats (eager use only)."""
        return (
            ("gamma0", float(self.gamma0)),
            ("B0", float(self.B0)),
            ("depth_ref", float(self.depth_ref)),
            ("scales", tuple(float(s) for s in self.scales)),
        )


class Support(eqx.Module):
    """Declared population support ``gamma in (lo, hi)``, ``B``, ``depth``.

    ``gamma`` lies above 1 and ``B`` above 0 (Gauss); ``depth`` in rad/m^2.
    ``truncated`` is a static tri-state for the excluded-tail term of
    ``eq: channel error budget``: ``None`` (unknown, ``excluded_tail``
    unbounded), ``False`` (declared complete, ``declared_zero``) and ``True``
    (a tail bound is a required input and the amplitude counts only the
    retained column). Nothing here verifies that a population lies inside.
    Shapes: three ``(lo, hi)`` pairs of scalar leaves. Assumes only what
    ``truncated`` declares; the excluded tail ``E_tail`` is never computed
    by the package.
    """

    LABEL: ClassVar[str] = "eq: channel error budget"
    gamma: tuple[jax.Array, jax.Array]
    B: tuple[jax.Array, jax.Array]
    depth: tuple[jax.Array, jax.Array]
    truncated: bool | None = eqx.field(static=True)

    def __init__(self, gamma, B, depth, truncated=None):
        if truncated is not None and not isinstance(truncated, bool):
            raise ValueError("truncated must be None, False or True")
        self.gamma = _interval(gamma, "gamma", above=1.0)
        self.B = _interval(B, "B", above=0.0)
        self.depth = _interval(depth, "depth")
        self.truncated = truncated

    def tail_kind(self) -> str:
        """``"unbounded"`` | ``"declared_zero"`` | ``"required_input"``."""
        if self.truncated is None:
            return "unbounded"
        return "required_input" if self.truncated else "declared_zero"

    def describe(self) -> tuple:
        pairs = [
            (name, (float(lo), float(hi)))
            for name, (lo, hi) in (
                ("gamma", self.gamma),
                ("B", self.B),
                ("depth", self.depth),
            )
        ]
        return (*pairs, ("truncated", self.truncated))

    def gamma_max(self) -> float:
        return float(self.gamma[1])

    def B_min(self) -> float:
        return float(self.B[0])


class PopulationSamples(eqx.Module):
    """Discrete electron-number measure on ``(gamma, B, mu, eta, phi, depth)``.

    All six arrays are real ``(S,)``; ``gamma >= 1``, ``B >= 0``,
    ``|mu|, |eta| <= 1``, ``phi`` in radians, ``depth`` in rad/m^2. Optional
    ``weights`` ``(S,)`` are relative, nonnegative, with positive total mass;
    they are normalised on use, so ``1e300``-scale weights are safe. Shape
    errors raise ``ValueError``; value errors use ``equinox.error_if``.
    ``B`` in Gauss, ``depth`` in rad/m^2, ``gamma`` dimensionless. A
    discrete measure carries no assumption; averages over it are exact for
    the measure and are not certified as approximations of any continuous
    population (sampling and quadrature error are inputs). ``[extension]``.
    """

    LABEL: ClassVar[str] = "[extension] discrete population measure"
    gamma: jax.Array
    B: jax.Array
    mu: jax.Array
    eta: jax.Array
    phi: jax.Array
    depth: jax.Array
    weights: jax.Array | None = None

    def __init__(self, gamma, B, mu, eta, phi, depth, weights=None):
        arrays = {
            name: _as_float(value, name)
            for name, value in zip(VARIABLES, (gamma, B, mu, eta, phi, depth))
        }
        shape = arrays["gamma"].shape
        if len(shape) != 1 or shape[0] == 0:
            raise ValueError("samples must be nonempty 1D arrays")
        if any(value.shape != shape for value in arrays.values()):
            raise ValueError("all six sample arrays must share the shape (S,)")
        arrays = {name: _finite(value, name) for name, value in arrays.items()}
        arrays["gamma"] = eqx.error_if(
            arrays["gamma"], jnp.any(arrays["gamma"] < 1), "gamma must be >= 1"
        )
        arrays["B"] = eqx.error_if(
            arrays["B"], jnp.any(arrays["B"] < 0), "B must be >= 0"
        )
        for name in ("mu", "eta"):
            arrays[name] = eqx.error_if(
                arrays[name],
                jnp.any(jnp.abs(arrays[name]) > 1),
                f"{name} must lie in [-1, 1]",
            )
        for name, value in arrays.items():
            setattr(self, name, value)
        if weights is not None:
            weights = _as_float(weights, "weights")
            if weights.shape != shape:
                raise ValueError("weights must have the sample shape (S,)")
            _, _ = _screen_samples(arrays["gamma"], weights, sample_name="gamma")
        self.weights = weights

    @property
    def size(self) -> int:
        return int(self.gamma.shape[0])

    def normalised_weights(self) -> jax.Array:
        """Weights summing to one (uniform when ``weights`` is ``None``)."""
        return _screen_samples(self.gamma, self.weights, sample_name="gamma")[1]

    @classmethod
    def product(cls, **marginals) -> "PopulationSamples":
        """Outer product of per-variable marginals (test helper).

        Each of the six keywords is an array of values (uniform weights) or a
        tuple ``(values, weights)``. The result enumerates every combination
        with product weights; it is a product measure by construction.
        """
        missing = [name for name in VARIABLES if name not in marginals]
        extra = [name for name in marginals if name not in VARIABLES]
        if missing or extra:
            raise ValueError(
                f"product needs exactly {VARIABLES}; missing {missing}, extra {extra}"
            )
        values, weights = [], []
        for name in VARIABLES:
            entry = marginals[name]
            if isinstance(entry, tuple) and len(entry) == 2:
                v, w = (_as_float(x, name) for x in entry)
            else:
                v = _as_float(entry, name)
                w = jnp.ones_like(v)
            if v.ndim != 1 or w.shape != v.shape:
                raise ValueError(
                    f"{name} marginal must be 1D values with matching weights"
                )
            values.append(v)
            weights.append(w / jnp.sum(w))
        grids = [g.ravel() for g in jnp.meshgrid(*values, indexing="ij")]
        mass = jnp.ones(())
        for g in jnp.meshgrid(*weights, indexing="ij"):
            mass = mass * g
        return cls(*grids, weights=mass.ravel())


__all__ = ["Reference", "Support", "PopulationSamples", "VARIABLES"]
