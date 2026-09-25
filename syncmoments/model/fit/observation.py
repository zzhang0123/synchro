"""Channel Stokes data with noise, mask, observing response and discrepancy.

LABEL: ``extra eq: finite fit model`` (``d = A C m + delta + eta``) and
``extra eq: data error propagation`` (``|R| @ envelope`` for a Stokes-space
discrepancy). :class:`StokesData` is the data side of ``fit_linear``,
``fisher``, ``LogDensity`` and ``identifiability``: it whitens the retained
data rows with the Cholesky factor of their noise covariance and applies the
optional linear observing response to a model column space.

Layout and units
----------------
* ``stokes``: ``(n_ch, 4)`` channel Stokes ``I, Q, U, V`` in the units of
  the prediction (channel-integrated power for ``unit_peak`` channels),
  flattened channel-major (row ``4 j + s`` is Stokes ``s`` of channel ``j``).
  With a ``response`` the data live in its row space and ``stokes`` is the
  data vector ``(n_data,)`` (a ``(n_ch, 4)`` array is accepted when
  ``4 n_ch == n_data`` and is flattened channel-major).
* ``noise``: ``(n_data,)`` variances or ``(n_data, n_data)`` covariance of
  ``eta``, same units squared. ``n_data = 4 n_ch`` without a response.
* ``mask``: rows that are KEPT (``True``/1). ``(4,)`` broadcasts to
  ``(n_ch, 4)``; with a response a mask over the data rows ``(n_data,)`` is
  accepted, and the Stokes layout only when ``n_data == 4 n_ch``. The mask
  must be concrete (it fixes the static row count); it is stored as a leaf
  and the kept rows as a static tuple.
* ``response``: ``(n_data, 4 n_ch)`` real matrix ``R`` mapping the flattened
  channel Stokes to the data; ``response_uncertainty`` is ``|delta R|`` of
  the same shape (nonnegative). ``fit_linear`` adds ``|delta R| (|S_hat| +
  E)`` (the second term of ``extra eq: data error propagation``, with its
  fitted channel Stokes ``S_hat``) to ``discrepancy_vector()`` in its bias
  bound, and reports that bound ``unbounded`` when a response is set
  without ``response_uncertainty`` (zeros declare the response exact).
* ``discrepancy``: an :class:`ErrorTerm` bounding ``|delta|`` either in
  Stokes space (``(n_ch, 4)``, ``(4,)`` or scalar; propagated by ``|R|``)
  or in data space (``(n_data,)``).

Shape and dtype errors raise ``ValueError`` at trace time; value errors
(non-finite data, nonpositive variances, an asymmetric or non-positive
definite covariance block, negative response uncertainty) use
``equinox.error_if``. ``whitened``, ``whiten``, ``design``, ``covariance``
and ``discrepancy_vector`` are ``jax.jit`` and autodiff safe.

Not certified: that ``noise`` describes the actual measurement errors, that
the response is exact (a missing ``response_uncertainty`` leaves the second
term of ``extra eq: data error propagation`` unbounded, and ``fit_linear``
then reports an ``unbounded`` bias bound), or that the discrepancy term
bounds the true model error.
"""

from __future__ import annotations

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..errors import ErrorTerm

N_STOKES = 4
# relative asymmetry tolerated in a covariance (float64 roundoff of a sum)
SYMMETRY_TOL = 1e-12


def _as_real(value, name):
    value = jnp.asarray(value)
    if jnp.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    return value.astype(jnp.result_type(value, 1.0))


def _finite(value, name):
    return eqx.error_if(value, jnp.any(~jnp.isfinite(value)), f"{name} must be finite")


def _concrete_mask(mask):
    try:
        array = np.asarray(mask)
    except Exception as exc:  # jax tracers cannot be materialised
        raise ValueError("mask must be concrete (not traced)") from exc
    if array.dtype != bool:
        if not np.all(np.isin(array, (0, 1))):
            raise ValueError("mask entries must be booleans or 0/1")
        array = array.astype(bool)
    return array


class StokesData(eqx.Module):
    """Channel Stokes data, noise, mask, observing response and discrepancy.

    See the module docstring for shapes and units. Masked rows are removed
    inside :meth:`whitened` (and :meth:`whiten`, :meth:`design`,
    :meth:`covariance`, :meth:`discrepancy_vector`) and from :meth:`dof`;
    :meth:`n_data` counts the rows before masking. The object is immutable;
    build a new one to change any field. Units: ``stokes`` and ``noise`` in
    the units of the prediction they are compared with (``noise`` squared).
    Assumes Gaussian noise with the supplied variances or covariance and a
    linear observing ``response``; ``discrepancy`` and
    ``response_uncertainty`` are caller declarations, not certified here.
    """

    LABEL: ClassVar[str] = "extra eq: finite fit model"
    stokes: jax.Array
    noise: jax.Array
    mask: jax.Array | None = None
    response: jax.Array | None = None
    discrepancy: ErrorTerm | None = None
    response_uncertainty: jax.Array | None = None
    _kept: tuple[int, ...] = eqx.field(static=True)
    _n_ch: int = eqx.field(static=True)

    def __init__(
        self,
        stokes,
        noise,
        mask=None,
        response=None,
        discrepancy=None,
        response_uncertainty=None,
    ):
        stokes = _as_real(stokes, "stokes")
        self.response = _check_response(response, stokes)
        n_ch, n_data = _layout(stokes, self.response)
        self.stokes = _finite(stokes, "stokes")
        self.noise = _check_noise(noise, n_data)
        self.mask, self._kept = _check_mask(mask, n_ch, n_data, self.response)
        self.discrepancy = _check_discrepancy(discrepancy, n_ch, n_data)
        self.response_uncertainty = _check_uncertainty(
            response_uncertainty, self.response
        )
        self._n_ch = n_ch

    # ---------------------------------------------------------------- layout
    @property
    def n_ch(self) -> int:
        """Number of channels of the model Stokes space (``4 n_ch`` columns)."""
        return self._n_ch

    def n_data(self) -> int:
        """Rows of the data vector before masking (``4 n_ch`` without a response)."""
        return int(self.noise.shape[0])

    def n_kept(self) -> int:
        """Rows retained by the mask."""
        return len(self._kept)

    def kept_rows(self) -> tuple[int, ...]:
        """Static indices of the retained data rows, increasing."""
        return self._kept

    def data_vector(self) -> jax.Array:
        """The data ``d`` as ``(n_data,)`` before masking (channel-major)."""
        return self.stokes.reshape(-1)

    def _rows(self):
        return jnp.asarray(self._kept, dtype=int)

    def select(self, x) -> jax.Array:
        """Kept rows of an ``(n_data, ...)`` array."""
        x = jnp.asarray(x)
        if x.ndim == 0 or x.shape[0] != self.n_data():
            raise ValueError(f"expected a leading axis of length {self.n_data()}")
        return jnp.take(x, self._rows(), axis=0)

    # ------------------------------------------------------------- response
    def design(self, columns) -> jax.Array:
        """``R @ columns`` (or ``columns``) on the kept rows: ``(n_kept, n_cols)``.

        ``columns`` is ``(4 n_ch, ...)`` in the flattened channel-major
        Stokes layout, e.g. a response matrix ``C`` from ``SpectralBasis``.
        """
        columns = jnp.asarray(columns)
        if columns.ndim == 0 or columns.shape[0] != N_STOKES * self._n_ch:
            raise ValueError(
                f"columns must have {N_STOKES * self._n_ch} rows (4 n_ch), "
                f"got shape {columns.shape}"
            )
        if self.response is not None:
            columns = jnp.tensordot(self.response, columns, axes=1)
        return self.select(columns)

    # ------------------------------------------------------------ whitening
    def covariance(self) -> jax.Array:
        """Noise covariance of the kept rows, dense ``(n_kept, n_kept)``."""
        rows = self._rows()
        if self.noise.ndim == 1:
            return jnp.diag(self.noise[rows])
        block = self.noise[jnp.ix_(rows, rows)]
        scale = jnp.max(jnp.abs(block))
        asym = jnp.max(jnp.abs(block - block.T))
        return eqx.error_if(
            block, asym > SYMMETRY_TOL * scale, "covariance must be symmetric"
        )

    def _whitening(self) -> jax.Array:
        """``L^-1`` with ``Sigma_kept = L L^T`` (lower triangular)."""
        if self.noise.ndim == 1:
            return jnp.diag(1.0 / jnp.sqrt(self.noise[self._rows()]))
        cov = self.covariance()
        L = jnp.linalg.cholesky(cov)
        L = eqx.error_if(
            L,
            jnp.any(~jnp.isfinite(L)),
            "covariance block of the kept rows must be positive definite",
        )
        eye = jnp.eye(L.shape[0], dtype=L.dtype)
        return jax.scipy.linalg.solve_triangular(L, eye, lower=True)

    def whitened(self) -> tuple[jax.Array, jax.Array]:
        """``(L^-1 d, L^-1)`` on the kept rows: ``(n_kept,)`` and ``(n_kept, n_kept)``.

        ``L`` is the lower Cholesky factor of the kept covariance block (the
        square roots of the variances when ``noise`` is one-dimensional), so
        ``(L^-1)^T L^-1 = Sigma_kept^-1`` and ``||L^-1 (d - G u)||^2`` is the
        chi-square of ``extra eq: finite fit model``.
        """
        Linv = self._whitening()
        return Linv @ self.select(self.data_vector()), Linv

    def whiten(self, x) -> jax.Array:
        """``L^-1 @ x[kept]`` for an ``(n_data, ...)`` array (data or design)."""
        return jnp.tensordot(self._whitening(), self.select(x), axes=1)

    # ---------------------------------------------------------- discrepancy
    def discrepancy_vector(self) -> jax.Array | None:
        """``|delta|`` on the kept rows ``(n_kept,)``; ``None`` without a valued term.

        A Stokes-space term (``(n_ch, 4)``, ``(4,)`` or scalar) is
        propagated by ``|R|`` (the first term of ``extra eq: data error
        propagation``); a data-space term ``(n_data,)`` is used as is. The
        second term ``|delta R S_hat|`` needs a Stokes estimate and is added
        by ``fit_linear`` from ``response_uncertainty``, not here.
        """
        term = self.discrepancy
        if term is None or term.value is None:
            return None
        value = term.value
        if value.shape == (self.n_data(),):
            # data space (with a response), or the flattened Stokes layout
            return self.select(value)
        full = jnp.broadcast_to(value, (self._n_ch, N_STOKES)).reshape(-1)
        if self.response is not None:
            full = jnp.abs(self.response) @ full
        return self.select(full)

    # ------------------------------------------------------------------ dof
    def dof(self, n_params: int) -> int:
        """``n_kept - n_params`` (signed; a rank-deficient fit should use its rank)."""
        if isinstance(n_params, bool) or not isinstance(n_params, (int, np.integer)):
            raise ValueError("n_params must be an int")
        if n_params < 0:
            raise ValueError("n_params must be nonnegative")
        return self.n_kept() - int(n_params)


# ---------------------------------------------------------------- validation


def _check_response(response, stokes):
    if response is None:
        return None
    response = _as_real(response, "response")
    if (
        response.ndim != 2
        or response.shape[1] % N_STOKES != 0
        or response.shape[1] == 0
    ):
        raise ValueError(
            f"response must be (n_data, 4 n_ch), got shape {response.shape}"
        )
    return _finite(response, "response")


def _layout(stokes, response):
    if response is None:
        if stokes.ndim != 2 or stokes.shape[1] != N_STOKES or stokes.shape[0] == 0:
            raise ValueError(f"stokes must be (n_ch, 4), got shape {stokes.shape}")
        return int(stokes.shape[0]), N_STOKES * int(stokes.shape[0])
    n_data, n_cols = (int(v) for v in response.shape)
    n_ch = n_cols // N_STOKES
    if stokes.ndim == 1:
        if stokes.shape[0] != n_data:
            raise ValueError(
                f"stokes must have {n_data} entries (response rows), got {stokes.shape}"
            )
    elif stokes.ndim == 2:
        if stokes.shape != (n_ch, N_STOKES) or N_STOKES * n_ch != n_data:
            raise ValueError(
                "a two-dimensional stokes array with a response must be (n_ch, 4) "
                f"with 4 n_ch == n_data; got {stokes.shape} for response {response.shape}"
            )
    else:
        raise ValueError(f"stokes must be (n_data,) or (n_ch, 4), got {stokes.shape}")
    return n_ch, n_data


def _check_noise(noise, n_data):
    noise = _as_real(noise, "noise")
    if noise.shape == (n_data,):
        noise = _finite(noise, "variances")
        return eqx.error_if(noise, jnp.any(noise <= 0), "variances must be positive")
    if noise.shape == (n_data, n_data):
        return _finite(noise, "covariance")
    raise ValueError(
        f"noise must be ({n_data},) variances or ({n_data}, {n_data}) covariance, "
        f"got shape {noise.shape}"
    )


def _check_mask(mask, n_ch, n_data, response):
    if mask is None:
        return None, tuple(range(n_data))
    array = _concrete_mask(mask)
    stokes_layout = N_STOKES * n_ch == n_data
    if array.shape == (n_data,) and (response is not None or n_data != N_STOKES):
        flat = array
    elif array.shape in ((N_STOKES,), (n_ch, N_STOKES)) and stokes_layout:
        array = np.broadcast_to(array, (n_ch, N_STOKES))
        flat = array.reshape(-1)
    elif array.shape == (N_STOKES,) or array.shape == (n_ch, N_STOKES):
        raise ValueError(
            "a Stokes-layout mask (n_ch, 4) | (4,) needs n_data == 4 n_ch; "
            f"with this response the mask must be over the {n_data} data rows"
        )
    else:
        raise ValueError(
            f"mask must be ({n_ch}, 4), (4,) or ({n_data},), got shape {array.shape}"
        )
    kept = tuple(int(i) for i in np.flatnonzero(flat))
    if not kept:
        raise ValueError("mask removes every row")
    return jnp.asarray(array), kept


def _check_discrepancy(term, n_ch, n_data):
    if term is None:
        return None
    if not isinstance(term, ErrorTerm):
        raise ValueError("discrepancy must be an ErrorTerm or None")
    if term.value is None:
        return term
    shape = term.value.shape
    if shape in ((), (N_STOKES,), (n_ch, N_STOKES), (n_data,)):
        return term
    raise ValueError(
        f"discrepancy value must be scalar, (4,), ({n_ch}, 4) or ({n_data},), got {shape}"
    )


def _check_uncertainty(delta, response):
    if delta is None:
        return None
    if response is None:
        raise ValueError("response_uncertainty requires a response")
    delta = _as_real(delta, "response_uncertainty")
    if delta.shape != response.shape:
        raise ValueError(
            f"response_uncertainty must match the response shape {response.shape}, "
            f"got {delta.shape}"
        )
    delta = _finite(delta, "response_uncertainty")
    return eqx.error_if(
        delta, jnp.any(delta < 0), "response_uncertainty must be nonnegative"
    )


__all__ = ["StokesData", "N_STOKES", "SYMMETRY_TOL"]
