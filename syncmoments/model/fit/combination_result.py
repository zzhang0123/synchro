"""Result types of ``fit_combinations`` (``syncmoments.model.fit.combination_result``).

LABEL: ``extra eq: finite fit model`` (the fitted combinations of ``d = R C a
+ delta + n``), ``extra eq: data error propagation`` (the propagated
discrepancy) and ``[extension]`` (combination fits).

A :class:`CombinationFit` reports, in the fit coordinates ``x`` (``a = a_off
+ J x``, ``n_x`` unknowns named by ``labels``): the ``n_R`` retained
combinations ``beta = B x`` with ``beta_hat`` and ``beta_sigma`` (``1/s``;
inside a cluster the square root of the covariance diagonal, and the ``beta``
of a cluster may be correlated, see the notes), the estimator ``K``
``(n_x, n_kept)`` with ``x_hat = K (d - offset)``, the resolution operator
``Pi = W B`` ``(n_x, n_x)``, the noise covariance (retained subspace only;
``K``, ``Pi`` and it do not depend on a cluster's basis), the retained,
weak, numerical-null and analytic-null directions (``Subspaces``, each
``(n_x, k)`` and ``M_x``-orthonormal) and four separate error terms: ``bias`` (declared
discrepancy, ``|K| delta``, the bias of ``x_hat`` relative to ``Pi x``),
``unresolved`` (``(I - Pi) x``: ``unbounded`` unless a coefficient bound is
declared), ``reduction_bias`` (approximate relations only) and, through
:meth:`CombinationFit.against_truth`, the measured bias of a synthetic study.

``representative`` ``(n_full,)`` is ``a_off + J x_hat``: a representative of
``Pi a`` plus noise and bias with only ``n_retained`` independent fitted
numbers. It is not a moment vector, it is not checked for feasibility and a
positive population need not realise it; a zero projection or a zero
projected covariance in a null slot is neither a physical zero nor a bound.
Units: ``x``, ``a`` and ``beta`` dimensionless (source-column weighted
``z``-moments), Stokes arrays in the units of the basis response, ``chi2``
dimensionless. Not certified: the noise model, the declared discrepancy and
coefficient bound, the physical adequacy of the model, unique moment recovery
and any global rank statement; the numerical rank and the retained set hold
for the stated channels, mask, response, noise, metric, scales and thresholds.
"""

from __future__ import annotations

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..errors import ErrorTerm, Provenance
from . import _combination_errors as err
from ._combination_json import fit_json, observable_json, truth_json
from .reduction import ResponseReduction

LABEL = (
    "extra eq: finite fit model",
    "extra eq: data error propagation",
    "[extension]",
)


class Subspaces(eqx.Module):
    """Directions in ``x``: ``retained``, ``weak``, ``numerical_null``, ``analytic_null``.

    Each ``(n_x, k)``, ``M_x``-orthonormal; together they span ``R^{n_x}``.
    Dimensionless. The split depends on declared thresholds and the noise
    model (module docstring); it does not certify a physical null space.
    """

    LABEL: ClassVar[str] = "[extension] combination subspaces"
    retained: jax.Array
    weak: jax.Array
    numerical_null: jax.Array
    analytic_null: jax.Array


class ObservableSummary(eqx.Module):
    """A linear observable ``J_obs x`` (or ``J_obs a``) of a :class:`CombinationFit`.

    ``value`` ``(n_obs,)``, ``noise_sigma`` and ``noise_covariance`` (the
    retained subspace only), ``gain = J K`` ``(n_obs, n_kept)`` (so callers
    can form ``J K delta``), the error terms ``bias``, ``reduction_bias`` and
    ``unresolved`` (``(n_obs,)`` each) and ``null_sensitivity`` (``J W`` for
    the weak, numerical-null and analytic-null directions). :meth:`total`
    adds the three error envelopes; noise stays separate. Units: those of
    ``J_obs`` times ``x``. Not certified: see the module docstring.
    """

    LABEL: ClassVar[tuple[str, ...]] = LABEL
    value: jax.Array
    noise_sigma: jax.Array
    noise_covariance: jax.Array
    gain: jax.Array
    bias: ErrorTerm
    reduction_bias: ErrorTerm
    unresolved: ErrorTerm
    null_sensitivity: dict
    note: str = eqx.field(static=True, default="")

    def total(self) -> ErrorTerm:
        """``bias + reduction_bias + unresolved`` (weakest kind; ``unbounded`` if any is)."""
        return err.combine(
            (self.bias, self.reduction_bias, self.unresolved), "observable error total"
        )

    def to_dict(self) -> dict:
        """Strict-JSON summary."""
        return observable_json(self)


class TruthComparison(eqx.Module):
    """Synthetic-study comparison of a :class:`CombinationFit` with a known truth.

    ``truth`` ``(n_full,)``, ``x_true`` ``(n_x,)``, ``beta_true = B x_true``,
    ``beta_hat``, ``projected_truth = Pi x_true`` (and
    ``projected_truth_full = a_off + J Pi x_true``), ``representative``,
    ``unresolved_truth = x_true - Pi x_true``; with a discrepancy
    ``bias_measured = K (delta + R delta_C a)_kept`` and ``beta_bias_measured``
    ("measured, synthetic example only"); with a noise realisation
    ``noise_part = K n`` and ``beta_noise_part``; with both,
    ``decomposition_residual = x_hat - Pi x - bias_measured - noise_part``.
    Real data cannot supply these fields. Units: those of ``x``. Not
    certified: see the module docstring.
    """

    LABEL: ClassVar[tuple[str, ...]] = LABEL
    truth: jax.Array
    x_true: jax.Array
    beta_true: jax.Array
    beta_hat: jax.Array
    projected_truth: jax.Array
    projected_truth_full: jax.Array
    representative: jax.Array
    unresolved_truth: jax.Array
    bias_measured: jax.Array | None = None
    beta_bias_measured: jax.Array | None = None
    noise_part: jax.Array | None = None
    beta_noise_part: jax.Array | None = None
    decomposition_residual: jax.Array | None = None
    note: str = eqx.field(static=True, default="")

    def to_dict(self) -> dict:
        """Strict-JSON summary."""
        return truth_json(self)


class CombinationFit(eqx.Module):
    """Identifiable combinations of a linear SED fit (module docstring).

    Fields: ``labels``, ``coordinates`` (static), ``reduction``, ``jacobian``
    ``J`` ``(n_full, n_x)``, ``offset`` ``(n_full,)``, ``metric`` ``M_x`` and
    ``metric_factor`` ``R_x`` ``(n_x, n_x)``, ``factor_L``/``factor_Q``
    (``T J R_x^-1 = L Q``), ``singular_values`` ``(r_T,)`` with ``classes``,
    ``clusters``, ``numerical_rank`` and ``n_retained``, the thresholds,
    ``combination_rows`` ``B`` ``(n_R, n_x)``, ``beta_hat``, ``beta_sigma``,
    ``directions``, ``estimator`` ``K`` and ``beta_estimator`` over
    ``kept_rows``, ``x_hat``, ``representative``, ``covariance``,
    ``projector``, ``chi2``, ``dof``, ``stokes_hat``/``stokes_sigma``
    ``(n_ch, 4)`` on every channel, the error terms and ``provenance``.
    Shapes, units and what is not certified: see the module docstring.
    """

    LABEL: ClassVar[tuple[str, ...]] = LABEL
    labels: tuple[str, ...] = eqx.field(static=True)
    coordinates: str = eqx.field(static=True)
    reduction: ResponseReduction
    jacobian: jax.Array
    offset: jax.Array
    metric: jax.Array
    metric_factor: jax.Array
    metric_note: str = eqx.field(static=True)
    factor_L: jax.Array
    factor_Q: jax.Array
    lq_method: str = eqx.field(static=True)
    singular_values: jax.Array
    classes: tuple[str, ...] = eqx.field(static=True)
    numerical_rank: int = eqx.field(static=True)
    n_retained: int = eqx.field(static=True)
    rank_tol: float = eqx.field(static=True)
    max_sigma: float = eqx.field(static=True)
    cluster_rtol: float = eqx.field(static=True)
    clusters: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    combination_rows: jax.Array
    beta_hat: jax.Array
    beta_sigma: jax.Array
    directions: Subspaces
    estimator: jax.Array
    beta_estimator: jax.Array
    kept_rows: tuple[int, ...] = eqx.field(static=True)
    x_hat: jax.Array
    representative: jax.Array
    covariance: jax.Array
    projector: jax.Array
    chi2: jax.Array
    dof: int = eqx.field(static=True)
    stokes_hat: jax.Array
    stokes_sigma: jax.Array
    bias: ErrorTerm
    beta_bias: ErrorTerm
    reduction_bias: ErrorTerm
    unresolved: ErrorTerm
    data_discrepancy: ErrorTerm
    reduction_envelope: ErrorTerm
    coefficient_bound: ErrorTerm | None
    response: jax.Array | None
    noise_model: str = eqx.field(static=True)
    provenance: Provenance = eqx.field(static=True)

    # -- simple views ----------------------------------------------------------
    def standard_errors(self) -> np.ndarray:
        """``sqrt(diag(Cov_x))``: retained-subspace noise only, not a total error."""
        return np.sqrt(np.maximum(np.diag(np.asarray(self.covariance)), 0.0))

    def q_hat(self) -> np.ndarray:
        """Grouped coefficients of the representative, ``T a_hat``."""
        return self.reduction.grouped(np.asarray(self.representative))

    def combination_labels(self) -> tuple[str, ...]:
        """``"beta_i = w a + ..."`` with the three largest-weight terms of each row."""
        out = []
        for i, row in enumerate(np.asarray(self.combination_rows)):
            top = np.argsort(-np.abs(row), kind="stable")[:3]
            terms = " + ".join(f"{row[j]:.3g} {self.labels[j]}" for j in top)
            out.append(f"beta_{i + 1} = {terms} + ...".replace("+ -", "- "))
        return tuple(out)

    def slot_status(self, rtol=err.PI_RTOL) -> tuple[str, ...]:
        """Per ``x`` slot: ``"resolved"`` (``Pi`` row ``e_i``), ``"unconstrained"`` (zero row) or ``"mixed"``.

        ``rtol`` defaults to the threshold of the ``unresolved`` check
        (``PI_RTOL``). "resolved" is a numerical statement about ``Pi``; it
        does not remove the slot's ``unresolved`` entry unless the whole
        ``unresolved`` term is ``not_applicable``.
        """
        Pi = np.asarray(self.projector)
        out = []
        for i, row in enumerate(Pi):
            unit = np.zeros_like(row)
            unit[i] = 1.0
            if np.max(np.abs(row - unit), initial=0.0) <= rtol:
                out.append("resolved")
            elif np.max(np.abs(row), initial=0.0) <= rtol:
                out.append("unconstrained")
            else:
                out.append("mixed")
        return tuple(out)

    # -- builders -----------------------------------------------------------------
    def observable(self, J_obs, *, on="x") -> ObservableSummary:
        """Summary of ``J_obs x`` (``on="x"``) or ``J_obs a`` (``on="full"``), see :class:`ObservableSummary`."""
        J_obs = np.atleast_2d(np.asarray(J_obs, dtype=float))
        J, a_off = np.asarray(self.jacobian), np.asarray(self.offset)
        if on == "full":
            if J_obs.shape[1] != J.shape[0]:
                raise ValueError(f"J_obs must be (n_obs, {J.shape[0]}) for on='full'")
            Jx, value0 = J_obs @ J, J_obs @ a_off
        elif on == "x":
            if J_obs.shape[1] != J.shape[1]:
                raise ValueError(f"J_obs must be (n_obs, {J.shape[1]}) for on='x'")
            Jx, value0 = J_obs, np.zeros(J_obs.shape[0])
        else:
            raise ValueError("on must be 'x' or 'full'")
        if not np.all(np.isfinite(J_obs)):
            raise ValueError("J_obs must be finite")
        K, Cov = np.asarray(self.estimator), np.asarray(self.covariance)
        gain = Jx @ K
        cov = Jx @ Cov @ Jx.T
        comp = np.eye(J.shape[1]) - np.asarray(self.projector)
        d = self.directions
        return ObservableSummary(
            value=jnp.asarray(Jx @ np.asarray(self.x_hat) + value0),
            noise_sigma=jnp.asarray(np.sqrt(np.maximum(np.diag(cov), 0.0))),
            noise_covariance=jnp.asarray(cov),
            gain=jnp.asarray(gain),
            bias=err.propagated(
                gain, self.data_discrepancy, "observable bias", "delta"
            ),
            reduction_bias=err.propagated(
                gain,
                self.reduction_envelope,
                "observable reduction bias",
                "R delta_C a",
            ),
            unresolved=err.unresolved_term(
                Jx, comp, J, self.coefficient_bound, "observable unresolved part"
            ),
            null_sensitivity={
                "weak": jnp.asarray(Jx @ np.asarray(d.weak)),
                "numerical_null": jnp.asarray(Jx @ np.asarray(d.numerical_null)),
                "analytic_null": jnp.asarray(Jx @ np.asarray(d.analytic_null)),
            },
            note=(
                f"J_obs on {on}; noise is the retained subspace only; unresolved "
                "not_applicable only by a numerical check of J (I - Pi)"
            ),
        )

    def against_truth(self, a_true, *, discrepancy=None, noise=None) -> TruthComparison:
        """Compare with a known full vector (synthetic studies only; :class:`TruthComparison`)."""
        J, a_off = np.asarray(self.jacobian), np.asarray(self.offset)
        a_true = np.asarray(a_true, dtype=float)
        if a_true.shape != (J.shape[0],):
            raise ValueError(f"a_true must be ({J.shape[0]},)")
        x_true = np.linalg.lstsq(J, a_true - a_off, rcond=None)[0]
        gap = np.linalg.norm(J @ x_true + a_off - a_true)
        if gap > 1e-12 * max(np.linalg.norm(a_true), 1e-300):
            raise ValueError(
                "a_true is outside the range of the parameter map (a = a_off + J x): "
                f"residual {gap:.3g}"
            )
        Pi, K = np.asarray(self.projector), np.asarray(self.estimator)
        K_beta = np.asarray(self.beta_estimator)
        fields = {}
        if discrepancy is not None:
            delta = err.data_space(self._data_like(), discrepancy, "discrepancy")
            dCa = np.asarray(self.reduction.delta_C) @ a_true
            delta = delta + err.data_space(self._data_like(), dCa.reshape(-1, 4), "dCa")
            fields["bias_measured"] = K @ delta
            fields["beta_bias_measured"] = K_beta @ delta
        if noise is not None:
            n = err.data_space(self._data_like(), noise, "noise")
            fields["noise_part"] = K @ n
            fields["beta_noise_part"] = K_beta @ n
        x_hat = np.asarray(self.x_hat)
        if discrepancy is not None and noise is not None:
            fields["decomposition_residual"] = (
                x_hat - Pi @ x_true - fields["bias_measured"] - fields["noise_part"]
            )
        return TruthComparison(
            truth=jnp.asarray(a_true),
            x_true=jnp.asarray(x_true),
            beta_true=jnp.asarray(np.asarray(self.combination_rows) @ x_true),
            beta_hat=self.beta_hat,
            projected_truth=jnp.asarray(Pi @ x_true),
            projected_truth_full=jnp.asarray(a_off + J @ (Pi @ x_true)),
            representative=self.representative,
            unresolved_truth=jnp.asarray(x_true - Pi @ x_true),
            **{k: jnp.asarray(v) for k, v in fields.items()},
            note="measured, synthetic example only; bias includes R delta_C a_true",
        )

    def _data_like(self):
        return _KeptRows(self.kept_rows, self.response, self.stokes_hat.shape[0])

    def to_dict(self) -> dict:
        """Strict JSON: every field, ``slot_status``, ``claims`` and ``limits``."""
        return fit_json(self)


class _KeptRows:
    """Minimal data-space view (``n_ch``, ``n_data``, ``n_kept``, ``select``, ``response``)."""

    def __init__(self, kept, response, n_ch):
        self._kept, self.response, self.n_ch = tuple(kept), response, int(n_ch)

    def n_data(self):
        return 4 * self.n_ch if self.response is None else int(self.response.shape[0])

    def n_kept(self):
        return len(self._kept)

    def select(self, x):
        x = np.asarray(x)
        if x.shape[0] != self.n_data():
            raise ValueError(f"expected a leading axis of length {self.n_data()}")
        return x[list(self._kept)]


__all__ = [
    "CombinationFit",
    "Subspaces",
    "ObservableSummary",
    "TruthComparison",
    "LABEL",
]
