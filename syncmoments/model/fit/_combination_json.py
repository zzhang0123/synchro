"""Strict-JSON rendering of the ``fit_combinations`` results (private).

Separates the three claims of ``docs/DESIGN.md`` Section 10.1 (analytic
redundancy, numerical rank, practical recoverability) and lists the known
limitations. Nothing here is public API.
"""

from __future__ import annotations

import math

import numpy as np

from .._provenance import strict_json

LIMITS = (
    "analytic grouping covers ContinuumKernel only; other kernels get structural "
    "zeros only",
    "relation-set completeness is proven for uncapped blocks only",
    "check_rtol and the rank decisions are float64 statements; fractional scales "
    "outside [0.02, 0.4] are untested",
    "max_sigma, rank_tol and cluster_rtol are declared conventions; the retained set "
    "also depends on noise, masks, response, metric and reference scales",
    "inside a cluster only the subspace is determined (the basis is a declared "
    "convention); the beta of a cluster are correlated by at most (rho^2 - 1)/2, "
    "rho the cluster's s_max/s_min (value in the notes); K, Pi and Cov do not "
    "depend on the cluster basis",
    "the representative is not a moment vector; no positivity or realisability test",
    "valued unresolved and approximate-reduction terms need a declared coefficient "
    "bound, which does not cover the excluded tail",
    "affine parameter maps only (no fit_bfgs or fit_nodal route)",
    "model-error budgets from predict are not injected into data.discrepancy",
    "eager and dense: not jit safe, n_x <= 2000",
)


def _term(term):
    return None if term is None else term.to_dict()


def _float(value):
    value = float(value)
    return value if math.isfinite(value) else ("inf" if value > 0 else "-inf")


def _claims(fit):
    s = np.asarray(fit.singular_values)
    s_max = float(s[0]) if s.size else 0.0
    cutoff = fit.rank_tol * s_max
    rank = fit.numerical_rank
    red = fit.reduction
    analytic_null = int(np.asarray(fit.directions.analytic_null).shape[1])
    response = (
        "identity" if fit.response is None else f"supplied {tuple(fit.response.shape)}"
    )
    return {
        "analytic": {
            "null_dim": analytic_null,
            "scope": red.scope,
            "sources": sorted({r.source for r in red.relations}),
        },
        "numerical": {
            "rank": rank,
            "rank_tol": fit.rank_tol,
            "rank_margin": (
                float(s[rank - 1]) / cutoff if rank and cutoff > 0 else None
            ),
            "next_below_margin": (
                float(s[rank]) / cutoff if rank < s.size and cutoff > 0 else None
            ),
            "n_kept": len(fit.kept_rows),
            "response": response,
            "noise": fit.noise_model,
            "metric_note": fit.metric_note,
        },
        "practical": {
            "max_sigma": _float(fit.max_sigma),
            "n_retained": fit.n_retained,
            "n_weak": sum(c == "weak" for c in fit.classes),
            "n_numerical_null": sum(c == "numerical_null" for c in fit.classes),
        },
    }


def _slots(fit):
    status = fit.slot_status()
    se = fit.standard_errors()
    bounded = fit.unresolved.value is not None
    applicable = fit.unresolved.kind != "not_applicable"
    out = []
    for i, (label, state) in enumerate(zip(fit.labels, status)):
        entry = {
            "label": label,
            "status": state,
            "x_hat": float(np.asarray(fit.x_hat)[i]),
            "noise_sigma_retained_only": float(se[i]),
        }
        if applicable:
            entry["unresolved"] = (
                float(np.asarray(fit.unresolved.value)[i]) if bounded else "unbounded"
            )
        out.append(entry)
    return out


def fit_json(fit) -> dict:
    """Strict-JSON ``CombinationFit`` summary."""
    arrays = (
        "jacobian",
        "offset",
        "metric",
        "metric_factor",
        "factor_L",
        "factor_Q",
        "singular_values",
        "combination_rows",
        "beta_hat",
        "beta_sigma",
        "estimator",
        "beta_estimator",
        "x_hat",
        "representative",
        "covariance",
        "projector",
        "stokes_hat",
        "stokes_sigma",
    )
    out = {name: np.asarray(getattr(fit, name)) for name in arrays}
    d = fit.directions
    out.update(
        labels=list(fit.labels),
        coordinates=fit.coordinates,
        metric_note=fit.metric_note,
        lq_method=fit.lq_method,
        classes=list(fit.classes),
        numerical_rank=fit.numerical_rank,
        n_retained=fit.n_retained,
        rank_tol=fit.rank_tol,
        max_sigma=_float(fit.max_sigma),
        cluster_rtol=fit.cluster_rtol,
        clusters=[list(c) for c in fit.clusters],
        kept_rows=list(fit.kept_rows),
        chi2=float(fit.chi2),
        dof=fit.dof,
        directions={
            "retained": np.asarray(d.retained),
            "weak": np.asarray(d.weak),
            "numerical_null": np.asarray(d.numerical_null),
            "analytic_null": np.asarray(d.analytic_null),
        },
        combination_labels=list(fit.combination_labels()),
        bias=_term(fit.bias),
        beta_bias=_term(fit.beta_bias),
        reduction_bias=_term(fit.reduction_bias),
        unresolved=_term(fit.unresolved),
        data_discrepancy=_term(fit.data_discrepancy),
        coefficient_bound=_term(fit.coefficient_bound),
        slot_status=list(fit.slot_status()),
        slots=_slots(fit),
        claims=_claims(fit),
        limits=list(LIMITS),
        reduction=fit.reduction.to_dict(),
        provenance=fit.provenance.to_dict(),
    )
    return strict_json(out)


def observable_json(summary) -> dict:
    """Strict-JSON ``ObservableSummary``."""
    out = {
        name: np.asarray(getattr(summary, name))
        for name in ("value", "noise_sigma", "noise_covariance", "gain")
    }
    out.update(
        bias=summary.bias.to_dict(),
        reduction_bias=summary.reduction_bias.to_dict(),
        unresolved=summary.unresolved.to_dict(),
        total=summary.total().to_dict(),
        null_sensitivity={
            k: np.asarray(v) for k, v in summary.null_sensitivity.items()
        },
        note=summary.note,
    )
    return strict_json(out)


def truth_json(comparison) -> dict:
    """Strict-JSON ``TruthComparison``."""
    names = (
        "truth",
        "x_true",
        "beta_true",
        "beta_hat",
        "projected_truth",
        "projected_truth_full",
        "representative",
        "unresolved_truth",
        "bias_measured",
        "beta_bias_measured",
        "noise_part",
        "beta_noise_part",
        "decomposition_residual",
    )
    out = {
        n: (
            None
            if getattr(comparison, n) is None
            else np.asarray(getattr(comparison, n))
        )
        for n in names
    }
    out["note"] = comparison.note
    return strict_json(out)


__all__ = ["fit_json", "observable_json", "truth_json", "LIMITS"]
