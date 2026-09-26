"""Validation checks of the SED reconstruction example (design Section 9.3, step 7).

Each entry is ``{"value", "label", "note"}`` with ``label`` one of
``finite check``, ``measured`` or ``statistical``; none is a certificate.
Monte Carlo tolerances are statistical, not bounds.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import subspace_angles

from syncmoments.model.fit import reduce_response

import sed_reconstruction_direct as direct
from sed_reconstruction_setup import (
    KAPPA,
    MANUSCRIPT_REPRESENTATIVES,
    SEED,
    combinations_fit,
    item,
    make_basis,
)


def monte_carlo(fit, sigma_kept, draws):
    """Max relative error of the diagonal of ``Cov_x`` and of ``beta_sigma^2``."""
    noise = np.random.default_rng(SEED + 1).standard_normal((sigma_kept.size, draws))
    noise *= sigma_kept[:, None]
    diag = np.diag(np.asarray(fit.covariance))
    keep = diag > 1e-12 * diag.max()
    sims = np.asarray(fit.estimator) @ noise
    x_err = np.max(np.abs(np.mean(sims[keep] ** 2, axis=1) / diag[keep] - 1))
    beta = np.asarray(fit.beta_estimator) @ noise
    b_err = np.max(
        np.abs(np.mean(beta**2, axis=1) * np.asarray(fit.beta_sigma) ** -2 - 1)
    )
    return float(x_err), float(b_err), int(keep.sum())


def refinement(cfg, centres, data, fit, reduction, a_true, sed):
    """Basis refinement: ``T``, ``n_R``, subspaces, ``beta_hat`` and predictions."""
    basis_r = make_basis(centres, *cfg.refined)
    red_r = reduce_response(
        basis_r, relations="continuum", representatives=MANUSCRIPT_REPRESENTATIVES
    )
    fit_r = combinations_fit(basis_r, data, red_r)
    C, C_r = np.asarray(reduction.C), np.asarray(red_r.C)
    change = (C_r - C) @ a_true
    over_I = np.abs(change.reshape(-1, 4)[:, :3]) / (sed[:, :1] * direct.NORM)
    same_n = fit_r.n_retained == fit.n_retained
    angle = beta_shift = float("nan")
    if same_n and fit.n_retained:
        W, W_r = (np.asarray(f.directions.retained) for f in (fit, fit_r))
        angle = float(np.max(subspace_angles(W, W_r)))
        shift = np.asarray(fit_r.beta_hat) - np.asarray(fit.beta_hat)
        beta_shift = float(np.max(np.abs(shift) / np.asarray(fit.beta_sigma)))
    size = "n_eta={}, n_nu={}, n_nodes_F={}".format(*cfg.refined)
    return {
        "refinement_T_max_change": item(
            float(np.max(np.abs(np.asarray(red_r.T) - np.asarray(reduction.T)))),
            "finite check",
            f"refined basis {size}",
        ),
        "refinement_n_retained_unchanged": item(bool(same_n), "finite check", size),
        "refinement_max_principal_angle": item(
            angle, "measured", "retained subspaces, radians"
        ),
        "refinement_max_beta_change_over_sigma": item(beta_shift, "measured", size),
        "fit_refinement_max_coefficient_change": item(
            float(
                np.max(
                    np.abs(
                        np.asarray(fit_r.representative)
                        - np.asarray(fit.representative)
                    )
                )
            ),
            "measured",
            "max |a_hat - a_hat_refined|",
        ),
        "basis_refinement_change_over_I": item(
            float(np.max(over_I)), "measured", "max |(C_refined - C) a_true| / I_direct"
        ),
    }


def rescaling(cfg, centres, f_eval, sed, layout, a_true, fit):
    """The continuum rescaling ambiguity on every ninth centre (design Section 9.3)."""
    n_eta, n_nu = cfg.direct
    alt = direct.direct_sed(
        centres[::9], f_eval, cfg.nodes, n_eta, n_nu, field_scale=KAPPA
    )
    err = float(np.max(np.abs(alt - sed[::9]) / sed[::9, :1]))
    m_alt = direct.moment_vector(layout.entries, cfg.nodes, field_scale=KAPPA)
    a_alt = m_alt / KAPPA  # source column divided by kappa
    B = np.asarray(fit.combination_rows)
    beta_gap = np.abs(B @ (a_alt - a_true)) / np.asarray(fit.beta_sigma)
    info = {
        "field_scale": KAPPA,
        "source_column_ratio": 1 / KAPPA,
        "support": direct.rescaled_support(KAPPA),
        "moments": m_alt,
        "full_vector": a_alt,
        "max_full_difference": float(np.max(np.abs(a_alt - a_true))),
        "beta_true_alternative": B @ a_alt,
        "max_beta_difference_over_sigma": float(np.max(beta_gap, initial=0.0)),
        "note": "same direct continuum SED, different supports and moments; "
        "fixed-support external priors may remove this ambiguity, the SED does not",
    }
    return item(err, "finite check", f"kappa = {KAPPA} on centres[::9]"), info


def grouping_checks(st):
    """Grouped against ungrouped under one metric, and the exactness of ``C = H T``."""
    fit, fit_u, reduction = st["fit"], st["fit_ungrouped"], st["reduction"]
    C = np.asarray(reduction.C)
    retained = np.flatnonzero(np.array(fit.classes) == "retained")
    s, s_u = np.asarray(fit.singular_values), np.asarray(fit_u.singular_values)
    analytic = [r for r in reduction.relations if r.source == "analytic"]
    null_err = max(
        (np.max(np.abs(sum(w * C[:, j] for j, w in r.weights))) for r in analytic),
        default=0.0,
    )
    absolute = np.max(np.abs(np.asarray(reduction.delta_C))) / direct.NORM
    diff = np.asarray(fit.representative) - np.asarray(fit_u.representative)
    return {
        "grouped_vs_ungrouped_max_coefficient_difference": item(
            float(np.max(np.abs(diff))),
            "finite check",
            "same metric; relations='none' vs 'continuum'",
        ),
        "grouped_vs_ungrouped_max_projector_difference": item(
            float(
                np.max(np.abs(np.asarray(fit.projector) - np.asarray(fit_u.projector)))
            ),
            "finite check",
        ),
        "grouped_vs_ungrouped_max_retained_singular_value_difference": item(
            float(np.max(np.abs(s[retained] / s_u[retained] - 1), initial=0.0)),
            "finite check",
            "relative",
        ),
        "lossless_grouping_max_response_error": item(
            reduction.max_delta_C,
            "finite check",
            f"max|C - H T| / max|C|; absolute {absolute:.3g} in units of NORM",
        ),
        "analytic_null_column_identity_max_error": item(
            float(null_err / np.max(np.abs(C))),
            "finite check",
            f"max |C w| / max|C| over {len(analytic)} generated relations",
        ),
    }


def basic_checks(st):
    """Moment routes, forward check, ``F``, quadrature, decomposition, Monte Carlo."""
    cfg, sed = st["cfg"], st["sed"]
    mc_x, mc_beta, mc_slots = monte_carlo(st["fit"], st["sigma_kept"], cfg.mc_draws)
    residual = np.asarray(st["comparison"].decomposition_residual)
    coarse = "nodes={}, n_eta={}, n_nu={}".format(cfg.coarse_nodes, *cfg.coarse_direct)
    return {
        "moment_routes_max_difference": item(
            float(np.max(np.abs(st["a_true"] - st["a_closed"]))),
            "finite check",
            "PopulationSamples + JointMoments.from_samples vs closed-form angular averages",
        ),
        "forward_check_max_relative_difference": item(
            st["forward_gap"], "finite check", "predict(...).stokes vs C a_true"
        ),
        "independent_F_relative_error": item(
            st["f_error"], "finite check", "off-grid quad oracle"
        ),
        "direct_quadrature_change_over_I": item(
            float(np.max(np.abs(sed - st["coarse"]) / sed[:, :1])), "measured", coarse
        ),
        "decomposition_residual": item(
            float(np.max(np.abs(residual))),
            "finite check",
            "max |a_hat - Pi a - K delta - K n|",
        ),
        "covariance_monte_carlo_max_relative_diagonal_error": item(
            mc_x,
            "statistical",
            f"{cfg.mc_draws} draws, {mc_slots} slots with diag > 1e-12 max",
        ),
        "beta_monte_carlo_max_relative_variance_error": item(
            mc_beta, "statistical", f"{cfg.mc_draws} draws"
        ),
    }
