"""Finite joint Legendre/Taylor response of a smooth-channel harmonic spectrum.

NumPy/SciPy implementation, independent of the companion package. The finite
quadrature and finite-difference comparisons are numerical evidence, not global
remainder certificates. Frequencies y=nu/nu_ref and depth zeta=2(c/nu_ref)^2 phi
are dimensionless. Output is channel-integrated power per source electron in
units e^2 Omega_0^2/(2 pi c), Omega_0=e B_0/(m_e c); R_j(nu)=bump(y).

Run from the manuscript root:
  python -m validation.full_response --output validation/full_response_results.json
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
from numpy.polynomial.legendre import leggauss, legvander
import scipy
from scipy.special import jv, jvp

GAMMA0 = 20.0
ZETA0 = 4.0
CENTRES = np.array([2.0, 4.0, 8.0])
HALF_WIDTHS = 0.65 * CENTRES
MAX_DEGREE = 2
LMAX = 8
POWERS = tuple(p for p in itertools.product(range(3), repeat=3) if sum(p) <= 2)


def response(y: np.ndarray) -> np.ndarray:
    """C-infinity compact bump, height one, NOT unit-area normalized."""
    t = (y[None, ...] - CENTRES.reshape((-1,) + (1,) * y.ndim)) / HALF_WIDTHS.reshape(
        (-1,) + (1,) * y.ndim
    )
    inside = np.abs(t) < 1
    out = np.zeros_like(t)
    out[inside] = np.exp(1 - 1 / (1 - t[inside] ** 2))
    return out


def harmonic_cutoff(gamma_max: float, field_min: float) -> int:
    """All m above this integer have zero channel weight on this support.

    y_m=m * (B/B0) * gamma0/gamma / D, 0<D<2. Therefore
    m < y_max * gamma_max * 2 / (gamma0 * field_min).
    This is a support cutoff, not an estimated asymptotic harmonic tail.
    """
    ymax = float(np.max(CENTRES + HALF_WIDTHS))
    return math.ceil(ymax * gamma_max * 2 / (GAMMA0 * field_min))


def channel_kernel(
    q: np.ndarray, mu: np.ndarray, eta: np.ndarray, nmax: int
) -> np.ndarray:
    """Return [I,V,P] x channel x depth-derivative(0,1,2) x angular grid.

    q=(gamma/gamma0-1, B/B0-1, zeta-zeta0). The sky phase is
    excluded. Each line carries its own Faraday phase and its own response.
    Unused I/V depth derivative slots are exactly zero.
    """
    gamma = GAMMA0 * (1 + q[0])
    field = 1 + q[1]
    if gamma <= 1 or field <= 0:
        raise ValueError("Require gamma>1 and B/B0>0")
    beta = np.sqrt(1 - gamma**-2)
    mu, eta = np.broadcast_arrays(mu, eta)
    d = 1 - beta * mu * eta
    sin_a = np.sqrt(1 - mu * mu)
    sin_t = np.sqrt(1 - eta * eta)
    out = np.zeros((3, len(CENTRES), 3, *mu.shape), dtype=complex)
    # Interior Gauss nodes avoid the removable axial formula singularity.
    if np.any(sin_t == 0):
        raise ValueError("Use interior angular nodes")
    for m in range(1, nmax + 1):
        y = m * field * GAMMA0 / (gamma * d)
        r = response(y)
        active = np.any(r != 0, axis=0)
        if not np.any(active):
            continue
        x = m * beta * sin_a[active] * sin_t[active] / d[active]
        parallel = (eta[active] - beta * mu[active]) / sin_t[active] * jv(m, x)
        perpendicular = beta * sin_a[active] * (jv(m - 1, x) - jv(m + 1, x)) / 2
        pref = field**2 * m**2 / (gamma**2 * d[active] ** 3)
        intensity = pref * (parallel**2 + perpendicular**2)
        qnat = pref * (parallel**2 - perpendicular**2)
        circular = 2 * pref * parallel * perpendicular
        out[0, :, 0, active] += (r[:, active] * intensity).T
        out[1, :, 0, active] += (r[:, active] * circular).T
        phase = np.exp(1j * (ZETA0 + q[2]) / y[active] ** 2)
        for b in range(3):
            term = r[:, active] * qnat * phase * (1j / y[active] ** 2) ** b
            out[2, :, b, active] += term.T
    return out


def direct_kernel(
    q: np.ndarray, mu: np.ndarray, eta: np.ndarray, nmax: int
) -> np.ndarray:
    """Separate vectorized line sum using jvp, for direct population reference.

    Same physical assumptions; this is an independent numerical route, not an
    external test of the helical-orbit model or Bessel implementation.
    """
    gamma, field = GAMMA0 * (1 + q[0]), 1 + q[1]
    mu, eta = np.broadcast_arrays(mu, eta)
    shape = mu.shape
    m = np.arange(1, nmax + 1)[:, None]
    mu, eta = mu.ravel()[None, :], eta.ravel()[None, :]
    beta = np.sqrt(1 - gamma**-2)
    d = 1 - beta * mu * eta
    y = m * field * GAMMA0 / (gamma * d)
    x = m * beta * np.sqrt(1 - mu**2) * np.sqrt(1 - eta**2) / d
    # Avoid evaluating Bessels for terms with identically zero response.
    active = np.any(response(y) != 0, axis=0)
    a = np.zeros_like(x)
    b = np.zeros_like(x)
    aa = np.broadcast_to((eta - beta * mu) / np.sqrt(1 - eta**2), x.shape)
    bb = np.broadcast_to(beta * np.sqrt(1 - mu**2), x.shape)
    mm = np.broadcast_to(m, x.shape)
    a[active] = aa[active] * jv(mm[active], x[active])
    b[active] = bb[active] * jvp(mm[active], x[active])
    pref = field**2 * m**2 / (gamma**2 * d**3)
    r = response(y)
    i = np.sum(r * pref * (a * a + b * b), axis=1)
    v = np.sum(r * pref * (2 * a * b), axis=1)
    p = np.sum(r * pref * (a * a - b * b) * np.exp(1j * (ZETA0 + q[2]) / y**2), axis=1)
    return np.stack((i, v, p)).reshape((3, len(CENTRES), *shape))


def angular_grid(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nodes, weights = leggauss(n)
    return nodes, weights, legvander(nodes, LMAX)


def project(
    kernel: np.ndarray, weights: np.ndarray, polynomials: np.ndarray
) -> np.ndarray:
    weighted = weights[:, None] * polynomials
    coeff = np.einsum("...ij,il,jk->...lk", kernel, weighted, weighted, optimize=True)
    degrees = 2 * np.arange(LMAX + 1) + 1
    return coeff * degrees[:, None] * degrees[None, :] / 4


def derivative_coefficients(n: int, step: float, nmax: int) -> np.ndarray:
    """Five-point centred gamma/B differences AFTER channel integration.

    Depth derivatives are analytic; the b! and r!s! factors are all included.
    Derivatives are with respect to dimensionless q; the moments use the same q.
    """
    nodes, weights, poly = angular_grid(n)
    values = {}
    for i, j in itertools.product(range(-2, 3), repeat=2):
        kernel = channel_kernel(
            np.array([i * step, j * step, 0.0]), nodes[:, None], nodes[None, :], nmax
        )
        values[i, j] = project(kernel, weights, poly)
    stencils = {
        0: np.array([0, 0, 1, 0, 0.0]),
        1: np.array([1, -8, 0, 8, -1.0]) / (12 * step),
        2: np.array([-1, 16, -30, 16, -1.0]) / (12 * step**2),
    }
    coeff = np.zeros((len(POWERS), 3, len(CENTRES), LMAX + 1, LMAX + 1), complex)
    for t, (r, s, b) in enumerate(POWERS):
        for i, j in itertools.product(range(-2, 3), repeat=2):
            weight = stencils[r][i + 2] * stencils[s][j + 2]
            if weight:
                coeff[t] += weight * values[i, j][:, :, b]
        coeff[t] /= math.factorial(r) * math.factorial(s) * math.factorial(b)
    return coeff


def population(angular_nodes: int, latent_nodes: int, width: float, nmax: int) -> dict:
    """u,v independent uniform[-1,1]; smooth conditional angular density.

    q_gamma=.2*w*u, q_B=.2*w*(.6*u+.4*v), q_zeta=w*(2*u+v).
    rho(mu,eta|u,v) proportional to exp((2+u)mu+(1+.5v)eta+.75mu eta).
    phi=.2+.4mu+.2v (mod 2pi). Thus all moments share one positive measure.
    """
    nodes, weights, poly = angular_grid(angular_nodes)
    mu, eta = nodes[:, None], nodes[None, :]
    base = weights[:, None] * weights[None, :]
    latent, lw = leggauss(latent_nodes)
    moments = np.zeros((len(POWERS), 2, LMAX + 1, LMAX + 1), complex)
    exact = np.zeros((3, len(CENTRES)), complex)
    angular_only = {limit: np.zeros_like(exact) for limit in (2, 4, 8)}
    for iu, u in enumerate(latent):
        for iv, v in enumerate(latent):
            q = width * np.array([0.2 * u, 0.2 * (0.6 * u + 0.4 * v), 2 * u + v])
            rho = np.exp((2 + u) * mu + (1 + 0.5 * v) * eta + 0.75 * mu * eta)
            measure = base * rho
            measure /= measure.sum()
            sky = np.exp(2j * (0.2 + 0.4 * mu + 0.2 * v))
            ang0 = np.einsum("ij,il,jk->lk", measure, poly, poly, optimize=True)
            ang2 = np.einsum("ij,il,jk->lk", measure * sky, poly, poly, optimize=True)
            weight = lw[iu] * lw[iv] / 4
            for t, powers in enumerate(POWERS):
                monomial = np.prod(q ** np.array(powers))
                moments[t, 0] += weight * monomial * ang0
                moments[t, 1] += weight * monomial * ang2
            raw = direct_kernel(q, mu, eta, nmax)
            exact[:2] += weight * np.einsum("acij,ij->ac", raw[:2], measure)
            exact[2] += weight * np.einsum("cij,ij->c", raw[2], measure * sky)
            # Exact q dependence, only angular projection truncated: separates
            # population-averaged angular truncation from Taylor truncation.
            projected = project(raw, weights, poly)
            for limit in angular_only:
                angular_only[limit][:2] += weight * np.einsum(
                    "aclk,lk->ac",
                    projected[:2, :, : limit + 1, : limit + 1],
                    ang0[: limit + 1, : limit + 1],
                )
                angular_only[limit][2] += weight * np.einsum(
                    "clk,lk->c",
                    projected[2, :, : limit + 1, : limit + 1],
                    ang2[: limit + 1, : limit + 1],
                )
    return {"moments": moments, "direct": exact, "angular_only": angular_only}


def contract(
    coeff: np.ndarray,
    moments: np.ndarray,
    degree: int,
    angular: int,
    drop_mixed: bool = False,
) -> np.ndarray:
    out = np.zeros((3, len(CENTRES)), complex)
    for t, powers in enumerate(POWERS):
        if sum(powers) > degree or (drop_mixed and sum(x > 0 for x in powers) > 1):
            continue
        c = coeff[t, :, :, : angular + 1, : angular + 1]
        m = moments[t, :, : angular + 1, : angular + 1]
        out[:2] += np.einsum("aclk,lk->ac", c[:2], m[0])
        out[2] += np.einsum("clk,lk->c", c[2], m[1])
    return out


def stokes(value: np.ndarray) -> np.ndarray:
    return np.stack((value[0].real, value[2].real, value[2].imag, value[1].real))


def normalized_error(a: np.ndarray, b: np.ndarray, reference: np.ndarray) -> float:
    return float(np.max(np.abs(stokes(a) - stokes(b)) / reference[0].real))


def checks(nmax: int) -> dict:
    rng = np.random.default_rng(20260923)
    mu, eta = rng.uniform(-0.95, 0.95, (2, 19))
    worst = 0.0
    for _ in range(4):
        q = rng.uniform(-1, 1, 3) * [0.2, 0.2, 2]
        a = channel_kernel(q, mu, eta, nmax)[:, :, 0]
        b = direct_kernel(q, mu, eta, nmax)
        worst = max(worst, float(np.max(np.abs(a - b)) / np.max(np.abs(b))))
        np.testing.assert_allclose(a, b, rtol=2e-12, atol=2e-16)
        doubled = channel_kernel(q, mu, eta, nmax * 2)[:, :, 0]
        np.testing.assert_array_equal(a, doubled)
    q = np.array([0.1, -0.05, 0.2])
    a = channel_kernel(q, mu, eta, nmax)[:, :, 0]
    rev = channel_kernel(q, -mu, -eta, nmax)[:, :, 0]
    np.testing.assert_allclose(a[[0, 2]], rev[[0, 2]], atol=2e-15)
    np.testing.assert_allclose(a[1], -rev[1], atol=2e-15)
    return {
        "kernel_routes_max_scaled_difference": worst,
        "cutoff_doubling": "identical",
        "angular_parity": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Small smoke configuration, not manuscript results",
    )
    parser.add_argument(
        "--saved-stages",
        type=Path,
        help="Explicitly postprocess saved quadrature/derivative arrays; hashes are recorded",
    )
    args = parser.parse_args()
    from validation import full_response_product as product

    start = time.perf_counter()
    grids = [(24, 4), (32, 6)] if args.quick else [(160, 12), (256, 16)]
    basis_grids = (16, 24) if args.quick else (96, 128)
    steps = (0.001, 0.0005)
    stage_hashes = {}
    nmax = harmonic_cutoff(GAMMA0 * 1.2, 0.8)
    test_results = checks(nmax)
    populations = {}
    for ng, nl in grids:
        for width in (0.5, 1.0):
            print(
                f"Population quadrature angular={ng}, latent={nl}, width={width}",
                flush=True,
            )
            if args.saved_stages:
                path = args.saved_stages / f"full_response_pop_{ng}_{nl}_{width}.npz"
                with np.load(path, allow_pickle=False) as data:
                    populations[ng, nl, width] = {
                        "moments": data["moments"],
                        "direct": data["direct"],
                        "angular_only": {
                            limit: data[f"angular_{limit}"] for limit in (2, 4, 8)
                        },
                    }
                stage_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            else:
                populations[ng, nl, width] = population(ng, nl, width, nmax)
    finest = grids[-1]
    coefficients = {}
    for ng in basis_grids:
        for step in steps:
            print(f"Projected derivatives angular={ng}, h={step}", flush=True)
            if args.saved_stages:
                path = (
                    args.saved_stages / f"full_response_product_coeff_{ng}_{step}.npy"
                )
                coefficients[ng, step] = np.load(path, allow_pickle=False)
                stage_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            else:
                coefficients[ng, step] = product.coefficients(ng, step, nmax)
    rows, stability = [], []
    coeff = coefficients[basis_grids[-1], steps[-1]]
    for width in (0.5, 1.0):
        pop = populations[*finest, width]
        reference = pop["direct"]
        for degree, angular in itertools.product((0, 1, 2), (2, 4, 8)):
            prediction = contract(coeff, pop["moments"], degree, angular)
            angular_average = pop["angular_only"][angular]
            rows.append(
                {
                    "width_scale": width,
                    "N": degree,
                    "L_mu": angular,
                    "L_eta": angular,
                    "direct_IQUV": stokes(reference).tolist(),
                    "finite_IQUV": stokes(prediction).tolist(),
                    "max_abs_stokes_error_over_channel_I": normalized_error(
                        prediction, reference, reference
                    ),
                    "angular_only_error_over_channel_I": normalized_error(
                        angular_average, reference, reference
                    ),
                    "taylor_vs_angular_only_over_channel_I": normalized_error(
                        prediction, angular_average, reference
                    ),
                }
            )
        pred = contract(coeff, pop["moments"], 2, 8)
        hpred = contract(coefficients[basis_grids[-1], steps[0]], pop["moments"], 2, 8)
        apred = contract(coefficients[basis_grids[0], steps[-1]], pop["moments"], 2, 8)
        previous = populations[*grids[-2], width]
        prevpred = contract(coeff, previous["moments"], 2, 8)
        mixed_drop = contract(coeff, pop["moments"], 2, 8, drop_mixed=True)
        stability.append(
            {
                "width_scale": width,
                "reference_refinement": [
                    normalized_error(
                        populations[*g, width]["direct"], reference, reference
                    )
                    for g in grids[:-1]
                ],
                "moment_quadrature_change": normalized_error(prevpred, pred, reference),
                "derivative_step_change": normalized_error(hpred, pred, reference),
                "derivative_angular_grid_change": normalized_error(
                    apred, pred, reference
                ),
                "mixed_term_deletion_change": normalized_error(
                    mixed_drop, pred, reference
                ),
            }
        )
        # At least one meaningful mixed term must contribute; a zeroed tensor
        # or accidentally factorized population must not silently pass.
        assert stability[-1]["mixed_term_deletion_change"] > 1e-5
        np.testing.assert_allclose(
            pop["moments"][POWERS.index((0, 0, 0)), 0, 0, 0], 1, atol=2e-14
        )
        for powers, expected in [
            ((1, 1, 0), 0.008 * width**2),
            ((1, 0, 1), (0.4 / 3) * width**2),
            ((0, 1, 1), (0.32 / 3) * width**2),
        ]:
            np.testing.assert_allclose(
                pop["moments"][POWERS.index(powers), 0, 0, 0], expected, atol=2e-14
            )
        if not args.quick:
            for key in [
                "derivative_step_change",
                "derivative_angular_grid_change",
                "moment_quadrature_change",
            ]:
                assert stability[-1][key] < 2e-6, (key, stability[-1][key])
            assert stability[-1]["reference_refinement"][-1] < 2e-6

    result = {
        "scope": "Finite numerical validation, not a global Taylor/angular remainder certificate or Galactic accuracy claim.",
        "configuration": {
            "gamma0": GAMMA0,
            "zeta0": ZETA0,
            "channel_centres": CENTRES.tolist(),
            "channel_half_widths": HALF_WIDTHS.tolist(),
            "nmax": nmax,
            "grids": grids,
            "basis_product_grids": basis_grids,
            "derivative_steps": steps,
            "quick": args.quick,
        },
        "checks": test_results,
        "rows": rows,
        "stability": stability,
        "elapsed_seconds": time.perf_counter() - start,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "quadrature_script_sha256": hashlib.sha256(
            Path(product.__file__).read_bytes()
        ).hexdigest(),
        "saved_stage_sha256": stage_hashes,
        "execution_mode": "postprocess_saved_stages"
        if args.saved_stages
        else "fresh_computation",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {"stability": stability, "elapsed_seconds": result["elapsed_seconds"]},
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
