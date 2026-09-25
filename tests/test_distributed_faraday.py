"""Independent emission-depth, transfer and finite-statistic checks."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.testing import assert_allclose
import pytest

from syncmoments.faraday import (
    emission_polarisation,
    joint_faraday_average,
    joint_faraday_moments,
)
from syncmoments.rm import (
    faraday_depth_practical,
    rotation_measure_practical,
    screen_polarisation,
)
from syncmoments.transfer import mueller_matrix, transfer_slab


def test_swapping_emitters_retains_marginals_but_changes_polarisation():
    lam = np.sqrt(np.pi / 4)
    # Far emitter sees the rotator; near emitter does not. The two source
    # populations and total ray depth are unchanged by exchanging A and B.
    first = emission_polarisation([1, 1j], [1, 0], lam, source_column=2)
    second = emission_polarisation([1j, 1], [1, 0], lam, source_column=2)
    assert_allclose(first, 2j, atol=1e-14)
    assert_allclose(second, 0, atol=1e-14)


@pytest.mark.parametrize("total_depth", [-7.0, 0.0, 7.0])
def test_uniform_mixed_slab_matches_sinc_and_matrix_transfer(total_depth):
    nodes, weights = leggauss(64)
    s = (nodes + 1) / 2
    lam = np.array([0.0, 0.3, np.sqrt(np.pi / 7), 1.0])
    intrinsic = 0.8 - 0.3j
    actual = jax.jit(emission_polarisation)(
        intrinsic, total_depth * (1 - s), lam, weights, source_column=3.0
    )
    x = total_depth * lam**2
    analytic = 3 * intrinsic * np.exp(1j * x) * np.sinc(x / np.pi)
    assert_allclose(actual, analytic, atol=6e-14)
    for k, wavelength in enumerate(lam):
        matrix = mueller_matrix(0, 0, 0, 0, 0, 0, 2 * total_depth * wavelength**2)
        stokes = transfer_slab(
            jnp.zeros(4), jnp.array([3.0, 2.4, -0.9, 0.0]), matrix, 1.0
        )
        assert_allclose(actual[k], stokes[1] + 1j * stokes[2], atol=6e-14)


def test_frequency_dependent_correlations_normalisation_and_screen_limit():
    lam = np.array([[0.2, 0.4], [0.6, 0.8]])
    depths = np.array([-0.7, 0.2, 1.1])
    emission = np.array([1, 2j, -0.3 + 0.7j])[:, None, None] * (
        1 + np.arange(3)[:, None, None] * lam
    )
    weights = np.array([1.0, 2.0, 4.0])
    oracle = 5 * np.sum(
        weights[:, None, None]
        / 7
        * emission
        * np.exp(2j * depths[:, None, None] * lam**2),
        axis=0,
    )
    compiled = jax.jit(emission_polarisation)
    for scale in [1.0, 4e307]:
        assert_allclose(
            compiled(emission, depths, lam, weights * scale, source_column=5.0),
            oracle,
            atol=2e-14,
        )
    assert_allclose(compiled(emission, depths, lam, source_column=0.0), 0.0)
    assert_allclose(
        compiled(emission, np.zeros(3), lam, weights, source_column=5.0),
        5 * np.average(emission, axis=0, weights=weights),
        atol=1e-14,
    )
    amplitudes = np.array([1, 2j, -0.3 + 0.7j])
    assert_allclose(
        compiled(amplitudes, depths, lam, weights),
        screen_polarisation(amplitudes, depths, lam, weights),
        atol=1e-14,
    )
    factorised = (
        5
        * np.average(emission, axis=0, weights=weights)
        * np.average(
            np.exp(2j * depths[:, None, None] * lam**2), axis=0, weights=weights
        )
    )
    assert np.max(np.abs(oracle - factorised)) > 0.5


def test_emission_depth_and_column_gradients():
    depths = jnp.array([-0.3, 0.0, 0.8])
    emission = jnp.array([1 + 0.2j, 0.7j, -0.4])
    weights = jnp.array([1.0, 2.0, 3.0])
    lam, column = 0.7, 4.0
    rotated = np.asarray(emission) * np.exp(2j * np.asarray(depths) * lam**2)

    def fn(d, w, n):
        return emission_polarisation(emission, d, lam, w, source_column=n).real

    actual = jax.jit(jax.grad(fn, argnums=(0, 1, 2)))(depths, weights, column)
    mean = np.average(rotated, weights=weights)
    expected = (
        (column * np.asarray(weights) / 6 * 2j * lam**2 * rotated).real,
        (column * (rotated - mean) / 6).real,
        mean.real,
    )
    for got, want in zip(actual, expected):
        assert_allclose(got, want, atol=3e-14)


def test_ray_columns_and_signed_channel_response_preserve_weighting():
    lam = np.array([0.2, 0.4, 0.7])
    columns = np.array([2.0, 7.0])
    ray_weights = np.array([0.3, 0.7])
    local_weights = np.array([[0.2, 0.8], [0.6, 0.4]])
    depths = np.array([[1.1, 0.1], [-0.3, -0.7]])
    emission = np.array([[1 + 0.2j, 0.7], [-0.8j, 0.3]])
    rays = [
        emission_polarisation(
            emission[k], depths[k], lam, local_weights[k], source_column=columns[k]
        )
        for k in range(2)
    ]
    masses = ray_weights[:, None] * columns[:, None] * local_weights
    flattened = emission_polarisation(
        emission.ravel(),
        depths.ravel(),
        lam,
        masses.ravel(),
        source_column=np.dot(ray_weights, columns),
    )
    assert_allclose(flattened, np.dot(ray_weights, rays), atol=2e-14)
    # Finite-phase envelopes also propagate through a signed observing map.
    basis = emission.ravel()[:, None]
    moments, next_absolute = joint_faraday_moments(
        basis, depths.ravel(), 4, masses.ravel()
    )
    prediction, envelope = joint_faraday_average(
        np.ones((3, 1)),
        moments,
        lam,
        absolute_next=next_absolute,
        source_error=0.0,
        source_column=np.dot(ray_weights, columns),
    )
    response = np.array([[1.0, -0.4, 0.3], [0.2, 0.5, 0.1]])
    assert np.all(
        np.abs(response @ (prediction - flattened))
        <= np.abs(response) @ envelope + 1e-14
    )


def test_frequency_dependent_emission_gradients_and_input_error_envelope():
    lam = jnp.array([0.0, 0.4, 0.8])
    depths = jnp.array([-0.5, 0.8])
    spectrum = jnp.array([[1.0, 0.5, 0.2], [-0.2, 0.7, 0.4]])
    weights = jnp.array([1.0, 3.0])
    column = 2.0
    derivative = jax.jit(
        jax.jacrev(
            lambda amplitudes: emission_polarisation(
                amplitudes, depths, lam, weights, source_column=column
            ).real
        )
    )(spectrum)
    oracle = np.zeros((3, 2, 3))
    for k in range(3):
        oracle[k, :, k] = (
            column
            * np.asarray(weights)
            / 4
            * np.cos(2 * np.asarray(depths) * float(lam[k]) ** 2)
        )
    assert_allclose(derivative, oracle, atol=1e-14)
    delta_depth = np.array([0.08, -0.1])
    delta_emission = np.array([[0.02j, 0.03, -0.01], [0.01, -0.02j, 0.04]])
    reference = emission_polarisation(
        spectrum, depths, lam, weights, source_column=column
    )
    changed = emission_polarisation(
        spectrum + delta_emission,
        depths + delta_depth,
        lam,
        weights,
        source_column=column,
    )
    bound = column * (
        np.average(np.abs(delta_emission), axis=0, weights=weights)
        + 2
        * np.asarray(lam) ** 2
        * np.average(
            np.abs(spectrum) * np.abs(delta_depth[:, None]), axis=0, weights=weights
        )
    )
    assert np.all(np.abs(changed - reference) <= bound + 1e-14)


def test_joint_moments_extreme_weights_and_compiled_sample_gradients():
    basis = jnp.array([[1.0, 0.3], [1.0, -0.7], [1.0, 0.2]])
    depth = jnp.array([-0.3, 0.0, 0.9])
    weights = jnp.array([1.0, 2.0, 4.0])
    fn = eqx.filter_jit(joint_faraday_moments)
    actual = fn(basis, depth, 3, weights * 4e307, reference_depth=0.1)
    reference = fn(basis, depth, 3, weights, reference_depth=0.1)
    for got, expected in zip(actual, reference):
        assert_allclose(got, expected, atol=1e-14)
    derivative = jax.jit(
        jax.jacrev(
            lambda d: joint_faraday_moments(basis, d, 3, weights, reference_depth=0.1)[
                0
            ]
        )
    )(depth)
    delta = np.asarray(depth) - 0.1
    for b in range(4):
        oracle = (
            np.zeros((2, 3))
            if b == 0
            else np.asarray(basis).T * np.asarray(weights) / 7 * b * delta ** (b - 1)
        )
        assert_allclose(derivative[:, b, :], oracle, atol=1e-14)


def test_finite_response_supports_a_constrained_synthetic_spectral_fit():
    # Fit two mixture weights, with the third fixed by normalization. This
    # is a same-family identifiability check, not an astrophysical validation.
    lam = np.linspace(0.05, 0.7, 30)
    depths = np.array([-0.5, 0.1, 0.8])
    basis = np.column_stack([np.ones(3), [-0.8, 0.2, 1.0]])
    coefficients = np.column_stack([1 + 0.2j * lam, 0.5 + lam**2])
    true_weights = np.array([0.2, 0.5, 0.3])
    data = np.asarray(
        emission_polarisation(basis @ coefficients.T, depths, lam, true_weights)
    )
    components = []
    for i in range(3):
        moments, next_absolute = joint_faraday_moments(
            basis[i : i + 1], depths[i : i + 1], 12
        )
        value, _ = joint_faraday_average(
            coefficients, moments, lam, absolute_next=next_absolute, source_error=0.0
        )
        components.append(np.asarray(value))
    components = np.stack(components, axis=1)
    design = components[:, :2] - components[:, 2:3]
    target = data - components[:, 2]
    fitted, _, rank, _ = np.linalg.lstsq(
        np.vstack([design.real, design.imag]),
        np.r_[target.real, target.imag],
        rcond=None,
    )
    fitted_weights = np.r_[fitted, 1 - fitted.sum()]
    assert rank == 2 and np.all(fitted_weights > 0)
    assert_allclose(fitted_weights, true_weights, atol=2e-12)


def test_downstream_depth_keeps_field_reversals_and_near_side_boundary():
    path = np.array([0.0, 0.2, 0.7, 1.0])
    density = np.ones_like(path)
    field = 2 * path - 1
    # The integral from s to 1 is s-s^2; total depth vanishes although
    # intermediate emitters still have positive intervening depth.
    depths = faraday_depth_practical(density, field, path)
    assert_allclose(depths, 0.812 * (path - path**2), atol=1e-16)
    assert_allclose(
        depths[0], rotation_measure_practical(density, field, path), atol=1e-16
    )
    assert depths[-1] == 0


def test_joint_moments_remainder_and_zero_phase_marginal():
    depths = np.array([-0.4, 0.1, 0.6])
    x = np.array([-0.8, 0.2, 1.0])
    basis = np.column_stack([np.ones(3), x, np.exp(2j * x)])
    weights = np.array([1.0, 2.0, 4.0])
    lam = np.linspace(0, 0.8, 9)
    coefficients = np.column_stack([1 + lam, 0.3j * lam, -0.2 + lam**2])
    true_emission = basis @ coefficients.T + 0.02 * x[:, None] ** 3 * (1 + lam)
    source_error = 0.02 * np.average(np.abs(x) ** 3, weights=weights) * (1 + lam)
    direct = emission_polarisation(
        true_emission, depths, lam, weights, source_column=5.0
    )
    previous = np.inf
    for degree in [0, 2, 4, 8]:
        moments, next_absolute = eqx.filter_jit(joint_faraday_moments)(
            basis, depths, degree, weights, reference_depth=0.1
        )
        for power in range(degree + 1):
            oracle = np.average(
                basis * (depths[:, None] - 0.1) ** power, axis=0, weights=weights
            )
            assert_allclose(moments[:, power], oracle, atol=1e-14)
        assert_allclose(
            next_absolute,
            np.average(
                np.abs(basis) * np.abs(depths[:, None] - 0.1) ** (degree + 1),
                axis=0,
                weights=weights,
            ),
            atol=1e-14,
        )
        predicted, error = jax.jit(joint_faraday_average)(
            coefficients,
            moments,
            lam,
            reference_depth=0.1,
            source_column=5.0,
            absolute_next=next_absolute,
            source_error=source_error,
        )
        assert np.all(np.abs(predicted - direct) <= error + 2e-14)
        assert float(np.max(error)) < previous
        previous = float(np.max(error))
    # A vanishing zeroth complex statistic must not erase mixed moments.
    moments, absolute_next = joint_faraday_moments([[1.0], [-1.0]], [-1, 1], 8)
    assert moments[0, 0] == 0
    predicted, error = joint_faraday_average(
        [1.0], moments, 0.5, absolute_next=absolute_next, source_error=0.0
    )
    assert abs(predicted + 1j * np.sin(0.5)) <= error + 1e-14
    assert abs(predicted) > 0.4


def test_finite_response_is_linear_in_fitted_moments_and_has_zero_depth_gradients():
    basis = jnp.array([[1.0, 0.3], [1.0, -0.5]])
    moments, absolute_next = joint_faraday_moments(basis, [0.0, 0.0], 2)
    coefficients = jnp.array([[1.0, 0.2j], [0.4j, -0.7]])
    lam = jnp.array([0.0, 0.6])

    def predict(m):
        return joint_faraday_average(
            coefficients, m, lam, absolute_next=absolute_next, source_error=0.0
        )[0].real

    actual = jax.jit(jax.jacrev(predict))(moments)
    t = 2 * np.asarray(lam) ** 2
    phase_coefficients = np.stack([np.ones(2), 1j * t, -t * t / 2], axis=-1)
    oracle = np.real(
        np.asarray(coefficients)[:, :, None] * phase_coefficients[:, None, :]
    )
    assert_allclose(actual, oracle, atol=1e-14)
    derivative = jax.jacrev(lambda d: joint_faraday_moments(basis, d, 2)[0])(
        jnp.zeros(2)
    )
    assert np.all(np.isfinite(derivative))
    assert_allclose(derivative[:, 1, :], np.asarray(basis).T / 2, atol=1e-14)


@pytest.mark.parametrize(
    "emission,depths,lam,weights,column",
    [
        ([1, 2], [], 0.3, None, 1),
        ([1], [0, 1], 0.3, None, 1),
        ([[1, 2], [3, 4]], [0, 1], [0.1, 0.2, 0.3], None, 1),
        ([np.nan, 2], [0, 1], 0.3, None, 1),
        ([1, 2], [0, np.inf], 0.3, None, 1),
        ([1, 2], [0, 1j], 0.3, None, 1),
        ([1, 2], [0, 1], 0.3j, None, 1),
        ([1, 2], [0, 1], 0.3, [-1, 2], 1),
        ([1, 2], [0, 1], 0.3, [0, 0], 1),
        ([1, 2], [0, 1], 0.3, None, -1),
        ([1, 2], [0, 1], 0.3, None, np.inf),
        ([1, 2], [0, 1], 1e308, None, 1),
    ],
)
def test_invalid_emission_inputs_fail_in_eager_and_jit(
    emission, depths, lam, weights, column
):
    for fn in [emission_polarisation, eqx.filter_jit(emission_polarisation)]:
        with pytest.raises(Exception):
            fn(emission, depths, lam, weights, source_column=column)


def test_finite_statistic_shape_realness_and_error_contracts():
    for degree in [-1, 1.5, True]:
        with pytest.raises(ValueError, match="degree"):
            joint_faraday_moments([[1], [2]], [0, 1], degree)
    for basis in [[1, 2], [[1], [np.nan]], np.ones((2, 0))]:
        with pytest.raises(Exception):
            joint_faraday_moments(basis, [0, 1], 2)
    m, a = joint_faraday_moments([[1], [2]], [0, 1], 2)
    for kwargs in [
        dict(absolute_next=[-1.0], source_error=0.0),
        dict(absolute_next=[1j], source_error=0.0),
        dict(absolute_next=a, source_error=-1.0),
        dict(absolute_next=a, source_error=np.nan),
    ]:
        with pytest.raises(Exception):
            eqx.filter_jit(joint_faraday_average)([1.0], m, 0.3, **kwargs)
    with pytest.raises(ValueError, match="coefficients"):
        joint_faraday_average([1.0, 2.0], m, 0.3, absolute_next=a, source_error=0.0)
    for args in [([1], [1, 1], [0, 1]), ([1, 1], [1, 1], [1, 0])]:
        with pytest.raises(ValueError):
            faraday_depth_practical(*args)
    with pytest.raises(ValueError, match="real"):
        faraday_depth_practical([1.0, 1.0], np.array([1.0, 1j]), [0.0, 1.0])


def test_overflow_is_rejected_in_phase_moments_and_depth_quadrature():
    with pytest.raises(Exception, match="finite"):
        eqx.filter_jit(joint_faraday_moments)([[1.0], [1.0]], [1e308, 1e308], 2)
    with pytest.raises(Exception, match="finite"):
        eqx.filter_jit(joint_faraday_average)(
            [1.0], [[1.0, 1.0]], 1e154, absolute_next=[1.0], source_error=0.0
        )
    with pytest.raises(ValueError, match="overflowed"):
        faraday_depth_practical([1e308, 1e308], [1e308, 1e308], [0.0, 1.0])
