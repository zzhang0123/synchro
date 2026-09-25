"""The references of the AD-transform matrix against v0.2.0.

The matrix (tests/test_ad_transform_matrix.py) compares every transform
with plain autodiff of the independent references in
tests/_ad_matrix_cases.py. Here those references, and the package's own
values and Jacobians, are checked against the released v0.2.0 (then named
synchro, plain autodiff, no custom rules) at the same arguments.

V020 holds, per group, the projections sum(a * cos(arange(a.size)))
of the value, of jacfwd(f) and of jacfwd(jacfwd(c . f)) (c the
matrix's cotangent), summed over the leaves. They were computed once with the
v0.2.0 export in jax 0.10.0 (DEV) by importing tests/_ad_matrix_cases.py
with syncmoments aliased to synchro (so each group's f is the
v0.2.0 function; bessel_band has no v0.2.0 counterpart and pins its
reference, syncmoments.bessel.bessel_jn; harmonic_lines pins v0.2.0's
harmonic_lines, three contours per point, where the package shares one). jax
0.10.2 (NEW) reproduces them to 2.1e-14 relative per projection. Tolerance:
1e-12 of sum(|a| |cos(...)|).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import syncmoments  # noqa: F401  (enables x64)
from _ad_matrix import cotangent
from _ad_matrix_cases import all_groups

GROUPS = {g.name: g for g in all_groups()}

# fmt: off
V020 = {  # (value, jacfwd, hessian) projections
    'transfer_slab[all]': (1.6692673188212772, -0.240036183878506, 3.170657157323343),
    'transfer_los[all]': (0.7143857166549571, -1.2603434460743193, 0.72715757557423),
    'transfer_los_scalar_ds[ds]': (0.8439240680115977, -0.425031923337418, -0.2045008638895711),
    'moment_driven_slab[all]': (0.6768948626759226, 1.0509948551116086, 0.005412289263678288),
    'moment_driven_slab_polarised[all]': (0.43229864803150886, 0.6650575557631188, -0.22631706917528666),
    'moment_driven_slab_cgs[all]': (5.5300270992024e-19, -0.3150972630544308, -0.010152356475162607),
    'gaussian_rm_cumulants[all]': (1.7165520920603148, 1.6808097314854906, 0.539259509626256),
    'gaussian_rm_cumulants_unweighted[rms]': (2.0418388730680017, 0.22816401865442928, 0.0817814413053538),
    'screen_polarisation[all]': (1.051375611214792, 0.04257060593699008, 2.4953129929607147),
    'mixed_moments[all]': (1.0690529772326864, 0.19179384310833117, -0.06992728826087076),
    'emission_polarisation[all]': (1.0382389128517955, 0.9855277354203175, 4.6980250368388),
    'joint_faraday_moments[all]': (-7.405636239178873, -21.89775110913337, 4.422706615158179),
    'PopulationSamples.normalised_weights[weights]': (-0.12870252283533654, 0.7594282099911962, -0.054055786143747196),
    'PopulationSamples.product[wg,wd]': (0.015933244241496114, 0.056096331240411086, -0.07677513998744175),
    'JointMoments.from_samples[all]': (1.6453535120660239, 0.8413398330884451, -5.0877439305795855),
    'EmpiricalScreen[all]': (1.205982775041039, 0.05899334002690665, 1.5930962667103024),
    'bessel_band[x]': (1.1165706262670254, 0.26487751267383836, 0.3375290446905378),
    'harmonic_lines[all]': (-0.4906663018021704, 6.239929232864525, 19.662751703155266),
}
# fmt: on


def _projection(tree):
    """(sum a cos(k), sum |a cos(k)|) over the leaves a (k flat index)."""
    total = scale = 0.0
    for leaf in jax.tree_util.tree_leaves(tree):
        weight = jnp.cos(jnp.arange(leaf.size, dtype=float)).reshape(leaf.shape)
        total += float(jnp.sum(leaf * weight))
        scale += float(jnp.sum(jnp.abs(leaf * weight)))
    return total, scale


def _close(tree, pinned):
    total, scale = _projection(tree)
    assert abs(total - pinned) <= 1e-12 * scale, (total, pinned, scale)


@pytest.mark.parametrize("name", sorted(V020))
def test_reference_matches_v020(name):
    group = GROUPS[name]
    value, jac, _ = V020[name]
    _close(jax.jit(group.ref)(group.x), value)
    _close(jax.jit(jax.jacfwd(group.ref))(group.x), jac)


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(V020))
def test_reference_hessian_matches_v020(name):
    group = GROUPS[name]
    c = cotangent(group)
    hessian = jax.jacfwd(jax.jacfwd(lambda y: jnp.vdot(c, group.ref(y))))
    _close(jax.jit(hessian)(group.x), V020[name][2])


@pytest.mark.parametrize("name", sorted(set(V020) - {"bessel_band[x]"}))
def test_package_matches_v020(name):
    group = GROUPS[name]
    value, jac, _ = V020[name]
    _close(group.f(group.x), value)
    _close(jax.jit(jax.jacrev(group.f))(group.x), jac)
