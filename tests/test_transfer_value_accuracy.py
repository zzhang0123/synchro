"""Accuracy of the value of ``transfer_slab`` against mpmath (40 digits).

Up to 0.3.0 the value's 5x5 exponential ``exp([[-K ds, eps ds 2^-k], [0, 0]])``
was ``jax.scipy.linalg.expm``, whose ``floor`` squaring count applies Pade 13
up to ``2 theta_13``. Just below ``|A|_1 = 2^(m+1) theta_13`` that cost relative
errors of 1.3e-9 to 1.7e-9 for a Faraday-dominated slab (m = 0 to 5) and up to
7.2e-10 on the realistic spectrum below (n_e = 0.03 cm^-3, B_par = 3 uG,
L = 1 kpc, 50 MHz to 20 GHz). Since 0.4.0 the value uses
``syncmoments._expm.expm_pade13`` (``ceil``), as the derivatives already did.

Error model (``A = -K ds``, ``u = 2^-53``, max-norm relative to the largest
entry of the truth): ``|out - truth| <= c u max(|A|_1, 1) |truth|``. A squaring
rounds at ``u`` and a rotation's exponential has condition number about
``|A|_1``, so ``c`` is not a rigorous constant: it is measured, at most 4.1 over
214 slabs (150 random, 24 Faraday slabs at ``2^(m+1) theta_13 (1 +- 1e-9)``,
m = 0..11, and 40 frequencies of the spectrum below), where 0.3.0 reached
``1.3e6 u max(|A|_1, 1)``. The tolerance is ``c = 16``. Where both methods are
at roundoff, ``ceil`` can be a few ulp worse than ``floor`` (one more squaring,
or Pade 13 where ``floor`` used a lower degree; at most 3.8 u in the 214
slabs); that is within the model.

The literals are ``mpmath.expm`` of the 5x5 matrix (inputs converted exactly,
``mp.dps = 40``) applied to ``[S; 1]``, rounded to float64. The test with
random slabs recomputes its truth and needs mpmath.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import syncmoments  # noqa: F401  (enables x64)
from syncmoments.rm import _pow2
from syncmoments.transfer import _source_exponent, mueller_matrix, transfer_slab

U64, U32 = 2.0**-53, 2.0**-24
C = 16.0
WRAPS = {"eager": lambda f: f, "jit": jax.jit}

FARADAY_K = [[0.001, 0.0, 0.0, 0.0], [0.0, 0.001, 1.0, 0.0], [0.0, -1.0, 0.001, 0.0], [0.0, 0.0, 0.0, 0.001]]  # fmt: skip
# name: (S, eps, K, ds, mpmath truth). Comment: 0.3.0 relative error, |K ds|_1.
# fmt: off
VALUE_CASES = {
    "faraday m=0 below": (  # 1.6e-09, 10.74
        [1.0, 0.6, 0.3, 0.1], [0.5, 0.2, -0.1, 0.0], FARADAY_K, 10.733107583968497,
        [6.327180940662022, 0.06754442450915525, -0.3033256864547936, 0.09893242866920601],
    ),
    "faraday m=3 below": (  # 1.7e-09, 85.95
        [1.0, 0.6, 0.3, 0.1], [0.5, 0.2, -0.1, 0.0], FARADAY_K, 85.86486067174798,
        [42.05859690897283, -0.05194522622001126, -0.24245094822096808, 0.09177182426673891],
    ),
    "faraday m=8 below": (  # 3.0e-10, 2750
        [1.0, 0.6, 0.3, 0.1], [0.5, 0.2, -0.1, 0.0], FARADAY_K, 2747.6755414959352,
        [468.02576077881736, 0.08311614703066084, 0.22773716627412552, 0.006407663170577679],
    ),
    "cgs 794 MHz": (  # 7.2e-10, 20.83
        [0.0, 0.0, 0.0, 0.0],
        [3.0110342524420384e-41, -2.3412100832208287e-41, -1.6017079939920856e-41, 0.0],
        [[3.347905901729492e-33, -2.5663866670434086e-33, -1.755759583361751e-33, 0.0], [-2.5663866670434086e-33, 3.347905901729492e-33, 6.748775622320054e-21, 5.596391014698052e-29], [-1.755759583361751e-33, -6.748775622320054e-21, 3.347905901729492e-33, -8.18022206444845e-29], [0.0, -5.596391014698052e-29, 8.18022206444845e-29, 3.347905901729492e-33]],
        3.086e+21,
        [9.292051702987859e-20, 1.2488134197212458e-22, -7.020391245021056e-21, 1.2289251761395546e-27],
    ),
    "cgs 199 MHz": (  # 6.4e-10, 330.8
        [0.0, 0.0, 0.0, 0.0],
        [2.0594017971180983e-38, -1.434189549743395e-38, -9.811818611184549e-39, 0.0],
        [[4.119592948044687e-31, -1.455171494353754e-31, -9.955363817369883e-32, 0.0], [-1.455171494353754e-31, 4.119592948044687e-31, 1.0719530259283658e-19, 3.5426985411013264e-27], [-9.955363817369883e-32, -1.0719530259283658e-19, 4.119592948044687e-31, -5.1783480992473145e-27], [0.0, -3.5426985411013264e-27, 5.1783480992473145e-27, 4.119592948044687e-31]],
        3.086e+21,
        [6.355313941866633e-17, 2.535486035856888e-19, -1.3919277432609577e-19, 3.1463986013652134e-24],
    ),
    "cgs 50 MHz": (  # 3.5e-10, 5254
        [0.0, 0.0, 0.0, 0.0],
        [7.773894396040051e-38, -4.545761383985092e-38, -3.109922684722475e-38, 0.0],
        [[7.558119339705503e-30, -6.0048780656282526e-30, -4.1081581142999474e-30, 0.0], [-6.0048780656282526e-30, 7.558119339705503e-30, 1.7026544577903208e-18, 2.2426440397318766e-25], [-4.1081581142999474e-30, -1.7026544577903208e-18, 7.558119339705503e-30, -3.278063703614946e-25], [0.0, -2.2426440397318766e-25, 3.278063703614946e-25, 7.558119339705503e-30]],
        3.086e+21,
        [2.3990237826401157e-16, -6.929675118726701e-21, -4.69866113786104e-20, 3.9641477340760687e-23],
    ),
    "random 67": (  # 8.5e-11, 78.72
        [-0.4210308622095371, -0.6616194265509461, -0.5554501579040428, -1.0777614685478931],
        [-0.07802007061163095, -0.008316489030606104, 0.009620440289182386, -0.047859780904161514],
        [[0.0059446508187127375, -0.0017028599508757333, 0.002819159471140476, -0.0005893681457473492], [-0.0017028599508757333, 0.0059446508187127375, -0.3229614453066279, 0.15567975951316906], [0.002819159471140476, 0.3229614453066279, 0.0059446508187127375, 1.6829603671312583], [-0.0005893681457473492, -0.15567975951316906, -1.6829603671312583, 0.0059446508187127375]],
        39.074383446564255,
        [-3.08427373913007, -0.3407088802199376, -0.6770239165369496, 0.845429246596451],
    ),
    "random 87": (  # 2.3e-11, 669.8
        [-0.9295695279875268, 0.25536736674286714, -1.8410225171059418, -0.06546352983029337],
        [-0.4180082067902966, -0.10717031200774295, 0.1827928269166433, -0.08258583261947025],
        [[0.004041364811464494, 0.0010673073524046046, 0.0009349497621858627, -0.0036310760632554816], [0.0010673073524046046, 0.004041364811464494, -19.646654885314994, -21.885920102752564], [0.0009349497621858627, 19.646654885314994, 0.004041364811464494, -4.83743715225271], [-0.0036310760632554816, 21.885920102752564, 4.83743715225271, 0.004041364811464494]],
        16.12619940054863,
        [-7.41527082246159, -0.3898341716922336, 2.3719749513811554, -0.5141150195193124],
    ),
    "random 21": (  # 8.1e-12, 42.61
        [0.39338573028374196, -0.5641124904961297, -0.7489737969543596, 0.12012568949433819],
        [0.052507179850311135, 0.1256884889086657, -0.25919328079989795, -0.12166220234012758],
        [[0.0030145446495151336, -0.0017418179703470925, -0.0017323877378783198, 0.0013996978044906205], [-0.0017418179703470925, 0.0030145446495151336, 0.8220696155266851, -0.6040665793735749], [-0.0017323877378783198, -0.8220696155266851, 0.0030145446495151336, -0.3479983757862944], [0.0013996978044906205, 0.6040665793735749, 0.3479983757862944, 0.0030145446495151336]],
        29.7785871195684,
        [1.940993508492476, 2.5968560268475294, -5.271800458467193, -5.635808262659174],
    ),
}
# fmt: on

THICK = mueller_matrix(5.0, 1.0, -0.5, 0.2, 4.0, -2.0, 1.0)
WEAK = mueller_matrix(0.3, 0.1, -0.05, 0.02, 0.4, -0.2, 0.1)
S0 = [1.0, 0.2, -0.1, 0.05]
D = np.array([1.0, 0.3, -0.2, 0.1])
# tests/test_transfer_derivatives.py PRIMAL_INPUTS: (S, eps, K, ds, dtype, truth)
# fmt: off
PRIMAL_CASES = {
    "weak": (S0, D * 0.7, WEAK, 0.9, jnp.float64, [1.2925076273887441, 0.2268301615000305, -0.1435647756558606, 0.07002150603992392]),
    "thick 1e306": (S0, D * 1e306, THICK, 20.0, jnp.float64, [1.930944063958339e+305, 2.330316949509952e+304, -2.0309706521440166e+304, 5.3497263250543204e+303]),
    "rotation": (S0, D, mueller_matrix(0, 0, 0, 0, 0.3, 0.2, 1.8), 1.0, jnp.float64, [2.0, 0.3718332218308052, 0.30959199618506994, 0.10362868567430247]),
    "tau 1e6": (S0, D, 1e6 * np.eye(4), 1.0, jnp.float64, [1e-06, 3e-07, -2.0000000000000002e-07, 1.0000000000000001e-07]),
    "weak f32": (S0, D * 0.7, WEAK, 0.9, jnp.float32, [1.29250760016779, 0.22683015574573268, -0.1435647752853166, 0.07002150594978676]),
    "rotation f32": (S0, D, mueller_matrix(0, 0, 0, 0, 0.3, 0.2, 1.8), 1.0, jnp.float32, [2.0, 0.37183324663494277, 0.3095919963477488, 0.10362868435530226]),
    "tau 1e6 f32": (S0, D, 1e6 * np.eye(4), 1.0, jnp.float32, [1e-06, 3.00000011920929e-07, -2.0000000298023224e-07, 1.0000000149011612e-07]),
}
# fmt: on


def _rel(got, truth):
    got, truth = np.asarray(got, np.float64), np.asarray(truth)
    return float(np.max(np.abs(got - truth)) / np.max(np.abs(truth)))


def _norm(K, ds):
    return float(np.max(np.sum(np.abs(np.asarray(K) * ds), axis=0)))


def _floor_value(S, eps, K, ds):
    """The 0.3.0 value: the same 5x5 matrix through ``jax.scipy.linalg.expm``."""
    column = eps * ds
    k = _source_exponent(jnp.max(jnp.abs(column)))
    M = jnp.zeros((5, 5)).at[:4, :4].set(-K * ds)
    M = M.at[:4, 4].set(column * _pow2(-k, column.dtype))
    y0 = jnp.concatenate([S, _pow2(k, column.dtype)[None]])
    return (jax.scipy.linalg.expm(M, max_squarings=32) @ y0)[:4]


@pytest.mark.parametrize("wrap", WRAPS, ids=list(WRAPS))
@pytest.mark.parametrize("name", VALUE_CASES)
def test_value_matches_mpmath(name, wrap):
    S, eps, K, ds, truth = (np.asarray(a, np.float64) for a in VALUE_CASES[name])
    got = WRAPS[wrap](transfer_slab)(S, eps, K, ds)
    assert _rel(got, truth) <= C * U64 * max(_norm(K, ds), 1.0)


def test_value_under_vmap_matches_mpmath():
    names = list(VALUE_CASES)
    S, eps, K, ds, truth = (
        jnp.asarray(np.stack([VALUE_CASES[n][i] for n in names])) for i in range(5)
    )
    got = jax.jit(jax.vmap(transfer_slab))(S, eps, K, ds)
    for i in range(len(names)):
        bound = C * U64 * max(_norm(K[i], float(ds[i])), 1.0)
        assert _rel(got[i], truth[i]) <= bound, names[i]


@pytest.mark.parametrize("name", VALUE_CASES)
def test_floor_squaring_count_was_less_accurate(name):
    """Regression record: the 0.3.0 value (``floor``) misses the same truths by
    8e-12 to 1.7e-9, i.e. 19 to 800 times the bound above."""
    S, eps, K, ds, truth = (jnp.asarray(a) for a in VALUE_CASES[name])
    err = _rel(_floor_value(S, eps, K, ds), truth)
    assert err > 10 * C * U64 * max(_norm(K, float(ds)), 1.0)


@pytest.mark.parametrize("name", PRIMAL_CASES)
def test_primal_inputs_match_mpmath(name):
    """The inputs whose values were pinned bit for bit to 0.3.0 before 0.4.0
    (``tests/test_transfer_derivatives.py``); float32 against the truth of the
    float32-rounded inputs, with ``u = 2^-24``."""
    S, eps, K, ds, dtype, truth = PRIMAL_CASES[name]
    S, eps, K = (jnp.asarray(a, dtype) for a in (S, eps, K))
    got = transfer_slab(S, eps, K, dtype(ds))
    assert got.dtype == dtype
    u = U64 if dtype == jnp.float64 else U32
    assert _rel(got, truth) <= C * u * max(_norm(K, ds), 1.0)


def _random_slab(rng):
    a = 10.0 ** rng.uniform(-4, 1)
    rot = rng.uniform(-1, 1, 3) * 10.0 ** rng.uniform(-2, 1.5)
    K = np.asarray(mueller_matrix(a, *(a * rng.uniform(-0.9, 0.9, 3)), *rot))
    ds = 10.0 ** rng.uniform(-1, 2.5)
    return rng.normal(size=4), rng.normal(size=4) * 10.0 ** rng.uniform(-3, 3), K, ds


def _mp_value(S, eps, K, ds):
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 40
    M = mp.zeros(5, 5)
    for r in range(4):
        for c in range(4):
            M[r, c] = -mp.mpf(float(K[r, c])) * mp.mpf(float(ds))
        M[r, 4] = mp.mpf(float(eps[r])) * mp.mpf(float(ds))
    E, y = mp.expm(M), [mp.mpf(float(s)) for s in S] + [mp.mpf(1)]
    return np.array(
        [float(mp.fsum(E[r, c] * y[c] for c in range(5))) for r in range(4)]
    )


def test_random_slabs_match_mpmath():
    """40 random slabs, ``|K ds|_1`` from 1e-3 to 1e4 (needs mpmath)."""
    pytest.importorskip("mpmath")
    rng = np.random.default_rng(11)
    slabs = [_random_slab(rng) for _ in range(40)]
    f = jax.jit(transfer_slab)
    for i, (S, eps, K, ds) in enumerate(slabs):
        err = _rel(f(S, eps, K, ds), _mp_value(S, eps, K, ds))
        assert err <= C * U64 * max(_norm(K, ds), 1.0), i
