"""High-precision reference constants of the boundary tests (mpmath, >= 30 digits).

The tests must not import mpmath (it is an optional dependency and absent in
some environments), so the references they compare their float64 pins with
are literals in ``tests/model/_boundary_oracles.py``. This script is the
generator and the check of those literals:

    PYTHONPATH=. python scripts/reference_constants.py          # print literals
    PYTHONPATH=. python scripts/reference_constants.py --check  # compare

Quantities (all dimensionless):

* ``J_REFERENCE[(m, x)] = (J_m(x), J_m'(x))`` at the Bessel corner cells;
* ``F_REFERENCE[x] = x int_x^inf K_{5/3}(t) dt`` and
  ``G_REFERENCE[x] = x K_{2/3}(x)`` (synchrotron functions);
* ``BUMP_AREA_REFERENCE = int_{-1}^{1} exp(1 - 1/(1 - t^2)) dt``.

Working precision ``WORKING_DPS = 40`` decimal digits; literals are printed with
``DIGITS = 32`` significant digits and ``--check`` requires agreement to
``1e-30`` relative. Quadratures are mpmath tanh-sinh with the break points
below; the integrands are smooth on each piece, so the working precision,
not the quadrature, limits the result (``--check`` also recomputes at 50 digits
and compares). Nothing here is imported by the package.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mpmath as mp

WORKING_DPS = 40
DIGITS = 32
CHECK_RTOL = mp.mpf("1e-30")

J_CELLS = ((300, 42.0), (300, 299.0), (1009, 1000.0), (16, 15.5), (1, 0.14))
FG_POINTS = (1e-6, 1e-3, 1.0, 10.0)


def bessel_pair(m, x):
    """``(J_m(x), J_m'(x))`` at the float64 argument ``x``."""
    x = mp.mpf(x)
    return mp.besselj(m, x), mp.besselj(m, x, derivative=1)


def synchrotron_F(x):
    """``x int_x^inf K_{5/3}(t) dt`` with decade break points from ``x`` to 100."""
    x = mp.mpf(x)
    points = [x]
    while points[-1] < 100:
        points.append(points[-1] * 10)
    points.append(mp.inf)
    return x * mp.quad(lambda t: mp.besselk(mp.mpf(5) / 3, t), points)


def synchrotron_G(x):
    x = mp.mpf(x)
    return x * mp.besselk(mp.mpf(2) / 3, x)


def bump_area():
    return mp.quad(lambda t: mp.exp(1 - 1 / (1 - t**2)), [-1, 0, 1])


def references(dps=WORKING_DPS):
    """All references at ``dps`` working digits, as mpmath numbers."""
    with mp.workdps(dps):
        return {
            "J_REFERENCE": {cell: bessel_pair(*cell) for cell in J_CELLS},
            "F_REFERENCE": {x: synchrotron_F(x) for x in FG_POINTS},
            "G_REFERENCE": {x: synchrotron_G(x) for x in FG_POINTS},
            "BUMP_AREA_REFERENCE": bump_area(),
        }


def _literal(value):
    if isinstance(value, tuple):
        return "(" + ", ".join(_literal(v) for v in value) + ")"
    return '"' + mp.nstr(value, DIGITS, min_fixed=1, max_fixed=0) + '"'


def render(refs):
    """Python source of the literal block for ``_boundary_oracles.py``."""
    lines = [
        f"# mpmath {mp.__version__}, mp.dps = {WORKING_DPS}, printed to {DIGITS}"
        " significant digits by scripts/reference_constants.py."
    ]
    for name, value in refs.items():
        if isinstance(value, dict):
            lines.append(f"{name} = {{")
            lines += [f"    {key!r}: {_literal(v)}," for key, v in value.items()]
            lines.append("}")
        else:
            lines.append(f"{name} = {_literal(value)}")
    return "\n".join(lines)


def _flatten(refs):
    for name, value in refs.items():
        items = value.items() if isinstance(value, dict) else [(None, value)]
        for key, v in items:
            parts = v if isinstance(v, tuple) else (v,)
            for i, part in enumerate(parts):
                yield (name, key, i), part


def check():
    """Compare the test literals with fresh 40- and 50-digit evaluations."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "model"))
    import _boundary_oracles as oracles  # noqa: PLC0415

    stored = {name: getattr(oracles, name) for name in references(15)}
    worst = mp.mpf(0)
    with mp.workdps(60):
        fresh = dict(_flatten(references(WORKING_DPS)))
        finer = dict(_flatten(references(50)))
        for label, literal in _flatten(stored):
            ref = finer[label]
            scale = max(abs(ref), mp.mpf("1e-300"))
            worst = max(worst, abs(mp.mpf(literal) - ref) / scale)
            worst = max(worst, abs(fresh[label] - ref) / scale)
    print(f"largest relative difference: {mp.nstr(worst, 3)}")
    return worst <= CHECK_RTOL


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.check:
        return 0 if check() else 1
    print(render(references()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
