"""Reproduce a finite, independent radiation-kernel audit and write its evidence.

SciPy supplies the comparison special functions; F's reference is a separate
log-frequency integral of scipy.special.kv. Tolerances are acceptance criteria
for the declared points, not universal kernel or physical error bounds. The
analytic omitted-tail bounds and full-physics discrepancies are separate from
these observed quadrature differences. Run from any directory; the default
record is review-results/radiation.json alongside this script's source tree.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import jax
import numpy as np
from scipy.integrate import quad
from scipy.special import jv, jvp, kv

from synchro.bessel import bessel_jn_and_prime, bessel_kn
from synchro.ultrarel import F, G

ROOT = Path(__file__).resolve().parents[1]


def _validate_import_roots() -> dict[str, str]:
    """Reject a shadow/installed package before recording this checkout's evidence."""
    expected = (ROOT / "synchro").resolve()
    paths = {}
    for name, module in tuple(sys.modules.items()):
        if name != "synchro" and not name.startswith("synchro."):
            continue
        filename = getattr(module, "__file__", None)
        actual = Path(filename).resolve() if filename is not None else None
        if actual is None or not actual.is_relative_to(expected):
            raise RuntimeError(
                f"Radiation audit provenance mismatch: imported {name} from "
                f"{actual}; expected a module under {expected}. "
                "Run with PYTHONPATH set to this checkout or install this checkout."
            )
        paths[name] = str(actual)
    required = {"synchro", "synchro.bessel", "synchro.ultrarel"}
    if not required.issubset(paths):
        raise RuntimeError(
            "Radiation audit provenance mismatch: required imported modules are missing"
        )
    return paths


def _source_metadata(import_paths: dict[str, str]) -> dict[str, Any]:
    files = sorted((ROOT / "synchro").glob("*.py")) + [Path(__file__).resolve()]
    hashes = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "git_head": revision.stdout.strip() if revision.returncode == 0 else None,
        "sha256": hashes,
        "import_paths": import_paths,
        "imported_file_sha256": {
            name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for name, path in import_paths.items()
        },
        "note": "Imported module paths are checked against this checkout before computation; per-file hashes identify uncommitted changes independently of git HEAD.",
    }


def _comparison(
    value: Any, reference: Any, rtol: float, floor: float = 1e-280
) -> dict[str, Any]:
    value, reference = float(value), float(reference)
    if not (np.isfinite(value) and np.isfinite(reference)):
        # Preserve a failed numerical case in valid JSON instead of losing
        # the evidence when allow_nan=False rejects an IEEE NaN/Infinity.
        return {
            "value": value if np.isfinite(value) else None,
            "reference": reference if np.isfinite(reference) else None,
            "nonfinite_values": {"value": repr(value), "reference": repr(reference)},
            "absolute_error": None,
            "relative_error": None,
            "criterion": None,
            "passed": False,
        }
    absolute = abs(value - reference)
    relative = abs(value / reference - 1) if abs(reference) > floor else None
    limit = rtol * abs(reference) if relative is not None else floor
    return {
        "value": value,
        "reference": reference,
        "absolute_error": absolute,
        "relative_error": relative,
        "criterion": limit,
        "passed": bool(np.isfinite(value) and absolute <= limit),
    }


def collect() -> dict[str, Any]:
    import_paths = _validate_import_roots()
    tolerances = {
        "J_relative": 2e-10,
        "J_prime_relative": 2e-9,
        "K_relative": 2e-10,
        "F_relative": 3e-10,
        "F_prime_relative": 3e-9,
        "G_relative": 2e-10,
        "tiny_reference_absolute_floor": 1e-280,
    }
    j_cases = []
    for n in (1, 5, 20, 80, 512, 1000, 3200):
        for ratio in (0.1, 0.5, 0.9, 0.999, 1.0, 1.01, 1.5):
            x = float(n * ratio)
            value, prime = bessel_jn_and_prime(n, x)
            j_cases.append(
                {
                    "n": n,
                    "x": x,
                    "J": _comparison(value, jv(n, x), tolerances["J_relative"]),
                    "J_prime": _comparison(
                        prime, jvp(n, x), tolerances["J_prime_relative"]
                    ),
                }
            )
    k_cases = []
    xs = np.geomspace(1e-12, 500, 80)
    for order in (0.0, 2 / 3, 5 / 3):
        values, references = np.asarray(bessel_kn(order, xs)), kv(order, xs)
        for x, value, reference in zip(xs, values, references, strict=True):
            k_cases.append(
                {
                    "order": order,
                    "x": float(x),
                    **_comparison(value, reference, tolerances["K_relative"]),
                }
            )
    continuum_cases = []
    for x in (1e-7, 0.0099, 0.01, 0.0101, 0.1, 1.0, 20.0):
        integral, error = quad(
            lambda u: np.exp(u) * kv(5 / 3, np.exp(u)),
            np.log(x),
            np.log(x + 80),
            epsabs=1e-11,
            epsrel=1e-11,
        )
        reference = x * integral
        derivative = reference / x - x * kv(5 / 3, x)
        continuum_cases.append(
            {
                "x": x,
                "F": _comparison(F(x), reference, tolerances["F_relative"]),
                "F_prime": _comparison(
                    jax.grad(F)(x), derivative, tolerances["F_prime_relative"]
                ),
                "G": _comparison(G(x), x * kv(2 / 3, x), tolerances["G_relative"]),
                "reference_quad_error_estimate": float(x * error),
                "reference_upper_frequency": x + 80,
            }
        )
    comparisons = (
        [entry[key] for entry in j_cases for key in ("J", "J_prime")]
        + k_cases
        + [entry[key] for entry in continuum_cases for key in ("F", "F_prime", "G")]
    )

    def max_relative(entries):
        return max(
            (
                entry["relative_error"]
                for entry in entries
                if entry["relative_error"] is not None
            ),
            default=None,
        )

    return {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Finite independent numerical comparisons, not a uniform kernel or all-physics certificate.",
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            **{
                name: version(name)
                for name in ("jax", "jaxlib", "equinox", "numpy", "scipy")
            },
            "jax_x64": bool(jax.config.x64_enabled),
        },
        "source": _source_metadata(import_paths),
        "tolerances": tolerances,
        "error_interpretation": [
            "The 49 integer-J cases test values and argument derivatives, including exponentially small tails and both sides of x=n.",
            "K uses80 logarithmic points for each of three orders; the finite sweep does not prove uniform accuracy between points.",
            "F's reference uses a separate SciPy K integral with upper frequency x+80. Its reported quad estimate excludes that reference's remaining upper tail and roundoff.",
            "The public K/F positive-tail bounds control their finite integration limit only; quadrature and floating-point error require separate validation.",
            "Reference physics, PDF closure, support exclusions, statistics uncertainty and observation/inference errors are not tested by this numerical-kernel audit.",
        ],
        "summary": {
            "passed": all(entry["passed"] for entry in comparisons),
            "scalar_comparisons": len(comparisons),
            "max_J_relative_error": max_relative([entry["J"] for entry in j_cases]),
            "max_J_prime_relative_error": max_relative(
                [entry["J_prime"] for entry in j_cases]
            ),
            "max_K_relative_error": max_relative(k_cases),
            "max_F_relative_error": max_relative(
                [entry["F"] for entry in continuum_cases]
            ),
            "max_F_prime_relative_error": max_relative(
                [entry["F_prime"] for entry in continuum_cases]
            ),
            "max_G_relative_error": max_relative(
                [entry["G"] for entry in continuum_cases]
            ),
        },
        "integer_J_cases": j_cases,
        "modified_K_cases": k_cases,
        "continuum_cases": continuum_cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "review-results" / "radiation.json"
    )
    args = parser.parse_args()
    result = collect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["summary"], indent=2))
    print(f"Evidence written to {args.output}")
    if not result["summary"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
