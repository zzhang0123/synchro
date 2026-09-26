"""Run every validation, design and figure script, reporting process completion.

Usage: PYTHONPATH=. python3 scripts/run_all.py [--no-figures]

The figure scripts are included by default so that figures/*.pdf cannot drift
out of step with the physics modules; pass --no-figures to skip them.
"""

from __future__ import annotations

import subprocess
import sys
import os

SCRIPTS = [
    "validate_model.py",  # emissivity side (7 checks)
    "validate_transfer.py",  # RT transfer (5 checks)
    "validate_kirchhoff.py",  # absorption / Kirchhoff (6 checks)
    "validate_rt_moments.py",  # Magnus / conversion / cross-cumulants (4 checks)
    "validate_sed.py",  # SED / spectral index / LOS cumulants (6 checks)
    "validate_cumulants.py",  # finite Bell contractions and scalar Gaussian example
    "test_design.py",  # JAX/Equinox jit + differentiability
]

# Companion demonstration figures; the manuscript uses separate independent
# generators. Regenerate these whenever their implementation or inputs change.
FIGURES = [
    "plot_derivative_spectra.py",  # figures/derivative_spectra.pdf   (Sec. 4.4)
    "compare_moment_cumulant.py",  # figures/moment_vs_cumulant.pdf   (Sec. 4.2)
    "application_foreground.py",  # figures/foreground_demo.pdf      (Sec. 5.5)
    "plot_transfer.py",  # figures/transfer.pdf             (Sec. 6)
    # figures/sed_moment_reconstruction.{pdf,png} and its JSON/NPZ/CSV data
    # (Sec. 5.3): public-API reduce_response + fit_combinations example
    "sed_reconstruction_example.py",
]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    # Diagnostics that print numbers are not scientific pass/fail assertions.
    # Run the actual regression suite before interpreting script completion.
    checked = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q"],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": ROOT},
    )
    if checked.returncode:
        sys.exit(checked.returncode)
    figures = "--no-figures" not in sys.argv
    names = SCRIPTS + (FIGURES if figures else [])
    failures = []
    for name in names:
        path = os.path.join(ROOT, "scripts", name)
        r = subprocess.run(
            [sys.executable, path],
            cwd=ROOT,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": ROOT},
        )
        status = "COMPLETED" if r.returncode == 0 else "FAIL"
        print(f"[{status}] {name}")
        if r.returncode != 0:
            failures.append(name)
            print(
                "    "
                + (
                    r.stderr.strip().splitlines()[-1]
                    if r.stderr.strip()
                    else "(no stderr)"
                )
            )
    print()
    if failures:
        print(f"{len(failures)} script(s) failed: {', '.join(failures)}")
        sys.exit(1)
    print(
        f"all {len(names)} scripts completed (scientific assertions live in pytest)"
        + ("" if figures else "  (figures skipped: --no-figures)")
    )


if __name__ == "__main__":
    main()
