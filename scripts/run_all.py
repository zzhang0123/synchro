"""Run every validation, design and figure script, reporting pass/fail.

Usage: PYTHONPATH=. python3 scripts/run_all.py [--no-figures]

The figure scripts are included by default so that figures/*.pdf cannot drift
out of step with the physics modules; pass --no-figures to skip them.
"""

from __future__ import annotations

import subprocess
import sys
import os

SCRIPTS = [
    "validate_model.py",      # emissivity side (7 checks)
    "validate_transfer.py",   # RT transfer (5 checks)
    "validate_kirchhoff.py",  # absorption / Kirchhoff (6 checks)
    "validate_rt_moments.py", # Magnus / conversion / cross-cumulants (4 checks)
    "validate_sed.py",        # SED / spectral index / LOS cumulants (6 checks)
    "validate_cumulants.py",  # strict cumulant expansion (Gaussian exactness)
    "test_design.py",         # JAX/Equinox jit + differentiability
]

# Figure-producing scripts. They are the paper's *products*: if the physics
# modules change and these are not re-run, figures/*.pdf silently goes stale.
FIGURES = [
    "plot_derivative_spectra.py",  # figures/derivative_spectra.pdf   (Sec. 4.4)
    "compare_moment_cumulant.py",  # figures/moment_vs_cumulant.pdf   (Sec. 4.2)
    "application_foreground.py",   # figures/foreground_demo.pdf      (Sec. 5.5)
    "plot_transfer.py",            # figures/transfer.pdf             (Sec. 6)
]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    figures = "--no-figures" not in sys.argv
    names = SCRIPTS + (FIGURES if figures else [])
    failures = []
    for name in names:
        path = os.path.join(ROOT, "scripts", name)
        r = subprocess.run([sys.executable, path], cwd=ROOT,
                           capture_output=True, text=True,
                           env={**os.environ, "PYTHONPATH": ROOT})
        status = "PASS" if r.returncode == 0 else "FAIL"
        print(f"[{status}] {name}")
        if r.returncode != 0:
            failures.append(name)
            print("    " + (r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "(no stderr)"))
    print()
    if failures:
        print(f"{len(failures)} script(s) failed: {', '.join(failures)}")
        sys.exit(1)
    print(f"all {len(names)} scripts passed"
          + ("" if figures else "  (figures skipped: --no-figures)"))


if __name__ == "__main__":
    main()
