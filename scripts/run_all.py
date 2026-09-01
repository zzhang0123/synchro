"""Run every validation and design script, reporting pass/fail.

Usage: PYTHONPATH=. python3 scripts/run_all.py
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
    "test_design.py",         # JAX/Equinox jit + differentiability
]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    failures = []
    for name in SCRIPTS:
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
    print(f"all {len(SCRIPTS)} scripts passed")


if __name__ == "__main__":
    main()
