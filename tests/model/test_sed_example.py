"""The SED reconstruction example (``scripts/sed_reconstruction_example.py``).

The example ports the smooth correlated continuum population of main.tex
Appendix C.1 to the public API (``reduce_response``, ``fit_combinations``)
with an independent NumPy/SciPy full-population integration for the mock
data (``scripts/sed_reconstruction_direct.py``). The fast tests run the
``--quick`` configuration (18 centres, 6 fitted), which is not comparable
with the historical numbers. The ``slow`` test runs the full configuration
and checks the numbers of the historical research reference
(``validation/sed_reconstruction_results.json`` of the manuscript, package
b39cf0c) with the tolerances of the T-006 design, Section 9.5. Every check is
a finite check of one configuration and seed.
"""

import ast
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import sed_reconstruction_example as example  # noqa: E402

EXAMPLE_FILES = (
    "sed_reconstruction_example.py",
    "sed_reconstruction_setup.py",
    "sed_reconstruction_direct.py",
    "sed_reconstruction_checks.py",
    "sed_reconstruction_report.py",
    "sed_reconstruction_summary.py",
    "sed_reconstruction_figure.py",
)
SIBLINGS = {Path(name).stem for name in EXAMPLE_FILES}
OUTPUTS = (".json", "_arrays.npz", "_combinations.csv")
STEM = "sed_moment_reconstruction"

# Historical research reference (manuscript validation/, package b39cf0c).
HIST_S_TOP = (
    789.3509669343886,
    560.4682107122701,
    560.4682107122701,
    145.214282937884,
    111.74749734080099,
    111.74749734080076,
    30.49464822422513,
    30.49464822422513,
    19.495926119771205,
    7.6114006721164,
    7.6114006721163765,
    2.4777387012086156,
    1.2217030256585564,
    1.2217030256585557,
    0.3162769472391017,
    0.31627694723910066,
)
HIST_CHI2 = 55.2360702169915
HIST_HELDOUT_RMS = 0.004763691669233394
HIST_HELDOUT_MAX = 0.02021104816914708


def _strict_load(path):
    def reject(token):
        raise ValueError(f"non-strict JSON token {token}")

    return json.loads(Path(path).read_text(), parse_constant=reject)


@pytest.fixture(scope="module")
def quick(tmp_path_factory):
    out = tmp_path_factory.mktemp("sed_quick")
    payload = example.main(["--quick", "--out", str(out), "--no-figure"])
    return out, payload


def test_example_quick_runs_and_writes_outputs(quick):
    out, payload = quick
    for ext in OUTPUTS:
        assert (out / f"{STEM}{ext}").is_file()
    assert not (out / f"{STEM}.png").exists()  # --no-figure
    saved = _strict_load(out / f"{STEM}.json")
    for key in (
        "configuration",
        "reduction",
        "fit",
        "slots",
        "combinations",
        "sed",
        "residuals",
        "validation",
        "illustrative_prior",
        "claims",
        "limits",
        "provenance",
    ):
        assert key in saved, key
    assert saved["configuration"]["quick"] is True
    assert "not comparable" in saved["configuration"]["note"]
    prov = saved["provenance"]
    assert prov["syncmoments_version"]
    assert set(prov["git"]) >= {"revision", "status_porcelain"}
    assert any(k.endswith("fit/combinations.py") for k in prov["source_sha256"])
    assert set(prov["script_sha256"]) == set(EXAMPLE_FILES)
    red = saved["reduction"]
    assert (red["n_full"], red["n_active"], red["n_q"]) == (60, 52, 30)
    assert red["exact"] is True
    assert saved["fit"]["n_retained"] >= 1
    val = saved["validation"]
    assert val["grouped_vs_ungrouped_max_coefficient_difference"]["value"] < 1e-12
    assert val["moment_routes_max_difference"]["value"] < 1e-12
    assert val["forward_check_max_relative_difference"]["value"] < 1e-12
    scale = max(1.0, float(np.max(np.abs(saved["slots"]["truth"]))))
    assert val["decomposition_residual"]["value"] < 1e-10 * scale
    for item in val.values():
        assert item["label"] in ("finite check", "measured", "statistical")
    # the unresolved part is reported unbounded, never zero
    assert saved["fit"]["unresolved"]["kind"] == "unbounded"
    prior = saved["illustrative_prior"]
    assert prior["unresolved"]["kind"] == "bound"
    assert prior["covers_unresolved_truth"] is True
    assert payload["fit"]["n_retained"] == saved["fit"]["n_retained"]


def test_example_quick_arrays_and_csv(quick):
    out, payload = quick
    arrays = np.load(out / f"{STEM}_arrays.npz")
    n_R = payload["fit"]["n_retained"]
    assert arrays["C"].shape == (4 * 18, 60) and arrays["T"].shape == (30, 60)
    assert arrays["K"].shape[0] == 60 and arrays["B"].shape == (n_R, 60)
    rows = (out / f"{STEM}_combinations.csv").read_text().strip().splitlines()
    assert rows[0].split(",")[:4] == ["i", "beta_true", "beta_hat", "beta_sigma"]
    assert len(rows) == 1 + n_R


def test_example_quick_slot_arrays_are_consistent(quick):
    _, payload = quick
    slots = payload["slots"]
    for key in (
        "truth",
        "projected_truth",
        "representative",
        "unresolved_truth",
        "bias_measured",
        "noise_sigma_retained_only",
        "slot_status",
    ):
        assert len(slots[key]) == 60, key
    truth = np.asarray(slots["truth"])
    proj = np.asarray(slots["projected_truth"])
    unres = np.asarray(slots["unresolved_truth"])
    np.testing.assert_allclose(proj + unres, truth, atol=1e-13)
    comb = payload["combinations"]
    assert len(comb["beta_hat"]) == payload["fit"]["n_retained"]
    assert all(len(row) == 60 for row in comb["rows"])


def test_figure_series_have_no_full_tensor(quick):
    import sed_reconstruction_figure as figure

    _, payload = quick
    series = figure.series(payload)
    assert series
    for name, values in series.items():
        assert np.asarray(values).shape[0] != 60, name


def test_example_figure_quick(quick, tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import sed_reconstruction_figure as figure

    _, payload = quick
    fig = figure.plot(payload, tmp_path / STEM)
    assert (tmp_path / f"{STEM}.png").is_file()
    assert (tmp_path / f"{STEM}.pdf").is_file()
    assert matplotlib.rcParams["text.usetex"] is False
    labels = [ax.get_ylabel() for ax in fig.axes]
    assert r"$S/\max\, I_{\rm direct}$" in labels
    assert not any(r"\max I" in label for label in labels)  # mathtext: "maxI"
    renderer = fig.canvas.get_renderer()
    data_axes = [ax for ax in fig.axes if ax.axison]
    legends = list(fig.legends) + [
        ax.get_legend() for ax in fig.axes if ax.get_legend() is not None
    ]
    assert len(legends) >= 2
    width, height = fig.bbox.width, fig.bbox.height
    for legend in legends:
        box = legend.get_window_extent(renderer)
        assert box.x0 >= 0 and box.y0 >= 0
        assert box.x1 <= width + 1 and box.y1 <= height + 1
        for ax in data_axes:
            other = ax.get_window_extent(renderer)
            overlap_x = min(box.x1, other.x1) - max(box.x0, other.x0)
            overlap_y = min(box.y1, other.y1) - max(box.y0, other.y0)
            assert overlap_x <= 0 or overlap_y <= 0, (legend, ax.get_title())
    import matplotlib.pyplot as plt

    plt.close(fig)


def _imports(path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            yield ("." * node.level) + (node.module or "")


def test_example_is_public_api_only():
    for name in EXAMPLE_FILES:
        for module in _imports(SCRIPTS / name):
            root = module.split(".")[0]
            if root == "syncmoments":
                parts = module.split(".")
                assert not any(p.startswith("_") for p in parts), (name, module)
            elif root in SIBLINGS or root in {"validation", "synchro"}:
                assert root in SIBLINGS, (name, module)


def test_direct_oracle_uses_no_package_solver():
    for module in _imports(SCRIPTS / "sed_reconstruction_direct.py"):
        if module.startswith("syncmoments"):
            assert module == "syncmoments.constants", module


@pytest.mark.slow
def test_example_full_against_historical(tmp_path):
    payload = example.main(["--out", str(tmp_path), "--no-figure"])
    red, fit, val = payload["reduction"], payload["fit"], payload["validation"]
    # asserted exactly (design Section 9.5)
    assert (red["n_full"], red["n_active"], red["n_q"]) == (60, 52, 30)
    assert fit["n_retained"] == 9 and fit["dof"] == 63
    # numerical rank: recorded, not asserted (5.8 % margin historically)
    assert fit["claims"]["numerical"]["rank"] in range(20, 31)
    s = np.asarray(fit["singular_values"][:16])
    np.testing.assert_allclose(s, HIST_S_TOP, rtol=1e-3)
    assert math.isclose(fit["chi2"], HIST_CHI2, rel_tol=1e-2)
    res = payload["residuals"]
    assert math.isclose(res["withheld_rms_over_I"], HIST_HELDOUT_RMS, rel_tol=1e-2)
    assert math.isclose(res["withheld_max_over_I"], HIST_HELDOUT_MAX, rel_tol=1e-2)
    assert res["max_forward_discrepancy_over_I"] < 1e-2
    assert val["grouped_vs_ungrouped_max_coefficient_difference"]["value"] < 1e-12
    assert val["lossless_grouping_max_response_error"]["value"] <= 1e-12
    assert val["covariance_monte_carlo_max_relative_diagonal_error"]["value"] < 0.06
    assert val["direct_quadrature_change_over_I"]["value"] < 2e-6
    assert val["basis_refinement_change_over_I"]["value"] < 3e-6
    assert val["refinement_n_retained_unchanged"]["value"] is True
    assert val["exact_scaling_invariance_over_I"]["value"] < 1e-12
    assert val["independent_F_relative_error"]["value"] < 1e-7
    assert val["moment_routes_max_difference"]["value"] < 1e-12


def test_quick_flag_parsing():
    args = example.parse_args(["--quick", "--no-figure", "--out", os.curdir])
    assert args.quick and args.no_figure
    assert example.parse_args([]).historical is None
