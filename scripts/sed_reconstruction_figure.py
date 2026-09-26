"""Figure of the SED reconstruction example: IQU fit, fitted combinations, two steps.

``plot(payload, out_stem)`` draws from the example's JSON payload only and
writes ``out_stem.png`` (200 dpi) and ``out_stem.pdf``. Panels (a-c): direct,
forward-moment and reduced-model I, Q, U with residuals in percent of I
(withheld centres as open markers). Panel (d): the retained combinations
``beta_i``, true against fitted with ``1/s_i`` bars. Panel (e): the exact
60 -> 30 grouping (no data) above the noise-dependent selection of singular
values. No 60-entry representative is plotted (``series``). Matplotlib
mathtext only (``text.usetex`` off); Matplotlib is imported inside ``plot``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

COLOURS = ("#235789", "#c55b32", "#638449")
NAMES = ("(a) Intensity $I$", "(b) Stokes $Q$", "(c) Stokes $U$")


def series(payload):
    """Every array the figure plots, keyed by name (none has the full-tensor length)."""
    sed, comb, fit = payload["sed"], payload["combinations"], payload["fit"]
    out = {
        "frequency_ghz": np.asarray(sed["frequency_hz"]) / 1e9,
        "fit_channel_indices": np.asarray(sed["fit_channel_indices"]),
        "withheld": np.asarray(payload["residuals"]["withheld"]),
        "singular_values": np.asarray(fit["singular_values"]),
        "beta_true": np.asarray(comb["beta_true"]),
        "beta_hat": np.asarray(comb["beta_hat"]),
        "beta_sigma": np.asarray(comb["beta_sigma"]),
    }
    for key in ("direct", "forward", "stokes_hat", "stokes_sigma", "data_fitted"):
        out[key] = np.asarray(sed[key])
    return out


def _sed_top(ax, s, c):
    """Direct, forward and fitted component ``c`` normalised by ``max I_direct``."""
    from matplotlib.ticker import NullFormatter

    scale = s["direct"][:, 0].max()
    nu, hat, sig = s["frequency_ghz"], s["stokes_hat"][:, c], s["stokes_sigma"][:, c]
    ax.plot(nu, s["direct"][:, c] / scale, color="black", lw=1.5)
    ax.plot(nu, s["forward"][:, c] / scale, color="#a0a0a0", lw=1.2, ls=":")
    ax.plot(nu, hat / scale, color=COLOURS[c], ls="--", lw=1.3)
    band = ((hat - sig) / scale, (hat + sig) / scale)
    ax.fill_between(nu, *band, color=COLOURS[c], alpha=0.18, linewidth=0)
    fit_nu = nu[s["fit_channel_indices"]]
    data = s["data_fitted"][:, c] / scale
    ax.plot(fit_nu, data, ".", ms=2.5, color=COLOURS[c], alpha=0.7)
    ax.set_xscale("log")
    ax.set_xticks([0.4, 1, 3], ["0.4", "1", "3"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(labelbottom=False)
    ax.set_title(NAMES[c], loc="left")
    if c == 0:
        ax.set_ylabel(r"$S/\max\, I_{\rm direct}$")


def _sed_residual(ax, s, c):
    """Fit and forward residuals in percent of ``I_direct``; withheld centres open."""
    nu, denom = s["frequency_ghz"], s["direct"][:, 0]
    fit_res = 100 * (s["stokes_hat"][:, c] - s["direct"][:, c]) / denom
    noise = 100 * s["stokes_sigma"][:, c] / denom
    forward = 100 * (s["forward"][:, c] - s["direct"][:, c]) / denom
    withheld = s["withheld"]
    ax.axhline(0, color=".6", lw=0.6)
    band = (fit_res - noise, fit_res + noise)
    ax.fill_between(nu, *band, color=COLOURS[c], alpha=0.18, linewidth=0)
    ax.plot(nu, forward, color=".4", ls=":", lw=1.2)
    ax.plot(nu, fit_res, color=COLOURS[c], ls="--", lw=1.2)
    ax.plot(
        nu[withheld],
        fit_res[withheld],
        "o",
        mfc="white",
        mec=COLOURS[c],
        ms=2.3,
        mew=0.6,
    )
    ax.set(ylim=(-3, 3), yticks=[-2, 0, 2], xlabel=r"$\nu$ [GHz]")
    if c == 0:
        ax.set_ylabel("residual [% of $I$]")
    ax.grid(alpha=0.12)


def _symlog_limits(ax, s):
    """Pad the symlog ``beta`` axis in display space (data margins vanish in symlog)."""
    ends = np.concatenate(
        (
            s["beta_true"],
            s["beta_hat"] - s["beta_sigma"],
            s["beta_hat"] + s["beta_sigma"],
        )
    )
    transform = ax.yaxis.get_transform()
    lo, hi = transform.transform([ends.min(), ends.max()])
    margin = 0.14 * (hi - lo)
    ax.set_ylim(transform.inverted().transform([lo - margin, hi + margin]))


def _cluster_brackets(ax, clusters):
    """Bracket the combinations that share a singular value (subspace only)."""
    label = "equal $s_i$: subspace only"
    for cid in sorted(set(clusters) - {-1}):
        members = [i + 1 for i, c in enumerate(clusters) if c == cid]
        lo, hi = min(members) - 0.3, max(members) + 0.3
        ax.plot(
            [lo, lo, hi, hi],
            [0.09, 0.04, 0.04, 0.09],
            color=".45",
            lw=0.8,
            transform=ax.get_xaxis_transform(),
            clip_on=False,
            label=label,
        )
        label = None


def _beta_panel(fig, cell, s, clusters):
    ax = fig.add_subplot(cell)
    n = s["beta_hat"].size
    x = np.arange(1, n + 1)
    ax.plot(x - 0.08, s["beta_true"], "x", color="black", ms=4, label=r"true $\beta_i$")
    ax.errorbar(
        x + 0.08,
        s["beta_hat"],
        yerr=s["beta_sigma"],
        fmt="o",
        ms=3,
        capsize=2,
        color=COLOURS[0],
        lw=0.8,
        label=r"fitted $\pm 1/s_i$",
    )
    ax.axhline(0, color=".6", lw=0.6)
    ax.set_yscale("symlog", linthresh=0.02)
    _symlog_limits(ax, s)
    ax.set_yticks([-1, -0.1, 0, 0.1, 1])
    ax.set(xticks=x, xlim=(0.4, n + 0.6), xlabel="retained combination $i$")
    ax.set_ylabel(r"$\beta_i$")
    ax.set_title(f"(d) {n} fitted combinations", loc="left")
    _cluster_brackets(ax, clusters)
    ax.legend(
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.30),
        handlelength=1.2,
        columnspacing=0.8,
        borderpad=0.1,
    )


def _flow_panel(fig, cell, red):
    """Step 1: the exact grouping ``a -> q = T a`` (counts from the reduction)."""
    flow = fig.add_subplot(cell)
    flow.set_axis_off()
    flow.set_title("(e) Two-step reduction", loc="left")
    flow.text(0.12, 0.66, f"$a$\n{red['n_full']} slots", ha="center", va="center")
    flow.text(0.86, 0.66, f"$q = Ta$\n{red['n_q']} groups", ha="center", va="center")
    arrow = {"arrowstyle": "->", "color": ".35", "lw": 1}
    flow.annotate("", xy=(0.66, 0.66), xytext=(0.30, 0.66), arrowprops=arrow)
    flow.text(
        0.5,
        0.0,
        "step 1: exact grouping, no data",
        ha="center",
        va="bottom",
        color=".3",
        fontsize="small",
    )


def _singular_panel(fig, cell, fit, sv):
    """Step 2: singular values, the noise cut ``1/max_sigma`` and ``rank_tol s_max``."""
    ax = fig.add_subplot(cell)
    n_R, cutoff = fit["n_retained"], 1 / fit["max_sigma"]
    idx = np.arange(1, sv.size + 1)
    ax.semilogy(idx, np.maximum(sv, 1e-13), ".", color=".55", ms=3)
    ax.semilogy(idx[:n_R], sv[:n_R], "o", color=COLOURS[0], ms=3)
    ax.axhline(cutoff, color=COLOURS[0], ls="--", lw=0.8)
    ax.axhline(fit["rank_tol"] * sv[0], color=".45", ls=":", lw=0.8)
    ax.text(
        0.97,
        0.95,
        f"step 2: noise cut\n{n_R} kept, $s_i \\geq {cutoff:g}$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=COLOURS[0],
        fontsize="small",
    )
    ax.text(
        0.03,
        0.04,
        "numerical rank tolerance",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color=".35",
        fontsize="small",
    )
    ax.set_xlabel(f"grouped mode index ({sv.size})")
    ax.set_ylabel(r"$s_i$")
    ax.set_ylim(min(1e-9, 0.3 * fit["rank_tol"] * sv[0]), sv[0] * 3e4)  # label room
    ax.set_yticks([1e-8, 1e-4, 1, 1e4])


def _top_legend(fig):
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], color="black", label="direct population"),
        Line2D([], [], color="#888888", ls=":", label="forward moments"),
        Line2D(
            [],
            [],
            color=COLOURS[0],
            ls="--",
            label=r"reduced-model fit, $1\sigma$ noise",
        ),
        Line2D([], [], color=COLOURS[0], marker=".", ls="none", label="fitted data"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.995),
        ncol=4,
        frameon=False,
        handlelength=1.3,
        columnspacing=0.8,
        handletextpad=0.4,
        fontsize="small",
    )


def plot(payload, out_stem):
    """Draw the figure from ``payload`` and save ``out_stem.{png,pdf}``; returns the figure."""
    import matplotlib

    matplotlib.rcParams["text.usetex"] = False
    import matplotlib.pyplot as plt

    s = series(payload)
    fig = plt.figure(figsize=(7.0, 6.6))
    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.4, 1],
        left=0.10,
        right=0.975,
        bottom=0.15,
        top=0.90,
        hspace=0.50,
        wspace=0.37,
    )
    for c in range(3):
        sub = grid[0, c].subgridspec(2, 1, height_ratios=[1, 0.75], hspace=0.08)
        top = fig.add_subplot(sub[0])
        _sed_top(top, s, c)
        _sed_residual(fig.add_subplot(sub[1], sharex=top), s, c)
    _top_legend(fig)
    n_withheld = int(np.sum(s["withheld"]))
    fig.text(
        0.54,
        0.462,
        f"residuals: open circles mark the {n_withheld} withheld centres",
        ha="center",
    )
    lower = grid[1, :].subgridspec(1, 2, width_ratios=[1.65, 1], wspace=0.34)
    _beta_panel(fig, lower[0, 0], s, payload["combinations"]["cluster"])
    right = lower[0, 1].subgridspec(2, 1, height_ratios=[0.47, 1], hspace=0.42)
    _flow_panel(fig, right[0], payload["reduction"])
    _singular_panel(fig, right[1], payload["fit"], s["singular_values"])
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".png"), dpi=200)
    fig.savefig(out_stem.with_suffix(".pdf"))
    return fig
