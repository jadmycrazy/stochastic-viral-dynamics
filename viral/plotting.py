"""Figures drawn from a results file. Needs matplotlib (`pip install -e ".[plot]"`)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"

# One blue ramp for neutralization, an ordered quantity: darker means more antibody.
NEUTRALIZATION_COLORS = ("#86b6ef", "#3987e5", "#1c5cab", "#0d366b")
STOCHASTIC_COLOR = "#2a78d6"
DETERMINISTIC_COLOR = "#eb6834"


def _style_axes(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(True, which="major", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT_SECONDARY, length=0, labelsize=10)
    ax.xaxis.label.set_color(TEXT_SECONDARY)
    ax.yaxis.label.set_color(TEXT_SECONDARY)


def _neutralization_label(fraction: float) -> str:
    if fraction == 0:
        return "No antibody"
    return f"{fraction:.0%} neutralized"


def plot_dose_response(results: dict[str, Any], path: str | Path) -> None:
    """Establishment probability against inoculum, with Gillespie checks overlaid."""
    import matplotlib.pyplot as plt

    doses = np.array(results["dose"]["v0"])
    curves = results["dose"]["curves"]

    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=SURFACE)
    _style_axes(ax)

    color_of = {}
    for curve, color in zip(curves, NEUTRALIZATION_COLORS, strict=True):
        color_of[curve["phi"]] = color
        ax.plot(
            doses,
            curve["p"],
            color=color,
            linewidth=2,
            solid_capstyle="round",
            label=_neutralization_label(curve["phi"]),
        )

    for index, check in enumerate(results["check"]):
        ax.errorbar(
            check["v0"],
            check["ssa"],
            yerr=2 * check["se"],
            fmt="o",
            markersize=8,
            markerfacecolor=color_of[check["phi"]],
            markeredgecolor=SURFACE,
            markeredgewidth=2,
            ecolor=TEXT_SECONDARY,
            elinewidth=1,
            zorder=3,
            label="Gillespie simulation (± 2 SE)" if index == 0 else None,
        )

    ax.set_xscale("log")
    ax.set_xlim(0.9, 1000)
    ax.set_ylim(0, 1.02)
    ax.set_xticks([1, 10, 100, 1000], labels=["1", "10", "100", "1,000"])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], labels=["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Infectious virions at exposure")
    ax.set_ylabel("Probability infection establishes")
    ax.set_title(
        "More antibody means more virions are needed to start an infection",
        loc="left",
        color=TEXT,
        fontsize=12,
        pad=12,
    )
    legend = ax.legend(frameon=False, loc="lower right", fontsize=10)
    for text in legend.get_texts():
        text.set_color(TEXT)

    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_trajectories(results: dict[str, Any], path: str | Path) -> None:
    """Viral load after one virion: stochastic spread against the deterministic curve."""
    import matplotlib.pyplot as plt

    trajectories = results["traj"]
    times = np.array(trajectories["t"])
    # Below about 100 particles the infection is still in the exact simulator,
    # so the recorded load starts there.
    floor = 100.0

    def visible(values: list[float]) -> np.ndarray:
        series = np.array(values, dtype=float)
        return np.where(series >= floor, series, np.nan)

    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=SURFACE)
    _style_axes(ax)

    low = np.maximum(np.array(trajectories["q05"]), floor)
    high = np.maximum(np.array(trajectories["q95"]), floor)
    ax.fill_between(
        times,
        low,
        high,
        color=STOCHASTIC_COLOR,
        alpha=0.18,
        linewidth=0,
        label="Stochastic, 5th to 95th percentile",
    )
    ax.plot(
        times,
        visible(trajectories["q50"]),
        color=STOCHASTIC_COLOR,
        linewidth=2,
        solid_capstyle="round",
        label="Stochastic, median of established infections",
    )
    ax.plot(
        times,
        visible(trajectories["ode"]),
        color=DETERMINISTIC_COLOR,
        linewidth=2,
        solid_capstyle="round",
        label="Deterministic model",
    )

    ax.axhline(
        1000.0,
        color=TEXT_SECONDARY,
        linewidth=1,
        label="Detection threshold, 1,000 virions/mL",
    )

    ax.set_yscale("log")
    ax.set_xlim(0, 20)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_ylim(floor, 1e6)
    ax.set_xlabel("Days since exposure to one virion")
    ax.set_ylabel("Infectious virions per mL")
    ax.set_title(
        "Infections that establish are detected sooner than the deterministic model predicts",
        loc="left",
        color=TEXT,
        fontsize=12,
        pad=12,
    )
    legend = ax.legend(frameon=False, loc="upper right", fontsize=10)
    for text in legend.get_texts():
        text.set_color(TEXT)

    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def make_figures(results_path: str | Path, output_dir: str | Path) -> list[Path]:
    """Draw every figure from a results file and return the paths written."""
    import matplotlib

    matplotlib.use("Agg")

    with Path(results_path).open() as handle:
        results = json.load(handle)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dose_path = output_dir / "dose_response.png"
    trajectory_path = output_dir / "viral_load.png"
    plot_dose_response(results, dose_path)
    plot_trajectories(results, trajectory_path)
    return [dose_path, trajectory_path]
