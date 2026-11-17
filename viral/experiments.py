"""The numerical experiments behind the results in the README.

`run_all` returns one dictionary that is written to `results/viral.json`. The
plotting code and the project web page both read that file.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .deterministic import solve_ode
from .gillespie import simulate_early_infection
from .langevin import simulate_langevin
from .parameters import DEFAULT, Parameters
from .theory import (
    basic_reproduction_number,
    critical_neutralization,
    establishment_probability,
)

DETECTION_THRESHOLD = 1000.0
"""Viral load (infectious virions/mL) taken as the point of detection."""

VALIDATION_NEUTRALIZATION = (0.0, 0.5, 0.8)
VALIDATION_RUNS = ((1, 20_000), (10, 4_000))
"""Pairs of (initial virions, number of Gillespie runs)."""

DOSE_NEUTRALIZATION = (0.0, 0.5, 0.8, 0.85)


def validate_against_theory(
    rng: np.random.Generator, params: Parameters = DEFAULT, verbose: bool = False
) -> list[dict[str, float]]:
    """Compare Gillespie establishment frequencies with the exact probability."""
    rows = []
    for neutralization in VALIDATION_NEUTRALIZATION:
        for initial_virions, runs in VALIDATION_RUNS:
            established = sum(
                simulate_early_infection(initial_virions, rng, params, neutralization).established
                for _ in range(runs)
            )
            frequency = established / runs
            standard_error = (frequency * (1 - frequency) / runs) ** 0.5
            row = {
                "phi": neutralization,
                "v0": initial_virions,
                "runs": runs,
                "ssa": frequency,
                "se": standard_error,
                "theory": establishment_probability(initial_virions, params, neutralization),
            }
            rows.append(row)
            if verbose:
                print(row, flush=True)
    return rows


def dose_response(params: Parameters = DEFAULT) -> dict[str, Any]:
    """Establishment probability against inoculum size, by neutralization."""
    doses = [int(round(x)) for x in np.unique(np.round(np.logspace(0, 3, 40)))]
    curves = [
        {
            "phi": neutralization,
            "r0": basic_reproduction_number(params, neutralization),
            "p": [establishment_probability(dose, params, neutralization) for dose in doses],
        }
        for neutralization in DOSE_NEUTRALIZATION
    ]
    return {"v0": doses, "curves": curves}


def neutralization_sweep(params: Parameters = DEFAULT) -> dict[str, Any]:
    """Per-virion establishment probability and R0 against neutralization."""
    fractions = [round(float(x), 3) for x in np.linspace(0, 0.95, 96)]
    return {
        "phi": fractions,
        "p1": [establishment_probability(1, params, f) for f in fractions],
        "r0": [basic_reproduction_number(params, f) for f in fractions],
        "threshold": critical_neutralization(params),
    }


def hybrid_trajectories(
    rng: np.random.Generator,
    params: Parameters = DEFAULT,
    n_established: int = 600,
    end_time: float = 30.0,
    verbose: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Viral load from a single virion, conditioned on establishment.

    Each exposure starts in the exact Gillespie simulator. Runs that establish
    are handed to the Langevin integrator at the state and time they reached
    the threshold. Returns the trajectory summary and the detection-time summary.
    """
    handoff_states: list[list[float]] = []
    handoff_times: list[float] = []
    attempts = 0
    while len(handoff_states) < n_established:
        attempts += 1
        outcome = simulate_early_infection(1, rng, params)
        if outcome.established:
            handoff_states.append(
                [
                    params.target_cells,
                    outcome.eclipse_cells,
                    outcome.infected_cells,
                    outcome.virions,
                ]
            )
            handoff_times.append(outcome.time)
    if verbose:
        print("established", len(handoff_states), "of", attempts, flush=True)

    times, viral_load = simulate_langevin(
        handoff_states, handoff_times, params, end_time=end_time, time_step=2e-3, rng=rng
    )
    ode_times, ode_load = solve_ode(1.0, params, end_time=end_time)

    detection_times = []
    for trajectory in viral_load:
        above = np.where(trajectory >= DETECTION_THRESHOLD)[0]
        if len(above):
            detection_times.append(float(times[above[0]]))
    detected = np.array(detection_times)

    # Before handoff the load is below the threshold of 200 particles, so it
    # counts as zero when taking percentiles across trajectories.
    load_or_zero = np.where(np.isnan(viral_load), 0, viral_load)

    def percentile(level: float) -> list[float]:
        return np.nanpercentile(load_or_zero, level, axis=0).tolist()

    trajectories = {
        "t": times.tolist(),
        "ode": ode_load.tolist(),
        "q05": percentile(5),
        "q25": percentile(25),
        "q50": percentile(50),
        "q75": percentile(75),
        "q95": percentile(95),
        "examples": [
            [None if np.isnan(x) else round(float(x), 2) for x in viral_load[i]] for i in range(12)
        ],
        "handoff": [float(x) for x in handoff_times[:12]],
    }
    detection = {
        "times": detected.round(3).tolist(),
        "median": float(np.median(detected)),
        "p05": float(np.percentile(detected, 5)),
        "p95": float(np.percentile(detected, 95)),
        "ode": float(ode_times[np.argmax(ode_load >= DETECTION_THRESHOLD)]),
        "n": len(detected),
        "tries": attempts,
    }
    return trajectories, detection


def run_all(
    seed: int = 2026, params: Parameters = DEFAULT, verbose: bool = False
) -> dict[str, Any]:
    """Run every experiment with one seeded random generator."""
    rng = np.random.default_rng(seed)
    started = time.time()

    results: dict[str, Any] = {
        "params": params.as_dict(),
        "r0": basic_reproduction_number(params),
    }
    results["check"] = validate_against_theory(rng, params, verbose)
    results["dose"] = dose_response(params)
    results["neut"] = neutralization_sweep(params)
    results["traj"], results["detect"] = hybrid_trajectories(rng, params, verbose=verbose)

    if verbose:
        summary = {k: v for k, v in results["detect"].items() if k != "times"}
        print("detect", summary, f"{time.time() - started:.0f}s", flush=True)
    return results


def write_results(results: dict[str, Any], path: str | Path) -> None:
    """Write results as compact JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(results, handle, separators=(",", ":"))
