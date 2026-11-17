"""Chemical Langevin equation for the growth phase, once counts are large."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .parameters import DEFAULT, Parameters

FloatArray = NDArray[np.float64]


def simulate_langevin(
    initial_states: ArrayLike,
    start_times: ArrayLike,
    params: Parameters = DEFAULT,
    neutralization: float = 0.0,
    end_time: float = 40.0,
    time_step: float = 1e-3,
    rng: np.random.Generator | None = None,
    record_every: float = 0.25,
) -> tuple[FloatArray, FloatArray]:
    """Integrate many trajectories of the chemical Langevin equation at once.

    Each reaction with propensity a contributes a * dt + sqrt(a * dt) * N(0, 1)
    firings per step (Euler-Maruyama). This approximates the exact jump process
    well once every reaction fires many times per step, which is why the
    Gillespie simulator handles the early phase and hands over to this one.

    States are clamped at zero after each step. The Gaussian noise can push a
    small population below zero, and a negative count would make the square
    root of a propensity undefined on the next step.

    Parameters
    ----------
    initial_states:
        Array of shape (n, 4) holding target cells, eclipse cells, productive
        cells and virions for each trajectory.
    start_times:
        Time at which each trajectory begins (days). A trajectory is frozen
        until then, so trajectories handed over at different times share a clock.

    Returns
    -------
    times:
        Recording times, every `record_every` days from 0 to `end_time`.
    viral_load:
        Array of shape (n, len(times)). Entries before a trajectory's start
        time are NaN.
    """
    rng = rng or np.random.default_rng()
    infectivity = params.effective_infection_rate(neutralization) / params.target_cells
    supply = params.target_supply_rate

    state = np.array(initial_states, dtype=float)
    n_trajectories = len(state)
    start = np.asarray(start_times, dtype=float)

    times = np.arange(0, end_time + 1e-9, record_every)
    viral_load = np.full((n_trajectories, len(times)), np.nan)

    n_steps = int(round(end_time / time_step))
    noise_scale = math.sqrt(time_step)
    next_record = 0

    for step in range(n_steps + 1):
        now = step * time_step
        while next_record < len(times) and times[next_record] <= now + 1e-12:
            has_started = start <= times[next_record]
            viral_load[:, next_record] = np.where(has_started, state[:, 3], np.nan)
            next_record += 1
        if step == n_steps:
            break

        target, eclipse, infected, virions = state.T
        propensities = np.stack(
            [
                infectivity * target * virions,  # 0: infection
                params.eclipse_exit_rate * eclipse,  # 1: eclipse ends
                params.infected_death_rate * infected,  # 2: infected-cell death
                params.production_rate * infected,  # 3: virion production
                params.clearance_rate * virions,  # 4: virion clearance
                np.full(n_trajectories, supply),  # 5: target-cell supply
                params.target_turnover_rate * target,  # 6: target-cell death
            ]
        )
        noise = rng.standard_normal(propensities.shape) * noise_scale
        firings = propensities * time_step + np.sqrt(propensities) * noise

        change = np.stack(
            [
                firings[5] - firings[6] - firings[0],  # target cells
                firings[0] - firings[1],  # eclipse cells
                firings[1] - firings[2],  # productive cells
                firings[3] - firings[4] - firings[0],  # virions
            ],
            axis=1,
        )
        active = start <= now
        state = np.where(active[:, None], np.maximum(state + change, 0), state)

    return times, viral_load
