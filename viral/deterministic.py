"""Deterministic (mean-field) version of the model, for comparison."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .parameters import DEFAULT, Parameters

FloatArray = NDArray[np.float64]


def solve_ode(
    initial_virions: float = 1.0,
    params: Parameters = DEFAULT,
    neutralization: float = 0.0,
    end_time: float = 40.0,
    time_step: float = 1e-3,
    record_every: float = 0.25,
) -> tuple[FloatArray, FloatArray]:
    """Integrate the ordinary differential equations with fourth-order Runge-Kutta.

    Returns the recording times and the viral load at each of them. With
    R0 > 1 this solution always grows, however small the inoculum, which is the
    behavior the stochastic simulations correct.
    """
    infectivity = params.effective_infection_rate(neutralization) / params.target_cells
    supply = params.target_supply_rate

    def derivative(state: FloatArray) -> FloatArray:
        target, eclipse, infected, virions = state
        new_infections = infectivity * target * virions
        return np.array(
            [
                supply - params.target_turnover_rate * target - new_infections,
                new_infections - params.eclipse_exit_rate * eclipse,
                params.eclipse_exit_rate * eclipse - params.infected_death_rate * infected,
                params.production_rate * infected
                - params.clearance_rate * virions
                - new_infections,
            ]
        )

    state = np.array([params.target_cells, 0.0, 0.0, float(initial_virions)])
    times = np.arange(0, end_time + 1e-9, record_every)
    viral_load: list[float] = []
    next_record = 0

    for step in range(int(round(end_time / time_step)) + 1):
        now = step * time_step
        while next_record < len(times) and times[next_record] <= now + 1e-12:
            viral_load.append(float(state[3]))
            next_record += 1

        slope_1 = derivative(state)
        slope_2 = derivative(state + time_step / 2 * slope_1)
        slope_3 = derivative(state + time_step / 2 * slope_2)
        slope_4 = derivative(state + time_step * slope_3)
        state = state + time_step / 6 * (slope_1 + 2 * slope_2 + 2 * slope_3 + slope_4)

    return times, np.array(viral_load)
