"""Exact stochastic simulation of the first hours and days of infection."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .parameters import DEFAULT, Parameters


class EarlyOutcome(NamedTuple):
    """State of one simulated exposure when the simulation stopped."""

    established: bool
    """True if the infected population reached the establishment threshold."""

    time: float
    """Time since exposure when the simulation stopped (days)."""

    eclipse_cells: int
    infected_cells: int
    virions: int


def simulate_early_infection(
    initial_virions: int,
    rng: np.random.Generator,
    params: Parameters = DEFAULT,
    neutralization: float = 0.0,
    threshold: int = 200,
    max_time: float = 60.0,
) -> EarlyOutcome:
    """Run Gillespie's direct method until the infection dies out or takes off.

    Target cells are held at their baseline value, because a few hundred
    infected cells cannot deplete 1e5 target cells per mL. That makes every
    reaction first order, which is the same assumption as the branching-process
    theory in `viral.theory`.

    The run stops when no virus or infected cells remain (extinction), when
    eclipse cells + productive cells + virions reach `threshold`
    (establishment), or after `max_time` days.
    """
    if initial_virions < 0:
        raise ValueError(f"initial_virions must not be negative, got {initial_virions}")

    infection = params.effective_infection_rate(neutralization)
    clearance = params.clearance_rate
    production = params.production_rate
    death = params.infected_death_rate
    eclipse_exit = params.eclipse_exit_rate

    eclipse = 0
    infected = 0
    virions = initial_virions
    time = 0.0

    while True:
        rate_infection = infection * virions
        rate_eclipse_exit = eclipse_exit * eclipse
        rate_death = death * infected
        rate_production = production * infected
        rate_clearance = clearance * virions
        total_rate = (
            rate_infection + rate_eclipse_exit + rate_death + rate_production + rate_clearance
        )
        if total_rate == 0:
            return EarlyOutcome(False, time, 0, 0, 0)

        time += rng.exponential(1 / total_rate)
        if time > max_time:
            return EarlyOutcome(False, time, eclipse, infected, virions)

        # Pick which reaction fired, with probability proportional to its rate.
        draw = rng.random() * total_rate
        if draw < rate_infection:
            virions -= 1
            eclipse += 1
        elif draw < rate_infection + rate_eclipse_exit:
            eclipse -= 1
            infected += 1
        elif draw < rate_infection + rate_eclipse_exit + rate_death:
            infected -= 1
        elif draw < rate_infection + rate_eclipse_exit + rate_death + rate_production:
            virions += 1
        else:
            virions -= 1

        if eclipse + infected + virions >= threshold:
            return EarlyOutcome(True, time, eclipse, infected, virions)
