"""Exact results from branching-process theory.

While target cells are plentiful, each virion and each infected cell acts
independently, so the early infection is a multi-type branching process. That
gives the probability of establishment in closed form, which the stochastic
simulations are checked against.
"""

from __future__ import annotations

from .parameters import DEFAULT, Parameters


def basic_reproduction_number(params: Parameters = DEFAULT, neutralization: float = 0.0) -> float:
    """Expected number of newly infected cells caused by one infected cell (R0).

    An infected cell releases p / delta virions on average, and each virion
    infects a cell with probability b / (c + b) before it is cleared.
    """
    infection = params.effective_infection_rate(neutralization)
    burst_size = params.production_rate / params.infected_death_rate
    return burst_size * infection / (params.clearance_rate + infection)


def extinction_probability(
    params: Parameters = DEFAULT,
    neutralization: float = 0.0,
    tolerance: float = 1e-15,
    max_iterations: int = 100_000,
) -> float:
    """Probability that the lineage started by one virion dies out.

    A virion is cleared with probability c / (c + b) or infects a cell with
    probability b / (c + b). An infected cell always leaves the eclipse phase,
    then releases virions at rate p until it dies at rate delta, so the number
    released is geometric with mean p / delta. Its generating function is
    G(s) = delta / (delta + p * (1 - s)).

    The extinction probability q is the smallest solution of
    q = c / (c + b) + b / (c + b) * G(q). Iterating from q = 0 converges to it.
    """
    infection = params.effective_infection_rate(neutralization)
    clearance = params.clearance_rate
    production = params.production_rate
    death = params.infected_death_rate

    def offspring_all_extinct(virion_extinction: float) -> float:
        return death / (death + production * (1 - virion_extinction))

    extinction = 0.0
    for _ in range(max_iterations):
        updated = clearance / (clearance + infection) + infection / (
            clearance + infection
        ) * offspring_all_extinct(extinction)
        if abs(updated - extinction) < tolerance:
            break
        extinction = updated
    return extinction


def establishment_probability(
    initial_virions: int, params: Parameters = DEFAULT, neutralization: float = 0.0
) -> float:
    """Probability that at least one of the initial virions founds an infection."""
    if initial_virions < 0:
        raise ValueError(f"initial_virions must not be negative, got {initial_virions}")
    return 1 - extinction_probability(params, neutralization) ** initial_virions


def critical_neutralization(params: Parameters = DEFAULT) -> float:
    """Neutralized fraction at which R0 falls to exactly 1.

    Above this fraction every infection attempt dies out.
    """
    burst_size = params.production_rate / params.infected_death_rate
    return 1 - (params.clearance_rate / params.infection_rate) / (burst_size - 1)
