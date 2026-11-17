"""Tests of the model against results that are known exactly."""

import numpy as np
import pytest

from viral import (
    DEFAULT,
    Parameters,
    basic_reproduction_number,
    critical_neutralization,
    establishment_probability,
    extinction_probability,
    simulate_early_infection,
    simulate_langevin,
    solve_ode,
)


def test_r0_matches_target():
    assert basic_reproduction_number() == pytest.approx(8.0)


def test_extinction_probability_solves_its_fixed_point_equation():
    q = extinction_probability()
    c = DEFAULT.clearance_rate
    b = DEFAULT.infection_rate
    burst_generating_function = DEFAULT.infected_death_rate / (
        DEFAULT.infected_death_rate + DEFAULT.production_rate * (1 - q)
    )
    assert q == pytest.approx(c / (c + b) + b / (c + b) * burst_generating_function)


def test_more_virions_establish_more_often():
    probabilities = [establishment_probability(n) for n in (1, 2, 10, 100)]
    assert probabilities == sorted(probabilities)
    assert establishment_probability(0) == 0.0


def test_neutralization_lowers_establishment():
    probabilities = [establishment_probability(1, neutralization=f) for f in (0.0, 0.5, 0.8, 0.85)]
    assert probabilities == sorted(probabilities, reverse=True)


def test_no_establishment_below_threshold():
    assert basic_reproduction_number(neutralization=0.95) < 1
    assert establishment_probability(1, neutralization=0.95) < 1e-9


def test_r0_is_one_at_critical_neutralization():
    assert basic_reproduction_number(neutralization=critical_neutralization()) == pytest.approx(1.0)


def test_gillespie_matches_branching_process():
    rng = np.random.default_rng(1)
    runs = 6000
    established = sum(simulate_early_infection(1, rng).established for _ in range(runs))
    frequency = established / runs
    standard_error = (frequency * (1 - frequency) / runs) ** 0.5
    assert abs(frequency - establishment_probability(1)) < 4 * standard_error


def test_gillespie_with_no_virions_goes_extinct_at_once():
    outcome = simulate_early_infection(0, np.random.default_rng(0))
    assert not outcome.established
    assert outcome.time == 0.0


def test_langevin_stays_nonnegative_and_grows():
    rng = np.random.default_rng(2)
    start = [[DEFAULT.target_cells, 20, 20, 160]] * 8
    _, viral_load = simulate_langevin(start, [0.0] * 8, end_time=6.0, time_step=2e-3, rng=rng)
    assert np.nanmin(viral_load) >= 0
    assert np.all(viral_load[:, -1] > viral_load[:, 0])


def test_langevin_holds_trajectories_until_their_start_time():
    rng = np.random.default_rng(3)
    start = [[DEFAULT.target_cells, 20, 20, 160]] * 2
    times, viral_load = simulate_langevin(start, [0.0, 1.0], end_time=2.0, time_step=2e-3, rng=rng)
    late = viral_load[1]
    assert np.all(np.isnan(late[times < 1.0]))
    assert late[times == 1.0][0] == 160


def test_ode_always_grows_above_threshold():
    times, viral_load = solve_ode(1.0, end_time=10.0)
    assert len(times) == len(viral_load)
    assert viral_load.max() > 1000


def test_ode_decays_when_neutralization_is_above_critical():
    _, viral_load = solve_ode(100.0, neutralization=0.95, end_time=5.0)
    assert viral_load[-1] < viral_load[0]


def test_parameters_reject_bad_values():
    with pytest.raises(ValueError):
        Parameters(clearance_rate=0.0)
    with pytest.raises(ValueError):
        DEFAULT.effective_infection_rate(1.5)
    with pytest.raises(ValueError):
        establishment_probability(-1)
