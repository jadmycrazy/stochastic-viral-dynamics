import numpy as np
from viral.model import PARAMS, r0, extinction_prob, establishment_prob, ssa_early, cle

def test_r0_matches_target():
    assert abs(r0() - 8.0) < 1e-9

def test_no_establishment_below_threshold():
    assert establishment_prob(1, phi=0.95) < 1e-9 and r0(phi=0.95) < 1

def test_gillespie_matches_branching_process():
    rng = np.random.default_rng(1)
    runs = 6000
    est = sum(ssa_early(1, rng)[0] for _ in range(runs)) / runs
    se = (est * (1 - est) / runs) ** 0.5
    assert abs(est - establishment_prob(1)) < 4 * se

def test_cle_stays_nonnegative_and_grows():
    rng = np.random.default_rng(2)
    grid, V = cle([[PARAMS["T0"], 20, 20, 160]] * 8, [0.0] * 8, t_end=6.0, dt=2e-3, rng=rng)
    assert np.nanmin(V) >= 0 and np.all(V[:, -1] > V[:, 0])
