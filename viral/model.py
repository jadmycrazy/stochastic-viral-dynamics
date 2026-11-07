"""
Early within-host HIV dynamics: target cells T, eclipse-phase infected cells E,
productively infected cells I, and infectious virions V.

Reactions (propensity, state change):
  infection     beta*T*V   T-1, E+1, V-1
  eclipse ends  k*E        E-1, I+1
  cell death    delta*I    I-1
  production    p*I        V+1
  clearance     c*V        V-1
  T supply      lam        T+1
  T death       dT*T       T-1

Clearance, death and eclipse rates are standard HIV estimates (c = 23/day,
Ramratnam et al. 1999; delta = 1/day and eclipse ~1 day, Markowitz et al. 2003).
Burst size and infectivity are set so R0 = 8 (Ribeiro et al. 2010).
Neutralizing antibody blocks a fraction phi of infection events.
"""
import math
import numpy as np

PARAMS = dict(
    c=23.0,        # virion clearance, 1/day
    delta=1.0,     # infected-cell death, 1/day
    k=1.0,         # eclipse exit, 1/day
    p=48.0,        # infectious virions produced per cell per day
    b=4.6,         # beta*T0: per-virion infection rate at baseline target cells, 1/day
    T0=1e5,        # activated target cells per mL
    dT=0.01,       # target-cell turnover, 1/day
)

def r0(P=PARAMS, phi=0.0):
    b = P["b"] * (1 - phi)
    return (P["p"] / P["delta"]) * b / (P["c"] + b)

def extinction_prob(P=PARAMS, phi=0.0):
    """Probability that the lineage of one virion dies out (multi-type branching process).
    A virion is cleared with prob c/(c+b) or infects with prob b/(c+b). An infected cell
    survives the eclipse, then releases a geometric number of virions with mean p/delta."""
    b = P["b"] * (1 - phi); c, p, d = P["c"], P["p"], P["delta"]
    G = lambda s: d / (d + p * (1 - s))
    q = 0.0
    for _ in range(100000):
        qn = c / (c + b) + b / (c + b) * G(q)
        if abs(qn - q) < 1e-15: break
        q = qn
    return q

def establishment_prob(v0, P=PARAMS, phi=0.0):
    return 1 - extinction_prob(P, phi) ** v0

def ssa_early(v0, rng, P=PARAMS, phi=0.0, threshold=200, tmax=60.0):
    """Exact Gillespie simulation with target cells held at T0 (they barely change early).
    Returns (established, time, E, I, V) when the population dies out or reaches threshold."""
    b = P["b"] * (1 - phi); c, p, d, k = P["c"], P["p"], P["delta"], P["k"]
    E = I = 0; V = v0; t = 0.0
    while True:
        a1, a2, a3, a4, a5 = b * V, k * E, d * I, p * I, c * V
        a0 = a1 + a2 + a3 + a4 + a5
        if a0 == 0: return False, t, 0, 0, 0
        t += rng.exponential(1 / a0)
        if t > tmax: return False, t, E, I, V
        u = rng.random() * a0
        if u < a1: V -= 1; E += 1
        elif u < a1 + a2: E -= 1; I += 1
        elif u < a1 + a2 + a3: I -= 1
        elif u < a1 + a2 + a3 + a4: V += 1
        else: V -= 1
        if E + I + V >= threshold: return True, t, E, I, V

def cle(states, t_start, P=PARAMS, phi=0.0, t_end=40.0, dt=1e-3, rng=None, record_every=0.25):
    """Chemical Langevin equation (Euler-Maruyama), vectorized over trajectories.
    states: array (n, 4) of T, E, I, V; t_start: per-trajectory start times."""
    rng = rng or np.random.default_rng()
    beta = P["b"] * (1 - phi) / P["T0"]; lam = P["dT"] * P["T0"]
    X = np.array(states, float); n = len(X); t0 = np.asarray(t_start, float)
    grid = np.arange(0, t_end + 1e-9, record_every)
    out = np.full((n, len(grid)), np.nan)
    steps = int(round(t_end / dt)); gi = 0; sq = math.sqrt(dt)
    for s in range(steps + 1):
        t = s * dt
        while gi < len(grid) and grid[gi] <= t + 1e-12:
            out[:, gi] = np.where(t0 <= grid[gi], X[:, 3], np.nan); gi += 1
        if s == steps: break
        live = t0 <= t
        T, E, I, V = X.T
        a = np.stack([beta * T * V, P["k"] * E, P["delta"] * I, P["p"] * I, P["c"] * V,
                      np.full(n, lam), P["dT"] * T])
        W = rng.standard_normal(a.shape) * sq
        inc = a * dt + np.sqrt(a) * W
        dX = np.stack([inc[5] - inc[6] - inc[0], inc[0] - inc[1], inc[1] - inc[2], inc[3] - inc[4] - inc[0]], 1)
        X = np.where(live[:, None], np.maximum(X + dX, 0), X)
    return grid, out

def ode(v0=1.0, P=PARAMS, phi=0.0, t_end=40.0, dt=1e-3, record_every=0.25):
    beta = P["b"] * (1 - phi) / P["T0"]; lam = P["dT"] * P["T0"]
    T, E, I, V = P["T0"], 0.0, 0.0, float(v0)
    grid = np.arange(0, t_end + 1e-9, record_every); out = []; gi = 0
    for s in range(int(round(t_end / dt)) + 1):
        t = s * dt
        while gi < len(grid) and grid[gi] <= t + 1e-12: out.append(V); gi += 1
        def f(T, E, I, V):
            inf = beta * T * V
            return lam - P["dT"] * T - inf, inf - P["k"] * E, P["k"] * E - P["delta"] * I, P["p"] * I - P["c"] * V - inf
        k1 = f(T, E, I, V); k2 = f(*(x + dt / 2 * y for x, y in zip((T, E, I, V), k1)))
        k3 = f(*(x + dt / 2 * y for x, y in zip((T, E, I, V), k2))); k4 = f(*(x + dt * y for x, y in zip((T, E, I, V), k3)))
        T, E, I, V = (x + dt / 6 * (a + 2 * b2 + 2 * c2 + d2) for x, a, b2, c2, d2 in zip((T, E, I, V), k1, k2, k3, k4))
    return grid, np.array(out)
