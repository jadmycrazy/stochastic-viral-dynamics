"""Run all experiments and export figure data.  Usage: python run_experiments.py results/viral.json"""
import json, sys, time
import numpy as np
from viral.model import PARAMS, r0, extinction_prob, establishment_prob, ssa_early, cle, ode

def main(out):
    rng = np.random.default_rng(2026); t0 = time.time()
    res = {"params": PARAMS, "r0": r0()}
    # 1. analytic establishment vs Gillespie, single virion and 10 virions
    check = []
    for phi in (0.0, 0.5, 0.8):
        for v0, runs in ((1, 20000), (10, 4000)):
            est = sum(ssa_early(v0, rng, phi=phi)[0] for _ in range(runs))
            p = est / runs; se = (p * (1 - p) / runs) ** 0.5
            check.append({"phi": phi, "v0": v0, "runs": runs, "ssa": p, "se": se, "theory": establishment_prob(v0, phi=phi)})
            print(check[-1], f"{time.time()-t0:.0f}s", flush=True)
    res["check"] = check
    # 2. establishment probability vs inoculum, by neutralization
    v0s = [int(round(x)) for x in np.unique(np.round(np.logspace(0, 3, 40)))]
    res["dose"] = {"v0": v0s, "curves": [{"phi": phi, "r0": r0(phi=phi), "p": [establishment_prob(v, phi=phi) for v in v0s]} for phi in (0.0, 0.5, 0.8, 0.85)]}
    # 3. per-virion establishment vs neutralization
    phis = [round(x, 3) for x in np.linspace(0, 0.95, 96)]
    res["neut"] = {"phi": phis, "p1": [establishment_prob(1, phi=f) for f in phis], "r0": [r0(phi=f) for f in phis],
                   "threshold": 1 - (PARAMS["c"] / PARAMS["b"]) / (PARAMS["p"] / PARAMS["delta"] - 1)}
    # 4. hybrid SSA -> SDE trajectories from a single virion, conditioned on establishment
    starts, times = [], []
    tries = 0
    while len(starts) < 600:
        tries += 1
        ok, t, E, I, V = ssa_early(1, rng)
        if ok: starts.append([PARAMS["T0"], E, I, V]); times.append(t)
    print("established", len(starts), "of", tries, f"{time.time()-t0:.0f}s", flush=True)
    grid, V = cle(starts, times, t_end=30.0, dt=2e-3, rng=rng)
    # before handoff, treat viral load as below threshold (it is < 200 particles)
    det = []
    for row in V:
        idx = np.where(row >= 1000)[0]; det.append(float(grid[idx[0]]) if len(idx) else None)
    det_ok = np.array([d for d in det if d is not None])
    gode, Vode = ode(1.0, t_end=30.0)
    q = lambda a: np.nanpercentile(np.where(np.isnan(V), 0, V), a, axis=0)
    res["traj"] = {"t": grid.tolist(), "ode": Vode.tolist(),
                   "q05": q(5).tolist(), "q25": q(25).tolist(), "q50": q(50).tolist(), "q75": q(75).tolist(), "q95": q(95).tolist(),
                   "examples": [[None if np.isnan(x) else round(float(x), 2) for x in V[i]] for i in range(12)],
                   "handoff": [float(x) for x in times[:12]]}
    ode_det = float(gode[np.argmax(Vode >= 1000)])
    res["detect"] = {"times": det_ok.round(3).tolist(), "median": float(np.median(det_ok)), "p05": float(np.percentile(det_ok, 5)),
                     "p95": float(np.percentile(det_ok, 95)), "ode": ode_det, "n": len(det_ok), "tries": tries}
    print("detect", {k: v for k, v in res["detect"].items() if k != "times"}, f"{time.time()-t0:.0f}s")
    json.dump(res, open(out, "w"), separators=(",", ":"))

if __name__ == "__main__":
    main(sys.argv[1])
