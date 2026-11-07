# Stochastic viral dynamics

Why do most HIV exposures never become infections, and how much does a neutralizing antibody change the odds?

A deterministic within-host model says any virus with R0 > 1 always takes off. Infections start from a handful of virions, so the early phase is a matter of chance. This repo simulates the first days of HIV infection three ways:

- **Exact stochastic simulation** (Gillespie) while counts are small
- **Chemical Langevin equation**, a stochastic differential equation integrated with Euler-Maruyama, once the infection grows
- **Branching-process theory** for the exact establishment probability, used to check the simulations

The model tracks target cells, eclipse-phase and productive infected cells, and infectious virions, with standard HIV rates (clearance 23/day, infected-cell death 1/day, eclipse 1 day) and R0 = 8. Neutralizing antibody blocks a fixed fraction of infection events.

## Results

| | Value |
|---|---|
| Chance one virion establishes infection (R0 = 8) | 14.6% |
| Ten virions | 79% |
| Per-virion risk at 50 / 80 / 85% neutralization | 7.0% / 1.8% / 0.83% |
| Neutralization where R0 falls below 1 | 89.4% |
| Time to 1,000 infectious virions/mL, established infections (median, 5-95%) | 5.25 days (4.0-7.75) |
| Same, deterministic model | 6.0 days |

Gillespie estimates match the branching-process formula within two standard errors in all six checks (20,000 runs for one virion, 4,000 for ten).

## Run

```
pip install -r requirements.txt
pytest
python run_experiments.py results/viral.json   # about 40 seconds
```

## Layout
- `viral/model.py` model, Gillespie simulator, branching-process solution, vectorized Euler-Maruyama integrator, ODE
- `run_experiments.py` validation, dose and neutralization sweeps, hybrid trajectories

## References
Gillespie, *J Phys Chem* 1977 and *J Chem Phys* 2000; Pearson, Krapivsky and Perelson, *PLoS Comput Biol* 2011; Ribeiro et al., *J Virol* 2010; Ramratnam et al., *Lancet* 1999; Markowitz et al., *J Virol* 2003; Keele et al., *PNAS* 2008; Corey et al., *NEJM* 2021.
