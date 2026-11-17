"""Command-line interface: `viral-dynamics --help`."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import numpy as np

from .experiments import run_all, write_results
from .gillespie import simulate_early_infection
from .theory import (
    basic_reproduction_number,
    critical_neutralization,
    establishment_probability,
)


def _fraction(text: str) -> float:
    value = float(text)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(f"must be between 0 and 1, got {text}")
    return value


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {text}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="viral-dynamics",
        description="Stochastic models of the first days of HIV infection.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    theory = commands.add_parser(
        "theory", help="exact establishment probability from branching-process theory"
    )
    simulate = commands.add_parser(
        "simulate", help="estimate establishment probability by Gillespie simulation"
    )
    for sub in (theory, simulate):
        sub.add_argument(
            "--virions", type=_positive_int, default=1, help="virions at exposure (default 1)"
        )
        sub.add_argument(
            "--neutralization",
            type=_fraction,
            default=0.0,
            help="fraction of infection events blocked by antibody, 0 to 1 (default 0)",
        )
    simulate.add_argument(
        "--runs", type=_positive_int, default=5000, help="simulated exposures (default 5000)"
    )
    simulate.add_argument("--seed", type=int, default=0, help="random seed (default 0)")

    experiments = commands.add_parser(
        "experiments", help="run every experiment and write the results file"
    )
    experiments.add_argument("--out", default="results/viral.json", help="output JSON path")
    experiments.add_argument("--seed", type=int, default=2026, help="random seed (default 2026)")

    figures = commands.add_parser("figures", help="draw figures from a results file")
    figures.add_argument("--results", default="results/viral.json", help="results JSON path")
    figures.add_argument("--out", default="docs/figures", help="output directory")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "theory":
        r0 = basic_reproduction_number(neutralization=args.neutralization)
        probability = establishment_probability(args.virions, neutralization=args.neutralization)
        print(f"R0:                         {r0:.3f}")
        print(f"Establishment probability:  {probability:.4f}")
        print(f"Critical neutralization:    {critical_neutralization():.4f}")

    elif args.command == "simulate":
        rng = np.random.default_rng(args.seed)
        established = sum(
            simulate_early_infection(
                args.virions, rng, neutralization=args.neutralization
            ).established
            for _ in range(args.runs)
        )
        frequency = established / args.runs
        standard_error = (frequency * (1 - frequency) / args.runs) ** 0.5
        exact = establishment_probability(args.virions, neutralization=args.neutralization)
        print(f"Simulated: {frequency:.4f} (standard error {standard_error:.4f})")
        print(f"Theory:    {exact:.4f}")

    elif args.command == "experiments":
        results = run_all(seed=args.seed, verbose=True)
        write_results(results, args.out)
        print(f"Wrote {args.out}")

    elif args.command == "figures":
        from .plotting import make_figures

        for path in make_figures(args.results, args.out):
            print(f"Wrote {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
