"""Model parameters for early within-host HIV infection.

Rates are per day. Clearance, infected-cell death and eclipse duration are
standard HIV estimates (Ramratnam et al. 1999; Markowitz et al. 2003). The
production and infection rates are set so that R0 = 8 (Ribeiro et al. 2010).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Parameters:
    """Rate constants of the target-cell-limited model with an eclipse phase."""

    clearance_rate: float = 23.0
    """Virion clearance, c (1/day)."""

    infected_death_rate: float = 1.0
    """Death of productively infected cells, delta (1/day)."""

    eclipse_exit_rate: float = 1.0
    """Rate at which eclipse-phase cells start producing virus, k (1/day)."""

    production_rate: float = 48.0
    """Infectious virions released per productive cell, p (1/day)."""

    infection_rate: float = 4.6
    """Per-virion infection rate at baseline target cells, b = beta * T0 (1/day)."""

    target_cells: float = 1e5
    """Activated target cells at baseline, T0 (cells/mL)."""

    target_turnover_rate: float = 0.01
    """Target-cell turnover, dT (1/day)."""

    def __post_init__(self) -> None:
        for name, value in self.as_dict().items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

    def effective_infection_rate(self, neutralization: float = 0.0) -> float:
        """Infection rate per virion when antibody blocks a fraction of infections."""
        check_neutralization(neutralization)
        return self.infection_rate * (1 - neutralization)

    @property
    def target_supply_rate(self) -> float:
        """Supply of new target cells that holds T at T0 without infection (cells/mL/day)."""
        return self.target_turnover_rate * self.target_cells

    def as_dict(self) -> dict[str, float]:
        """Parameters under their short symbols, as used in the results file."""
        return {
            "c": self.clearance_rate,
            "delta": self.infected_death_rate,
            "k": self.eclipse_exit_rate,
            "p": self.production_rate,
            "b": self.infection_rate,
            "T0": self.target_cells,
            "dT": self.target_turnover_rate,
        }


def check_neutralization(neutralization: float) -> None:
    """Raise if the neutralized fraction is outside [0, 1]."""
    if not 0.0 <= neutralization <= 1.0:
        raise ValueError(f"neutralization must be between 0 and 1, got {neutralization}")


DEFAULT = Parameters()
