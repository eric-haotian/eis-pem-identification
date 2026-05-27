"""Parameter metadata and optimisation-space transformations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """Specification of one physical model parameter."""

    name: str
    initial_value: float
    bounds: tuple[float, float]
    unit: str
    log_transform: bool = False

    def __post_init__(self) -> None:
        lower, upper = self.bounds
        if not self.name:
            raise ValueError("parameter name cannot be empty")
        if not np.all(np.isfinite([lower, upper, self.initial_value])):
            raise ValueError(f"{self.name} values and bounds must be finite")
        if lower >= upper:
            raise ValueError(f"{self.name} lower bound must be below upper bound")
        if not lower <= self.initial_value <= upper:
            raise ValueError(f"{self.name} initial value must be within its bounds")
        if self.log_transform and lower <= 0:
            raise ValueError(f"{self.name} log transformed bounds must be positive")

    @property
    def optimization_bounds(self) -> tuple[float, float]:
        """Bounds in the search space used by the optimizer."""

        if self.log_transform:
            return (float(np.log10(self.bounds[0])), float(np.log10(self.bounds[1])))
        return self.bounds

    def to_optimization(self, physical_value: float) -> float:
        """Map one physical parameter value to optimiser coordinates."""

        if not np.isfinite(physical_value):
            raise ValueError(f"{self.name} physical value must be finite")
        if self.log_transform:
            if physical_value <= 0:
                raise ValueError(f"{self.name} log transformed value must be positive")
            return float(np.log10(physical_value))
        return float(physical_value)

    def from_optimization(self, optimized_value: float) -> float:
        """Map one optimiser coordinate back to a physical parameter value."""

        if not np.isfinite(optimized_value):
            raise ValueError(f"{self.name} optimized value must be finite")
        if self.log_transform:
            return float(10.0**optimized_value)
        return float(optimized_value)
