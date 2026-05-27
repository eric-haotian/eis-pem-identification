"""Local sensitivity and identifiability diagnostics for PEM fits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .dataset import EISDataset
from .forward_models import ForwardModel
from .parameters import ParameterSpec

if TYPE_CHECKING:
    from .measurements import ParameterMeasurementDataset


@dataclass(frozen=True)
class IdentifiabilityReport:
    """Numerical rank and per-parameter sensitivity at a parameter point."""

    parameter_names: tuple[str, ...]
    parameter_values: NDArray[np.float64]
    sensitivity_norms: NDArray[np.float64]
    singular_values: NDArray[np.float64]
    rank: int
    condition_number: float
    correlation_matrix: NDArray[np.float64] | None = None

    @property
    def parameter_count(self) -> int:
        return len(self.parameter_names)

    def to_frame(self) -> pd.DataFrame:
        """Return a parameter-indexed summary suitable for CSV output."""

        return pd.DataFrame(
            {
                "parameter": self.parameter_names,
                "value": self.parameter_values,
                "sensitivity_norm": self.sensitivity_norms,
                "jacobian_rank": self.rank,
                "parameter_count": self.parameter_count,
                "condition_number": self.condition_number,
            }
        )

    def singular_values_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "singular_value_index": np.arange(1, self.singular_values.size + 1),
                "singular_value": self.singular_values,
            }
        )

    def correlation_frame(self) -> pd.DataFrame | None:
        """Return the parameter correlation matrix as a labelled DataFrame."""
        if self.correlation_matrix is None:
            return None
        return pd.DataFrame(
            self.correlation_matrix,
            index=self.parameter_names,
            columns=self.parameter_names,
        )

    def export_csv(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_csv(output_path, index=False)
        return output_path

    def export_singular_values_csv(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.singular_values_frame().to_csv(output_path, index=False)
        return output_path

    def export_correlation_csv(self, path: str | Path) -> Path | None:
        """Export the parameter correlation matrix to CSV."""
        frame = self.correlation_frame()
        if frame is None:
            return None
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path)
        return output_path


def evaluate_local_identifiability(
    model: ForwardModel,
    dataset: EISDataset,
    parameter_specs: Sequence[ParameterSpec],
    theta: NDArray[np.floating],
    relative: bool = True,
    step_size: float = 1e-5,
    auxiliary_measurements: ParameterMeasurementDataset | None = None,
) -> IdentifiabilityReport:
    """Evaluate a finite-difference residual Jacobian at ``theta``."""

    specs = tuple(parameter_specs)
    values = np.asarray(theta, dtype=float)
    if values.shape != (len(specs),):
        raise ValueError("theta must match parameter_specs length")
    if not np.isfinite(step_size) or step_size <= 0:
        raise ValueError("step_size must be finite and positive")
    search_point = np.asarray(
        [spec.to_optimization(value) for spec, value in zip(specs, values, strict=True)]
    )
    reference = model.simulate(dataset.freq_hz, values)
    scale = (
        np.maximum(np.abs(dataset.z_obs), 1e-12)
        if relative
        else np.ones_like(reference.real)
    )

    def decode(search_values: NDArray[np.floating]) -> NDArray[np.float64]:
        return np.asarray(
            [
                spec.from_optimization(value)
                for spec, value in zip(specs, search_values, strict=True)
            ]
        )

    def prediction_delta(search_values: NDArray[np.floating]) -> NDArray[np.float64]:
        physical_values = decode(search_values)
        delta = (model.simulate(dataset.freq_hz, physical_values) - reference) / scale
        vector = np.concatenate((delta.real, delta.imag))
        if auxiliary_measurements is not None:
            reference_auxiliary = auxiliary_measurements.relative_residuals(
                tuple(spec.name for spec in specs), values
            )
            perturbed_auxiliary = auxiliary_measurements.relative_residuals(
                tuple(spec.name for spec in specs), physical_values
            )
            vector = np.concatenate((vector, perturbed_auxiliary - reference_auxiliary))
        return vector

    extra_rows = (
        0
        if auxiliary_measurements is None
        else len(auxiliary_measurements.parameter_names)
    )
    jacobian = np.empty(
        (dataset.freq_hz.size * 2 + extra_rows, len(specs)), dtype=float
    )
    for index in range(len(specs)):
        upper = search_point.copy()
        lower = search_point.copy()
        upper[index] += step_size
        lower[index] -= step_size
        jacobian[:, index] = (prediction_delta(upper) - prediction_delta(lower)) / (
            2.0 * step_size
        )
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    tolerance = (
        singular_values[0] * max(jacobian.shape) * np.finfo(float).eps
        if singular_values.size
        else 0.0
    )
    rank = int(np.sum(singular_values > tolerance))
    condition_number = (
        float(singular_values[0] / singular_values[-1])
        if singular_values.size and singular_values[-1] > 0
        else float("inf")
    )
    # Compute correlation matrix from Fisher information
    information = jacobian.T @ jacobian
    diag = np.sqrt(np.maximum(np.diag(information), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        outer = np.outer(diag, diag)
        correlation = np.where(outer > 0, information / outer, 0.0)
    np.fill_diagonal(correlation, 1.0)
    return IdentifiabilityReport(
        parameter_names=tuple(spec.name for spec in specs),
        parameter_values=values,
        sensitivity_norms=np.linalg.norm(jacobian, axis=0),
        singular_values=singular_values,
        rank=rank,
        condition_number=condition_number,
        correlation_matrix=correlation,
    )
