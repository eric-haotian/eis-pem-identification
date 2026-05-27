"""Identifiability-aware parameter selection for robust SEIS PEM fits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .dataset import EISDataset
from .forward_models import ForwardModel
from .parameters import ParameterSpec
from .seis_model import DEFAULT_SELECTED_PARAMETER_NAMES

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ParameterSelection:
    """Selected free parameters and documented constraints for one experiment."""

    parameter_names: tuple[str, ...]
    reference_values: dict[str, float]
    free_names: tuple[str, ...]
    fixed_values: dict[str, float]
    statuses: dict[str, str]
    reasons: dict[str, str]
    ci95_relative: dict[str, float]
    original_singular_values: FloatArray
    selected_singular_values: FloatArray
    original_condition_number: float
    reduced_condition_number: float
    assumed_noise_level: float
    max_condition_number: float
    max_relative_ci95: float
    protected_names: tuple[str, ...]

    def __post_init__(self) -> None:
        parameter_set = set(self.parameter_names)
        free_set = set(self.free_names)
        fixed_set = set(self.fixed_values)
        if not self.parameter_names or len(parameter_set) != len(self.parameter_names):
            raise ValueError("parameter_names must be unique and non-empty")
        if (
            free_set.intersection(fixed_set)
            or free_set.union(fixed_set) != parameter_set
        ):
            raise ValueError("free and fixed parameters must partition parameter_names")
        for mapping in (
            self.reference_values,
            self.statuses,
            self.reasons,
            self.ci95_relative,
        ):
            if set(mapping) != parameter_set:
                raise ValueError("selection metadata must cover all parameters")
        if not set(self.protected_names).issubset(parameter_set):
            raise ValueError("protected_names must occur in parameter_names")

    @property
    def fixed_names(self) -> tuple[str, ...]:
        """Return constrained parameter names in original parameter order."""

        return tuple(name for name in self.parameter_names if name in self.fixed_values)

    def to_frame(self) -> pd.DataFrame:
        """Return one audit row per complete-model physical parameter."""

        return pd.DataFrame(
            {
                "parameter": self.parameter_names,
                "status": [self.statuses[name] for name in self.parameter_names],
                "reason": [self.reasons[name] for name in self.parameter_names],
                "reference_value": [
                    self.reference_values[name] for name in self.parameter_names
                ],
                "fixed_value": [
                    self.fixed_values.get(name, np.nan) for name in self.parameter_names
                ],
                "ci95_relative": [
                    self.ci95_relative[name] for name in self.parameter_names
                ],
                "is_protected": [
                    name in self.protected_names for name in self.parameter_names
                ],
                "original_condition_number": self.original_condition_number,
                "reduced_condition_number": self.reduced_condition_number,
            }
        )

    def singular_values_frame(self) -> pd.DataFrame:
        """Return full and selected-model spectra for conditioning diagnostics."""

        frames = []
        for stage, singular_values in (
            ("full", self.original_singular_values),
            ("selected", self.selected_singular_values),
        ):
            frames.append(
                pd.DataFrame(
                    {
                        "stage": stage,
                        "singular_value_index": np.arange(1, singular_values.size + 1),
                        "singular_value": singular_values,
                    }
                )
            )
        return pd.concat(frames, ignore_index=True)

    def export_csv(self, path: str | Path) -> Path:
        """Export parameter selection decisions and their numerical context."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_csv(output_path, index=False)
        return output_path

    def export_singular_values_csv(self, path: str | Path) -> Path:
        """Export singular values before and after parameter selection."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.singular_values_frame().to_csv(output_path, index=False)
        return output_path


@dataclass(frozen=True)
class IdentifiabilitySelector:
    """Select EIS-supported parameters while preserving named target parameters."""

    max_condition_number: float = 1e4
    assumed_noise_level: float = 0.005
    max_relative_ci95: float = 0.10
    protected_names: tuple[str, ...] = DEFAULT_SELECTED_PARAMETER_NAMES
    step_size: float = 1e-5
    eps: float = 1e-12
    covariance_rcond: float = 1e-14

    def __post_init__(self) -> None:
        for name, value in (
            ("max_condition_number", self.max_condition_number),
            ("max_relative_ci95", self.max_relative_ci95),
            ("step_size", self.step_size),
            ("eps", self.eps),
            ("covariance_rcond", self.covariance_rcond),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.assumed_noise_level) or self.assumed_noise_level < 0:
            raise ValueError("assumed_noise_level must be finite and non-negative")
        if len(set(self.protected_names)) != len(self.protected_names):
            raise ValueError("protected_names must be unique")

    def select(
        self,
        dataset: EISDataset,
        model: ForwardModel,
        parameter_specs: Sequence[ParameterSpec],
        theta_reference: NDArray[np.floating],
    ) -> ParameterSelection:
        """Select the free vector for a local relative-residual PEM problem."""

        specs = tuple(parameter_specs)
        if not specs:
            raise ValueError("parameter_specs cannot be empty")
        parameter_names = tuple(spec.name for spec in specs)
        if len(set(parameter_names)) != len(parameter_names):
            raise ValueError("parameter names must be unique")
        if not set(self.protected_names).issubset(parameter_names):
            raise ValueError("protected parameters must occur in parameter_specs")
        reference = np.asarray(theta_reference, dtype=float)
        if reference.shape != (len(specs),) or not np.all(np.isfinite(reference)):
            raise ValueError("theta_reference must be finite and match parameter_specs")

        jacobian = _relative_prediction_jacobian(
            dataset=dataset,
            model=model,
            specs=specs,
            theta_reference=reference,
            step_size=self.step_size,
            eps=self.eps,
        )
        full_singular_values, original_condition_number = _spectrum(jacobian)
        active = list(range(len(specs)))
        fixed_values: dict[str, float] = {}
        statuses = {name: "estimated" for name in parameter_names}
        reasons = {
            name: "selected_under_identifiability_thresholds"
            for name in parameter_names
        }
        ci95 = {name: float("nan") for name in parameter_names}
        protected = set(self.protected_names)

        # Correlation-based pre-screening: fix the weaker member of
        # highly correlated non-protected parameter pairs (|ρ| > 0.98).
        sensitivity_norms = np.linalg.norm(jacobian, axis=0)
        information = jacobian.T @ jacobian
        diag = np.sqrt(np.maximum(np.diag(information), 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            outer = np.outer(diag, diag)
            correlation = np.where(outer > 0, information / outer, 0.0)
        np.fill_diagonal(correlation, 0.0)
        removed_by_correlation: set[int] = set()
        for i in range(len(specs)):
            if i in removed_by_correlation or parameter_names[i] in protected:
                continue
            for j in range(i + 1, len(specs)):
                if j in removed_by_correlation or parameter_names[j] in protected:
                    continue
                if abs(correlation[i, j]) > 0.98:
                    # Fix the one with the lower sensitivity norm
                    victim = i if sensitivity_norms[i] < sensitivity_norms[j] else j
                    removed_by_correlation.add(victim)
                    name = parameter_names[victim]
                    fixed_values[name] = specs[victim].initial_value
                    statuses[name] = "fixed_identifiability"
                    reasons[name] = "correlation_above_threshold"
        active = [i for i in active if i not in removed_by_correlation]

        while True:
            active_jacobian = jacobian[:, active]
            _, singular_values, right_vectors = np.linalg.svd(
                active_jacobian, full_matrices=False
            )
            condition_number = _condition_number(singular_values)
            if condition_number <= self.max_condition_number:
                break
            removal_position = next(
                (
                    int(position)
                    for position in np.argsort(np.abs(right_vectors[-1]))[::-1]
                    if parameter_names[active[int(position)]] not in protected
                ),
                None,
            )
            if removal_position is None:
                break
            index = active.pop(removal_position)
            name = parameter_names[index]
            fixed_values[name] = specs[index].initial_value
            statuses[name] = "fixed_identifiability"
            reasons[name] = "weak_singular_direction"

        while True:
            active_ci95 = _relative_ci95(
                jacobian[:, active],
                [specs[index] for index in active],
                reference[active],
                self.assumed_noise_level,
                self.covariance_rcond,
            )
            candidates = [
                (active_ci95[position], position)
                for position, index in enumerate(active)
                if parameter_names[index] not in protected
                and active_ci95[position] > self.max_relative_ci95
            ]
            if not candidates:
                break
            removed_ci95, removal_position = max(candidates)
            index = active.pop(removal_position)
            name = parameter_names[index]
            fixed_values[name] = specs[index].initial_value
            statuses[name] = "fixed_identifiability"
            reasons[name] = "predicted_ci95_above_limit"
            ci95[name] = float(removed_ci95)

        selected_singular_values, reduced_condition_number = _spectrum(
            jacobian[:, active]
        )
        selected_ci95 = _relative_ci95(
            jacobian[:, active],
            [specs[index] for index in active],
            reference[active],
            self.assumed_noise_level,
            self.covariance_rcond,
        )
        for position, index in enumerate(active):
            name = parameter_names[index]
            ci95[name] = float(selected_ci95[position])
            if name in protected and selected_ci95[position] > self.max_relative_ci95:
                statuses[name] = "protected_high_uncertainty"
                reasons[name] = "protected_parameter_retained_despite_ci95_limit"

        return ParameterSelection(
            parameter_names=parameter_names,
            reference_values=dict(zip(parameter_names, reference, strict=True)),
            free_names=tuple(parameter_names[index] for index in active),
            fixed_values=fixed_values,
            statuses=statuses,
            reasons=reasons,
            ci95_relative=ci95,
            original_singular_values=full_singular_values,
            selected_singular_values=selected_singular_values,
            original_condition_number=original_condition_number,
            reduced_condition_number=reduced_condition_number,
            assumed_noise_level=self.assumed_noise_level,
            max_condition_number=self.max_condition_number,
            max_relative_ci95=self.max_relative_ci95,
            protected_names=self.protected_names,
        )


@dataclass(frozen=True)
class ReducedParameterModel:
    """Expose selected free parameters while simulating the complete model."""

    full_model: ForwardModel
    selection: ParameterSelection

    def __post_init__(self) -> None:
        model_names = getattr(self.full_model, "parameter_names", None)
        if (
            model_names is not None
            and tuple(model_names) != self.selection.parameter_names
        ):
            raise ValueError("full model parameter order must match the selection")

    def expand_theta(self, theta_free: NDArray[np.floating]) -> FloatArray:
        """Expand physical free-parameter values to complete-model order."""

        free_values = np.asarray(theta_free, dtype=float)
        if free_values.shape != (len(self.selection.free_names),):
            raise ValueError("theta_free must match selected free parameter count")
        values = dict(self.selection.fixed_values)
        values.update(dict(zip(self.selection.free_names, free_values, strict=True)))
        return np.asarray(
            [values[name] for name in self.selection.parameter_names], dtype=float
        )

    def simulate(
        self, freq_hz: NDArray[np.floating], theta: NDArray[np.floating]
    ) -> NDArray[np.complex128]:
        """Simulate the full physical model from only selected free values."""

        return self.full_model.simulate(freq_hz, self.expand_theta(theta))


def _relative_prediction_jacobian(
    dataset: EISDataset,
    model: ForwardModel,
    specs: tuple[ParameterSpec, ...],
    theta_reference: FloatArray,
    step_size: float,
    eps: float,
) -> FloatArray:
    search_reference = np.asarray(
        [
            spec.to_optimization(value)
            for spec, value in zip(specs, theta_reference, strict=True)
        ],
        dtype=float,
    )
    prediction_reference = model.simulate(dataset.freq_hz, theta_reference)
    scale = np.maximum(np.abs(prediction_reference), eps)

    def decode(search_values: FloatArray) -> FloatArray:
        return np.asarray(
            [
                spec.from_optimization(value)
                for spec, value in zip(specs, search_values, strict=True)
            ],
            dtype=float,
        )

    def delta(search_values: FloatArray) -> FloatArray:
        prediction_delta = (
            model.simulate(dataset.freq_hz, decode(search_values))
            - prediction_reference
        ) / scale
        return np.concatenate((prediction_delta.real, prediction_delta.imag))

    jacobian = np.empty((dataset.freq_hz.size * 2, len(specs)), dtype=float)
    for index in range(len(specs)):
        upper = search_reference.copy()
        lower = search_reference.copy()
        upper[index] += step_size
        lower[index] -= step_size
        jacobian[:, index] = (delta(upper) - delta(lower)) / (2.0 * step_size)
    return jacobian


def _spectrum(jacobian: FloatArray) -> tuple[FloatArray, float]:
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    return singular_values, _condition_number(singular_values)


def _condition_number(singular_values: FloatArray) -> float:
    if singular_values.size == 0 or singular_values[-1] <= 0:
        return float("inf")
    return float(singular_values[0] / singular_values[-1])


def _relative_ci95(
    jacobian: FloatArray,
    specs: Sequence[ParameterSpec],
    theta_reference: FloatArray,
    noise_level: float,
    covariance_rcond: float,
) -> FloatArray:
    information = jacobian.T @ jacobian
    covariance = noise_level**2 * np.linalg.pinv(
        information, rcond=covariance_rcond, hermitian=True
    )
    standard_deviation = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    relative_standard_deviation = np.asarray(
        [
            np.log(10.0) * deviation if spec.log_transform else deviation / abs(value)
            for spec, value, deviation in zip(
                specs, theta_reference, standard_deviation, strict=True
            )
        ],
        dtype=float,
    )
    return 1.96 * relative_standard_deviation
