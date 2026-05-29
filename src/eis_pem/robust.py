"""Identifiability-aware parameter selection for robust SEIS PEM fits."""
# learning AI website www.haotianblog.com
from __future__ import annotations

import enum
import logging
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

logger = logging.getLogger(__name__)


class IdentifiabilityStrategy(enum.Enum):
    """Parameter selection strategy.

    CLASSIC: Original three-stage selection (correlation, SVD, CI95).
    ADAPTIVE: Four-criterion joint evaluation with iterative refinement.
              Recommended for real experimental data with noise, limited
              frequency range, and model mismatch.
    """

    CLASSIC = "classic"
    ADAPTIVE = "adaptive"


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
    """Select EIS-supported parameters while preserving named target parameters.

    Two strategies are available:

    **CLASSIC** (default): Three-stage selection \u2014 correlation pre-screening,
    SVD condition number reduction, and CI95 pruning. Backward-compatible
    with the original implementation.

    **ADAPTIVE**: Four-criterion joint evaluation with iterative refinement.
    Uses rank(J), \u03ba(J), CI95, and parameter correlation together to decide
    which parameters to fix. Recommended for real experimental data where
    noise is higher, frequency range is limited, and model mismatch exists.
    """

    max_condition_number: float = 1e4
    assumed_noise_level: float = 0.005
    max_relative_ci95: float = 0.10
    protected_names: tuple[str, ...] = DEFAULT_SELECTED_PARAMETER_NAMES
    step_size: float = 1e-5
    eps: float = 1e-12
    covariance_rcond: float = 1e-14

    # Adaptive strategy fields (only used when strategy=ADAPTIVE)
    strategy: IdentifiabilityStrategy = IdentifiabilityStrategy.CLASSIC
    min_rank_fraction: float = 0.5
    max_correlation_threshold: float = 0.95
    eigenvalue_gap_ratio: float = 100.0
    max_selection_iterations: int = 50

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
        if not isinstance(self.strategy, IdentifiabilityStrategy):
            raise ValueError("strategy must be an IdentifiabilityStrategy")
        if not 0 < self.min_rank_fraction <= 1:
            raise ValueError("min_rank_fraction must be in (0, 1]")
        if not 0 < self.max_correlation_threshold < 1:
            raise ValueError("max_correlation_threshold must be in (0, 1)")
        if not np.isfinite(self.eigenvalue_gap_ratio) or self.eigenvalue_gap_ratio <= 1:
            raise ValueError("eigenvalue_gap_ratio must be finite and > 1")
        if self.max_selection_iterations < 1:
            raise ValueError("max_selection_iterations must be >= 1")

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

        if self.strategy == IdentifiabilityStrategy.ADAPTIVE:
            return self._select_adaptive(jacobian, specs, parameter_names, reference)
        return self._select_classic(jacobian, specs, parameter_names, reference)

    def _select_classic(
        self,
        jacobian: FloatArray,
        specs: tuple[ParameterSpec, ...],
        parameter_names: tuple[str, ...],
        reference: FloatArray,
    ) -> ParameterSelection:
        """Original three-stage selection: correlation, SVD, CI95."""

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

    def _select_adaptive(
        self,
        jacobian: FloatArray,
        specs: tuple[ParameterSpec, ...],
        parameter_names: tuple[str, ...],
        reference: FloatArray,
    ) -> ParameterSelection:
        """Four-criterion adaptive selection with iterative refinement.

        Simultaneously evaluates rank(J), κ(J), CI95, and parameter
        correlation at each iteration. Unlike the classic mode which
        applies criteria sequentially, the adaptive mode re-evaluates
        all four diagnostics after each parameter fixation, choosing
        the single worst-offending parameter to fix at each step.

        This prevents over-aggressive pruning that can occur when
        sequential stages interact poorly.
        """

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

        for iteration in range(self.max_selection_iterations):
            if len(active) <= len(protected):
                break

            active_jacobian = jacobian[:, active]
            n_active = len(active)

            # --- Criterion 1: Rank deficiency ---
            singular_values = np.linalg.svd(active_jacobian, compute_uv=False)
            effective_rank = _effective_rank(
                singular_values, self.eigenvalue_gap_ratio
            )
            min_required_rank = max(
                1, int(np.ceil(n_active * self.min_rank_fraction))
            )
            rank_deficient = effective_rank < min_required_rank

            # --- Criterion 2: Condition number ---
            condition_number = _condition_number(singular_values)
            ill_conditioned = condition_number > self.max_condition_number

            # --- Criterion 3: CI95 ---
            active_ci95 = _relative_ci95(
                active_jacobian,
                [specs[index] for index in active],
                reference[active],
                self.assumed_noise_level,
                self.covariance_rcond,
            )

            # --- Criterion 4: Correlation ---
            information = active_jacobian.T @ active_jacobian
            diag = np.sqrt(np.maximum(np.diag(information), 0.0))
            with np.errstate(divide="ignore", invalid="ignore"):
                outer = np.outer(diag, diag)
                corr = np.where(outer > 0, information / outer, 0.0)
            np.fill_diagonal(corr, 0.0)
            max_abs_corr = np.max(np.abs(corr)) if corr.size > 0 else 0.0
            high_correlation = max_abs_corr > self.max_correlation_threshold

            # --- Check if all criteria pass ---
            all_ci95_ok = all(
                active_ci95[pos] <= self.max_relative_ci95
                or parameter_names[active[pos]] in protected
                for pos in range(n_active)
            )

            if (
                not rank_deficient
                and not ill_conditioned
                and all_ci95_ok
                and not high_correlation
            ):
                logger.info(
                    "Adaptive selection converged after %d iterations: "
                    "%d free parameters, κ=%.2e, rank=%d/%d",
                    iteration,
                    n_active,
                    condition_number,
                    effective_rank,
                    n_active,
                )
                break

            # --- Score each non-protected parameter for removal ---
            # Higher score = more likely to be fixed
            removal_scores: list[tuple[float, int, str]] = []
            sensitivity_norms = np.linalg.norm(active_jacobian, axis=0)
            max_sens = np.max(sensitivity_norms) if sensitivity_norms.size > 0 else 1.0

            for pos in range(n_active):
                idx = active[pos]
                if parameter_names[idx] in protected:
                    continue

                score = 0.0
                reason_parts = []

                # CI95 penalty
                ci_val = active_ci95[pos]
                if ci_val > self.max_relative_ci95:
                    score += 10.0 * (ci_val / self.max_relative_ci95)
                    reason_parts.append("high_ci95")

                # Correlation penalty — max |ρ| with any other parameter
                max_corr_for_param = np.max(np.abs(corr[pos, :])) if n_active > 1 else 0.0
                if max_corr_for_param > self.max_correlation_threshold:
                    score += 5.0 * max_corr_for_param
                    reason_parts.append("high_correlation")

                # Low sensitivity penalty (inverse of normalised sensitivity)
                rel_sens = sensitivity_norms[pos] / max_sens if max_sens > 0 else 0.0
                score += 1.0 * (1.0 - rel_sens)
                if rel_sens < 0.01:
                    reason_parts.append("near_zero_sensitivity")

                # Singular direction alignment penalty (if ill-conditioned)
                if ill_conditioned and singular_values.size > 0:
                    _, _, vt = np.linalg.svd(active_jacobian, full_matrices=False)
                    alignment = abs(vt[-1, pos]) if vt.shape[0] > 0 else 0.0
                    score += 3.0 * alignment
                    if alignment > 0.3:
                        reason_parts.append("weak_singular_direction")

                reason_str = "+".join(reason_parts) if reason_parts else "low_sensitivity"
                removal_scores.append((score, pos, reason_str))

            if not removal_scores:
                break

            # Remove the parameter with the highest score
            removal_scores.sort(reverse=True)
            _, removal_pos, removal_reason = removal_scores[0]
            idx = active.pop(removal_pos)
            name = parameter_names[idx]
            fixed_values[name] = specs[idx].initial_value
            statuses[name] = "fixed_identifiability"
            reasons[name] = f"adaptive_{removal_reason}"
            ci95[name] = float(active_ci95[removal_pos])

            logger.debug(
                "Iteration %d: fixed %s (reason=%s, ci95=%.4f)",
                iteration,
                name,
                removal_reason,
                active_ci95[removal_pos],
            )

        # --- Final diagnostics on remaining active set ---
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


def _effective_rank(
    singular_values: FloatArray, gap_threshold: float = 100.0
) -> int:
    """Effective rank based on eigenvalue gap analysis.

    More conservative than numerical rank.  Scans singular values from
    largest to smallest; the effective rank is the position just before
    the first gap whose ratio exceeds *gap_threshold*.

    Parameters
    ----------
    singular_values
        Sorted (descending) singular values from an SVD.
    gap_threshold
        A consecutive ratio ``σ_i / σ_{i+1} > gap_threshold`` defines
        a gap.  Default 100.
    """

    values = np.asarray(singular_values, dtype=float)
    if values.size == 0:
        return 0
    # Drop numerically zero entries
    positive = values[values > 0]
    if positive.size <= 1:
        return int(positive.size)
    for k in range(positive.size - 1):
        ratio = positive[k] / positive[k + 1]
        if ratio > gap_threshold:
            return k + 1
    return int(positive.size)


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
