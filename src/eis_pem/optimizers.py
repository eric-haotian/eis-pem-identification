"""Optimisation wrappers for PEM parameter identification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import differential_evolution, least_squares

from .costs import EISPredictionErrorCost
from .dataset import EISDataset
from .forward_models import ForwardModel
from .parameters import ParameterSpec
from .results import IdentificationResult

if TYPE_CHECKING:
    from .measurements import ParameterMeasurementDataset


@dataclass(frozen=True)
class DifferentialEvolutionOptimizer:
    """Bounded global optimiser for a complex-impedance PEM cost."""

    seed: int = 42
    workers: int = 1
    polish: bool = True
    popsize: int = 15
    maxiter: int = 1000
    tol: float = 1e-8
    relative: bool = False
    eps: float = 1e-12

    def fit(
        self,
        dataset: EISDataset,
        model: ForwardModel,
        parameter_specs: Sequence[ParameterSpec],
        weights: NDArray[np.floating] | None = None,
    ) -> IdentificationResult:
        specs = tuple(parameter_specs)
        if not specs:
            raise ValueError("parameter_specs cannot be empty")
        if len({spec.name for spec in specs}) != len(specs):
            raise ValueError("parameter names must be unique")

        cost = EISPredictionErrorCost(
            dataset=dataset,
            model=model,
            weights=weights,
            relative=self.relative,
            eps=self.eps,
        )

        def decode(search_values: NDArray[np.floating]) -> NDArray[np.float64]:
            return np.asarray(
                [
                    spec.from_optimization(value)
                    for spec, value in zip(specs, search_values, strict=True)
                ],
                dtype=float,
            )

        def objective(search_values: NDArray[np.floating]) -> float:
            return cost(decode(search_values))

        scipy_result = differential_evolution(
            objective,
            bounds=[spec.optimization_bounds for spec in specs],
            seed=self.seed,
            workers=self.workers,
            updating="immediate" if self.workers == 1 else "deferred",
            polish=self.polish,
            popsize=self.popsize,
            maxiter=self.maxiter,
            tol=self.tol,
        )
        theta_vector = decode(scipy_result.x)
        z_fit = model.simulate(dataset.freq_hz, theta_vector)
        residuals = dataset.z_obs - z_fit
        theta_best = {
            spec.name: float(value)
            for spec, value in zip(specs, theta_vector, strict=True)
        }
        metadata = {
            "optimizer": "differential_evolution",
            "seed": self.seed,
            "success": bool(scipy_result.success),
            "message": str(scipy_result.message),
            "nfev": int(scipy_result.nfev),
            "nit": int(scipy_result.nit),
            "search_space": {
                spec.name: "log10" if spec.log_transform else "linear" for spec in specs
            },
            "relative_residuals": self.relative,
        }
        return IdentificationResult(
            theta_best=theta_best,
            final_cost=float(scipy_result.fun),
            z_fit=z_fit,
            residuals=residuals,
            dataset=dataset,
            metadata=metadata,
        )


@dataclass(frozen=True)
class LeastSquaresOptimizer:
    """Bounded local PEM optimiser suitable for higher-dimensional models."""

    relative: bool = False
    eps: float = 1e-12
    max_nfev: int = 20000
    ftol: float = 1e-10
    xtol: float = 1e-10
    gtol: float = 1e-10
    modulus_weighting: bool = False
    n_starts: int = 1

    def fit(
        self,
        dataset: EISDataset,
        model: ForwardModel,
        parameter_specs: Sequence[ParameterSpec],
        weights: NDArray[np.floating] | None = None,
        auxiliary_measurements: ParameterMeasurementDataset | None = None,
    ) -> IdentificationResult:
        specs = tuple(parameter_specs)
        if not specs:
            raise ValueError("parameter_specs cannot be empty")
        if len({spec.name for spec in specs}) != len(specs):
            raise ValueError("parameter names must be unique")
        cost = EISPredictionErrorCost(
            dataset=dataset,
            model=model,
            weights=weights,
            relative=self.relative,
            modulus_weighting=self.modulus_weighting,
            eps=self.eps,
        )

        def decode(search_values: NDArray[np.floating]) -> NDArray[np.float64]:
            return np.asarray(
                [
                    spec.from_optimization(value)
                    for spec, value in zip(specs, search_values, strict=True)
                ],
                dtype=float,
            )

        def residual_vector(search_values: NDArray[np.floating]) -> NDArray[np.float64]:
            theta_vector = decode(search_values)
            residuals = cost.residuals(theta_vector)
            vector = np.concatenate((residuals.real, residuals.imag))
            if auxiliary_measurements is not None:
                vector = np.concatenate(
                    (
                        vector,
                        auxiliary_measurements.relative_residuals(
                            tuple(spec.name for spec in specs), theta_vector
                        ),
                    ),
                )
            return vector

        initial = np.asarray(
            [spec.to_optimization(spec.initial_value) for spec in specs], dtype=float
        )
        bounds = np.asarray([spec.optimization_bounds for spec in specs], dtype=float)

        # Generate multi-start initial points
        start_points = [initial]
        if self.n_starts > 1:
            rng = np.random.default_rng(seed=0)
            ndim = len(specs)
            for _ in range(self.n_starts - 1):
                point = np.empty(ndim, dtype=float)
                for j in range(ndim):
                    low, high = bounds[j, 0], bounds[j, 1]
                    point[j] = rng.uniform(low, high)
                start_points.append(point)

        best_result = None
        best_cost_value = float("inf")

        for x0 in start_points:
            scipy_result = least_squares(
                residual_vector,
                x0=x0,
                bounds=(bounds[:, 0], bounds[:, 1]),
                max_nfev=self.max_nfev,
                ftol=self.ftol,
                xtol=self.xtol,
                gtol=self.gtol,
            )
            cost_value = float(np.sum(residual_vector(scipy_result.x) ** 2))
            if cost_value < best_cost_value:
                best_cost_value = cost_value
                best_result = scipy_result

        scipy_result = best_result
        theta_vector = decode(scipy_result.x)
        z_fit = model.simulate(dataset.freq_hz, theta_vector)
        residuals = dataset.z_obs - z_fit
        theta_best = {
            spec.name: float(value)
            for spec, value in zip(specs, theta_vector, strict=True)
        }
        metadata = {
            "optimizer": "least_squares",
            "success": bool(scipy_result.success),
            "message": str(scipy_result.message),
            "nfev": int(scipy_result.nfev),
            "cost": float(scipy_result.cost),
            "optimality": float(scipy_result.optimality),
            "search_space": {
                spec.name: "log10" if spec.log_transform else "linear" for spec in specs
            },
            "relative_residuals": self.relative,
            "modulus_weighting": self.modulus_weighting,
            "n_starts": self.n_starts,
            "auxiliary_measurement_count": (
                0
                if auxiliary_measurements is None
                else len(auxiliary_measurements.parameter_names)
            ),
        }
        return IdentificationResult(
            theta_best=theta_best,
            final_cost=best_cost_value,
            z_fit=z_fit,
            residuals=residuals,
            dataset=dataset,
            metadata=metadata,
        )
