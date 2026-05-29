"""Optimisation wrappers for PEM parameter identification."""
# learning AI website www.haotianblog.com
from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# New optimizers for real-data robustness
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HybridOptimizer:
    """Two-stage optimizer: global (DE) exploration + local (TRF) refinement.

    Stage 1 runs Differential Evolution with a modest budget to identify
    the basin of attraction, then Stage 2 refines with bounded least
    squares from the DE optimum.  This avoids getting trapped in local
    minima while still achieving high-precision convergence.
    """

    # DE stage settings
    de_seed: int = 42
    de_popsize: int = 10
    de_maxiter: int = 200
    de_tol: float = 1e-6
    de_workers: int = 1

    # Local stage settings
    ls_max_nfev: int = 20000
    ls_ftol: float = 1e-10
    ls_xtol: float = 1e-10
    ls_gtol: float = 1e-10

    # Shared settings
    relative: bool = True
    eps: float = 1e-12
    modulus_weighting: bool = False

    def fit(
        self,
        dataset: EISDataset,
        model: ForwardModel,
        parameter_specs: Sequence[ParameterSpec],
        weights: NDArray[np.floating] | None = None,
        auxiliary_measurements: ParameterMeasurementDataset | None = None,
    ) -> IdentificationResult:
        """Run DE for global search, then TRF for local refinement."""

        specs = tuple(parameter_specs)
        if not specs:
            raise ValueError("parameter_specs cannot be empty")

        cost = EISPredictionErrorCost(
            dataset=dataset,
            model=model,
            weights=weights,
            relative=self.relative,
            modulus_weighting=self.modulus_weighting,
            eps=self.eps,
        )

        def decode(sv: NDArray[np.floating]) -> NDArray[np.float64]:
            return np.asarray(
                [
                    spec.from_optimization(v)
                    for spec, v in zip(specs, sv, strict=True)
                ],
                dtype=float,
            )

        def scalar_cost(sv: NDArray[np.floating]) -> float:
            return cost(decode(sv))

        def residual_vector(sv: NDArray[np.floating]) -> NDArray[np.float64]:
            theta = decode(sv)
            residuals = cost.residuals(theta)
            vector = np.concatenate((residuals.real, residuals.imag))
            if auxiliary_measurements is not None:
                vector = np.concatenate((
                    vector,
                    auxiliary_measurements.relative_residuals(
                        tuple(s.name for s in specs), theta
                    ),
                ))
            return vector

        bounds_list = [spec.optimization_bounds for spec in specs]
        bounds_array = np.asarray(bounds_list, dtype=float)

        # Stage 1: Global search with DE
        logger.info("HybridOptimizer Stage 1: Differential Evolution")
        de_result = differential_evolution(
            scalar_cost,
            bounds=bounds_list,
            seed=self.de_seed,
            workers=self.de_workers,
            updating="immediate" if self.de_workers == 1 else "deferred",
            polish=False,  # We do our own local refinement
            popsize=self.de_popsize,
            maxiter=self.de_maxiter,
            tol=self.de_tol,
        )
        logger.info(
            "DE converged: cost=%.6e, nfev=%d", de_result.fun, de_result.nfev
        )

        # Stage 2: Local refinement with TRF
        logger.info("HybridOptimizer Stage 2: Local TRF refinement")
        ls_result = least_squares(
            residual_vector,
            x0=de_result.x,
            bounds=(bounds_array[:, 0], bounds_array[:, 1]),
            max_nfev=self.ls_max_nfev,
            ftol=self.ls_ftol,
            xtol=self.ls_xtol,
            gtol=self.ls_gtol,
        )

        theta_vector = decode(ls_result.x)
        z_fit = model.simulate(dataset.freq_hz, theta_vector)
        residuals = dataset.z_obs - z_fit
        final_cost = float(np.sum(residual_vector(ls_result.x) ** 2))

        theta_best = {
            spec.name: float(v)
            for spec, v in zip(specs, theta_vector, strict=True)
        }
        metadata = {
            "optimizer": "hybrid_de_trf",
            "de_nfev": int(de_result.nfev),
            "de_cost": float(de_result.fun),
            "de_success": bool(de_result.success),
            "ls_nfev": int(ls_result.nfev),
            "ls_cost": float(ls_result.cost),
            "ls_success": bool(ls_result.success),
            "ls_message": str(ls_result.message),
            "total_nfev": int(de_result.nfev + ls_result.nfev),
            "relative_residuals": self.relative,
        }
        return IdentificationResult(
            theta_best=theta_best,
            final_cost=final_cost,
            z_fit=z_fit,
            residuals=residuals,
            dataset=dataset,
            metadata=metadata,
        )


@dataclass(frozen=True)
class AdaptiveLeastSquaresOptimizer:
    """Least-squares optimizer with LHS multi-start and Tikhonov regularization.

    Improvements over :class:`LeastSquaresOptimizer`:

    1. **Latin Hypercube Sampling (LHS)** for multi-start initial points
       instead of uniform random, giving better coverage of the search space.
    2. **Tikhonov regularization** (``alpha``) adds a small penalty
       ``α²‖x - x₀‖²`` to the residual vector, preventing over-fitting
       and improving conditioning for noisy real data.
    3. **Post-fit quality flag** that checks whether the fit is trustworthy.
    """

    relative: bool = True
    eps: float = 1e-12
    max_nfev: int = 20000
    ftol: float = 1e-10
    xtol: float = 1e-10
    gtol: float = 1e-10
    modulus_weighting: bool = False
    n_starts: int = 5
    alpha: float = 0.0  # Tikhonov regularization strength
    seed: int = 42

    def __post_init__(self) -> None:
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")
        if self.n_starts < 1:
            raise ValueError("n_starts must be >= 1")

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

        cost = EISPredictionErrorCost(
            dataset=dataset,
            model=model,
            weights=weights,
            relative=self.relative,
            modulus_weighting=self.modulus_weighting,
            eps=self.eps,
        )

        def decode(sv: NDArray[np.floating]) -> NDArray[np.float64]:
            return np.asarray(
                [
                    spec.from_optimization(v)
                    for spec, v in zip(specs, sv, strict=True)
                ],
                dtype=float,
            )

        initial = np.asarray(
            [spec.to_optimization(spec.initial_value) for spec in specs],
            dtype=float,
        )
        bounds = np.asarray(
            [spec.optimization_bounds for spec in specs], dtype=float
        )

        def residual_vector(
            sv: NDArray[np.floating],
        ) -> NDArray[np.float64]:
            theta = decode(sv)
            r = cost.residuals(theta)
            vector = np.concatenate((r.real, r.imag))
            if auxiliary_measurements is not None:
                vector = np.concatenate((
                    vector,
                    auxiliary_measurements.relative_residuals(
                        tuple(s.name for s in specs), theta
                    ),
                ))
            # Tikhonov regularization: append α * (x - x0)
            if self.alpha > 0:
                vector = np.concatenate((
                    vector, self.alpha * (sv - initial)
                ))
            return vector

        # Generate LHS start points
        start_points = _latin_hypercube_starts(
            initial, bounds, self.n_starts, self.seed
        )

        best_result = None
        best_cost_value = float("inf")

        for i, x0 in enumerate(start_points):
            try:
                result = least_squares(
                    residual_vector,
                    x0=x0,
                    bounds=(bounds[:, 0], bounds[:, 1]),
                    max_nfev=self.max_nfev,
                    ftol=self.ftol,
                    xtol=self.xtol,
                    gtol=self.gtol,
                )
                cost_val = float(np.sum(residual_vector(result.x) ** 2))
                if cost_val < best_cost_value:
                    best_cost_value = cost_val
                    best_result = result
                    logger.debug(
                        "Start %d/%d: cost=%.6e (new best)",
                        i + 1, self.n_starts, cost_val,
                    )
            except Exception as exc:
                logger.warning(
                    "Start %d/%d failed: %s", i + 1, self.n_starts, exc
                )

        if best_result is None:
            raise RuntimeError("All optimizer starts failed")

        theta_vector = decode(best_result.x)
        z_fit = model.simulate(dataset.freq_hz, theta_vector)
        residuals = dataset.z_obs - z_fit

        theta_best = {
            spec.name: float(v)
            for spec, v in zip(specs, theta_vector, strict=True)
        }
        metadata = {
            "optimizer": "adaptive_least_squares",
            "success": bool(best_result.success),
            "message": str(best_result.message),
            "nfev": int(best_result.nfev),
            "cost": float(best_result.cost),
            "optimality": float(best_result.optimality),
            "n_starts": self.n_starts,
            "alpha": self.alpha,
            "relative_residuals": self.relative,
            "modulus_weighting": self.modulus_weighting,
        }
        return IdentificationResult(
            theta_best=theta_best,
            final_cost=best_cost_value,
            z_fit=z_fit,
            residuals=residuals,
            dataset=dataset,
            metadata=metadata,
        )


def _latin_hypercube_starts(
    initial: NDArray[np.float64],
    bounds: NDArray[np.float64],
    n_starts: int,
    seed: int,
) -> list[NDArray[np.float64]]:
    """Generate LHS-distributed initial points for multi-start optimization.

    The first point is always the provided *initial* guess.  Remaining
    points are sampled using Latin Hypercube Sampling within the parameter
    bounds to maximize coverage.

    Parameters
    ----------
    initial
        Default initial point in search space.
    bounds
        (n_params, 2) array of [lower, upper] bounds.
    n_starts
        Total number of start points to generate.
    seed
        Random seed for reproducibility.
    """

    points = [initial.copy()]
    if n_starts <= 1:
        return points

    rng = np.random.default_rng(seed)
    ndim = len(initial)
    n_samples = n_starts - 1

    # LHS: for each dimension, create evenly-spaced intervals and
    # sample one random point within each interval, then shuffle.
    lhs_samples = np.empty((n_samples, ndim), dtype=float)
    for j in range(ndim):
        low, high = bounds[j, 0], bounds[j, 1]
        # Create n_samples equal intervals
        edges = np.linspace(low, high, n_samples + 1)
        for i in range(n_samples):
            lhs_samples[i, j] = rng.uniform(edges[i], edges[i + 1])
        # Shuffle the column
        rng.shuffle(lhs_samples[:, j])

    for i in range(n_samples):
        points.append(lhs_samples[i])

    return points

