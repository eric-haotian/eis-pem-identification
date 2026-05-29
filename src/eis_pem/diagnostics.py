"""Local sensitivity and identifiability diagnostics for PEM fits."""
# learning AI website www.haotianblog.com
from __future__ import annotations

import textwrap
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
    effective_rank: int | None = None
    parameter_ci95: NDArray[np.float64] | None = None
    collinearity_indices: NDArray[np.float64] | None = None
    identifiability_grades: dict[str, str] | None = None

    @property
    def parameter_count(self) -> int:
        return len(self.parameter_names)

    def to_frame(self) -> pd.DataFrame:
        """Return a parameter-indexed summary suitable for CSV output."""

        data: dict[str, object] = {
            "parameter": self.parameter_names,
            "value": self.parameter_values,
            "sensitivity_norm": self.sensitivity_norms,
            "jacobian_rank": self.rank,
            "parameter_count": self.parameter_count,
            "condition_number": self.condition_number,
        }
        if self.effective_rank is not None:
            data["effective_rank"] = self.effective_rank
        if self.parameter_ci95 is not None:
            data["ci95_relative"] = self.parameter_ci95
        if self.collinearity_indices is not None:
            data["collinearity_index"] = self.collinearity_indices
        if self.identifiability_grades is not None:
            data["grade"] = [
                self.identifiability_grades.get(name, "")
                for name in self.parameter_names
            ]
        return pd.DataFrame(data)

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

    def graded_summary(self) -> str:
        """Return human-readable identifiability summary with grades."""

        header = "Parameter Identifiability Report"
        lines = [header, "=" * len(header), ""]

        # Summary line
        lines.append(
            f"Rank: {self.rank} / {self.parameter_count}  "
            f"(effective: {self.effective_rank if self.effective_rank is not None else 'N/A'})  "
            f"Condition: {self.condition_number:.2e}"
        )
        lines.append("")

        # Build table
        col_name = "Parameter"
        col_grade = "Grade"
        col_value = "Value"
        col_sens = "Sensitivity"
        col_ci95 = "CI95 (%)"
        col_collin = "Collinearity"

        rows: list[tuple[str, str, str, str, str, str]] = []
        for i, name in enumerate(self.parameter_names):
            grade = (
                self.identifiability_grades.get(name, "-")
                if self.identifiability_grades is not None
                else "-"
            )
            value = f"{self.parameter_values[i]:.4e}"
            sens = f"{self.sensitivity_norms[i]:.4e}"
            ci = (
                f"{self.parameter_ci95[i] * 100:.1f}"
                if self.parameter_ci95 is not None
                else "-"
            )
            collin = (
                f"{self.collinearity_indices[i]:.3f}"
                if self.collinearity_indices is not None
                else "-"
            )
            rows.append((name, grade, value, sens, ci, collin))

        # Compute column widths
        headers = (col_name, col_grade, col_value, col_sens, col_ci95, col_collin)
        widths = [len(h) for h in headers]
        for row in rows:
            for j, cell in enumerate(row):
                widths[j] = max(widths[j], len(cell))

        fmt = "  ".join(f"{{:<{w}}}" for w in widths)
        lines.append(fmt.format(*headers))
        lines.append("  ".join("-" * w for w in widths))
        for row in rows:
            lines.append(fmt.format(*row))

        return "\n".join(lines)

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


def _compute_effective_rank(
    singular_values: NDArray[np.float64],
    gap_threshold: float = 100.0,
) -> int:
    """Return effective rank based on eigenvalue gap analysis.

    The effective rank is determined by finding the first index where the
    ratio of consecutive singular values exceeds *gap_threshold*.  This is
    more conservative than the numerical rank that uses machine-epsilon
    tolerances, and better reflects the practical number of parameters
    that can be identified from noisy data.

    Parameters
    ----------
    singular_values:
        Singular values from the Jacobian SVD, sorted in descending order.
    gap_threshold:
        Minimum ratio ``σ_i / σ_{i+1}`` considered a significant gap.
        Default is 100.

    Returns
    -------
    int
        Effective rank (1-based).  If no gap exceeds the threshold, the
        full length of *singular_values* is returned.
    """

    if singular_values.size == 0:
        return 0
    if not np.isfinite(gap_threshold) or gap_threshold <= 0:
        raise ValueError("gap_threshold must be finite and positive")

    sv = np.asarray(singular_values, dtype=float)
    for i in range(len(sv) - 1):
        if sv[i + 1] <= 0 or sv[i] / sv[i + 1] > gap_threshold:
            return i + 1
    return len(sv)


def _compute_collinearity_indices(
    jacobian: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return Belsley collinearity index per parameter.

    The collinearity diagnostics follow Belsley, Kuh & Welsch (1980).
    For each parameter, the index is the maximum variance-decomposition
    proportion among all condition indices that exceed 30.

    Parameters
    ----------
    jacobian:
        The (n_obs × n_params) Jacobian matrix.

    Returns
    -------
    NDArray[np.float64]
        Collinearity index per parameter, shape ``(n_params,)``.
        Values near 1.0 indicate severe collinearity.
    """

    n_params = jacobian.shape[1]
    _, s_vals, vt = np.linalg.svd(jacobian, full_matrices=False)

    # Condition indices: κ_j = σ_max / σ_j
    sigma_max = s_vals[0] if s_vals.size else 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        condition_indices = np.where(s_vals > 0, sigma_max / s_vals, np.inf)

    # Variance decomposition proportions
    # V matrix: rows = components (j), cols = parameters (k)
    # v_{jk} are elements of V^T, so vt[j, k]
    v_squared = vt ** 2  # shape (min(n,p), n_params)
    s_squared = s_vals ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        # φ_{jk} = v²_{jk} / σ²_j
        phi = v_squared / s_squared[:, np.newaxis]  # (n_sv, n_params)

    # Normalize: π_{jk} = φ_{jk} / Σ_j φ_{jk}
    phi_sum = phi.sum(axis=0, keepdims=True)  # (1, n_params)
    with np.errstate(divide="ignore", invalid="ignore"):
        proportions = np.where(phi_sum > 0, phi / phi_sum, 0.0)

    # For each parameter, find max proportion where condition index > 30
    high_condition_mask = condition_indices > 30  # (n_sv,)
    collinearity = np.zeros(n_params, dtype=float)
    if np.any(high_condition_mask):
        masked_proportions = proportions[high_condition_mask, :]  # (n_high, n_params)
        collinearity = masked_proportions.max(axis=0)

    return collinearity


def _compute_ci95_from_jacobian(
    jacobian: NDArray[np.float64],
    specs: Sequence[ParameterSpec],
    theta: NDArray[np.floating],
    noise_level: float,
    rcond: float = 1e-14,
) -> NDArray[np.float64]:
    """Return relative 95 %% confidence-interval half-widths per parameter.

    The Fisher information matrix ``F = J^T J`` is inverted to obtain the
    parameter covariance, scaled by ``noise_level²``.  For log-transformed
    parameters the CI in search space is converted back to relative
    physical-space uncertainty via the ``ln(10)`` factor.

    Parameters
    ----------
    jacobian:
        The (n_obs × n_params) Jacobian matrix (in search space).
    specs:
        Parameter specifications, used to detect log transforms.
    theta:
        Physical-space parameter values.
    noise_level:
        Estimated standard deviation of the residuals.
    rcond:
        Cut-off ratio for pseudo-inverse singular values.

    Returns
    -------
    NDArray[np.float64]
        Relative CI95 half-width per parameter (dimensionless).
    """

    fisher = jacobian.T @ jacobian
    cov = noise_level ** 2 * np.linalg.pinv(fisher, rcond=rcond)

    theta_arr = np.asarray(theta, dtype=float)
    n_params = len(specs)
    ci95 = np.empty(n_params, dtype=float)

    for i, spec in enumerate(specs):
        variance = max(cov[i, i], 0.0)
        se = np.sqrt(variance)
        ci_abs = 1.96 * se
        if spec.log_transform:
            # CI in log10 space → relative CI in physical space
            # δ(physical)/physical ≈ ln(10) * δ(log10(physical))
            ci95[i] = np.log(10.0) * ci_abs
        else:
            denom = np.abs(theta_arr[i])
            ci95[i] = ci_abs / denom if denom > 0 else np.inf

    return ci95


def compute_post_fit_diagnostics(
    model: ForwardModel,
    dataset: EISDataset,
    parameter_specs: Sequence[ParameterSpec],
    theta_fitted: NDArray[np.floating],
    residuals: NDArray[np.complex128] | None = None,
    relative: bool = True,
    step_size: float = 1e-5,
    assumed_noise_level: float | None = None,
    auxiliary_measurements: ParameterMeasurementDataset | None = None,
) -> IdentifiabilityReport:
    """Enhanced post-fit diagnostics with noise estimation and grading.

    This function wraps :func:`evaluate_local_identifiability` and augments
    the resulting :class:`IdentifiabilityReport` with:

    * **CI95** – 95 %% confidence-interval half-widths derived from the
      Fisher information matrix.
    * **Effective rank** – a conservative rank estimate based on
      eigenvalue gap analysis.
    * **Collinearity indices** – per-parameter Belsley collinearity
      diagnostics.
    * **Identifiability grades** – A/B/C/D/F letter grades combining
      CI95 and collinearity.

    Parameters
    ----------
    model:
        Forward model used for Jacobian evaluation.
    dataset:
        EIS data (frequencies and observations).
    parameter_specs:
        Specification of each parameter (name, bounds, transform).
    theta_fitted:
        Physical-space parameter values at the fitted optimum.
    residuals:
        Complex residuals ``z_obs - z_model`` at ``theta_fitted``.
        If *None*, they are computed internally.
    relative:
        Whether the Jacobian uses relative (magnitude-scaled) residuals.
    step_size:
        Finite-difference step in search space.
    assumed_noise_level:
        If provided, used directly as σ.  Otherwise σ is estimated from
        *residuals* via ``sqrt(RSS / (n_obs - n_params))``.
    auxiliary_measurements:
        Optional auxiliary scalar observations.

    Returns
    -------
    IdentifiabilityReport
        Enhanced report with all new diagnostic fields populated.
    """

    specs = tuple(parameter_specs)
    theta = np.asarray(theta_fitted, dtype=float)

    # ------------------------------------------------------------------
    # 1. Base report from evaluate_local_identifiability
    # ------------------------------------------------------------------
    base = evaluate_local_identifiability(
        model=model,
        dataset=dataset,
        parameter_specs=specs,
        theta=theta,
        relative=relative,
        step_size=step_size,
        auxiliary_measurements=auxiliary_measurements,
    )

    # ------------------------------------------------------------------
    # 2. Reconstruct the Jacobian (same procedure as base function)
    # ------------------------------------------------------------------
    values = theta.copy()
    search_point = np.asarray(
        [spec.to_optimization(v) for spec, v in zip(specs, values, strict=True)]
    )
    reference = model.simulate(dataset.freq_hz, values)
    scale = (
        np.maximum(np.abs(dataset.z_obs), 1e-12)
        if relative
        else np.ones_like(reference.real)
    )

    def decode(sv: NDArray[np.floating]) -> NDArray[np.float64]:
        return np.asarray(
            [spec.from_optimization(v) for spec, v in zip(specs, sv, strict=True)]
        )

    def prediction_delta(sv: NDArray[np.floating]) -> NDArray[np.float64]:
        physical = decode(sv)
        delta = (model.simulate(dataset.freq_hz, physical) - reference) / scale
        vector = np.concatenate((delta.real, delta.imag))
        if auxiliary_measurements is not None:
            ref_aux = auxiliary_measurements.relative_residuals(
                tuple(s.name for s in specs), values
            )
            pert_aux = auxiliary_measurements.relative_residuals(
                tuple(s.name for s in specs), physical
            )
            vector = np.concatenate((vector, pert_aux - ref_aux))
        return vector

    extra_rows = (
        0
        if auxiliary_measurements is None
        else len(auxiliary_measurements.parameter_names)
    )
    n_freq = dataset.freq_hz.size
    n_params = len(specs)
    jacobian = np.empty((n_freq * 2 + extra_rows, n_params), dtype=float)
    for idx in range(n_params):
        upper = search_point.copy()
        lower = search_point.copy()
        upper[idx] += step_size
        lower[idx] -= step_size
        jacobian[:, idx] = (
            prediction_delta(upper) - prediction_delta(lower)
        ) / (2.0 * step_size)

    # ------------------------------------------------------------------
    # 3. Noise estimation
    # ------------------------------------------------------------------
    if residuals is None:
        z_model = model.simulate(dataset.freq_hz, values)
        residuals_c = dataset.z_obs - z_model
    else:
        residuals_c = np.asarray(residuals, dtype=complex)

    n_obs = 2 * n_freq + extra_rows
    if assumed_noise_level is not None:
        sigma = float(assumed_noise_level)
    else:
        rss = float(np.sum(np.abs(residuals_c) ** 2))
        dof = max(n_obs - n_params, 1)
        sigma = float(np.sqrt(rss / dof))

    # ------------------------------------------------------------------
    # 4. CI95
    # ------------------------------------------------------------------
    ci95 = _compute_ci95_from_jacobian(jacobian, specs, theta, sigma)

    # ------------------------------------------------------------------
    # 5. Effective rank
    # ------------------------------------------------------------------
    effective_rank = _compute_effective_rank(base.singular_values)

    # ------------------------------------------------------------------
    # 6. Collinearity indices
    # ------------------------------------------------------------------
    collinearity = _compute_collinearity_indices(jacobian)

    # ------------------------------------------------------------------
    # 7. Grading
    # ------------------------------------------------------------------
    grades: dict[str, str] = {}
    for i, name in enumerate(base.parameter_names):
        ci_val = ci95[i]
        col_val = collinearity[i]
        if ci_val < 0.05 and col_val < 0.5:
            grade = "A"
        elif ci_val < 0.10 and col_val < 0.7:
            grade = "B"
        elif ci_val < 0.20 and col_val < 0.85:
            grade = "C"
        elif ci_val < 0.50 and col_val < 0.95:
            grade = "D"
        else:
            grade = "F"
        grades[name] = grade

    # ------------------------------------------------------------------
    # 8. Build enhanced report
    # ------------------------------------------------------------------
    return IdentifiabilityReport(
        parameter_names=base.parameter_names,
        parameter_values=base.parameter_values,
        sensitivity_norms=base.sensitivity_norms,
        singular_values=base.singular_values,
        rank=base.rank,
        condition_number=base.condition_number,
        correlation_matrix=base.correlation_matrix,
        effective_rank=effective_rank,
        parameter_ci95=ci95,
        collinearity_indices=collinearity,
        identifiability_grades=grades,
    )
