"""Model selection for EIS equivalent circuit models.

Compares candidate ECM models using information criteria (AIC, BIC, AICc),
residual diagnostics, and F-tests.  The core function :func:`compare_models`
fits each candidate and returns a :class:`ModelComparisonResult` that
identifies the best model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats as _stats

from .dataset import EISDataset
from .results import IdentificationResult

FloatArray = NDArray[np.float64]


# ---------------------------------------------------------------------------
# Per-model fit result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelFitResult:
    """Fit statistics for a single candidate model.

    Attributes
    ----------
    model_name : str
        Short name of the ECM model (e.g. ``"2RCPE"``).
    circuit_string : str
        Human-readable circuit description.
    n_params : int
        Number of free parameters.
    n_observations : int
        Number of real-valued observations (2 × n_freq for complex data).
    rss : float
        Residual sum of squares (real + imaginary).
    log_likelihood : float
        Gaussian log-likelihood assuming i.i.d. errors.
    aic : float
        Akaike Information Criterion.
    aicc : float
        Corrected AIC for finite sample sizes.
    bic : float
        Bayesian Information Criterion.
    durbin_watson : float
        Durbin-Watson statistic for residual autocorrelation (ideal ≈ 2).
    ljung_box_p : float
        Ljung-Box test p-value for residual whiteness (p > 0.05 = white).
    identification_result : IdentificationResult
        Full optimizer result.
    """

    model_name: str
    circuit_string: str
    n_params: int
    n_observations: int
    rss: float
    log_likelihood: float
    aic: float
    aicc: float
    bic: float
    durbin_watson: float
    ljung_box_p: float
    identification_result: IdentificationResult


# ---------------------------------------------------------------------------
# Model comparison result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelComparisonResult:
    """Comparison of multiple candidate models.

    Attributes
    ----------
    candidates : tuple[ModelFitResult, ...]
        Per-model fit results, ordered by AICc.
    best_by_aic : str
        Model name with lowest AIC.
    best_by_bic : str
        Model name with lowest BIC.
    best_by_aicc : str
        Model name with lowest AICc (recommended).
    delta_aic : dict[str, float]
        ΔAIC relative to the best AIC model.
    akaike_weights : dict[str, float]
        Model probabilities from Akaike weights.
    f_test_results : dict[tuple[str, str], float]
        Pairwise F-test p-values for nested model pairs.
    """

    candidates: tuple[ModelFitResult, ...]
    best_by_aic: str
    best_by_bic: str
    best_by_aicc: str
    delta_aic: dict[str, float]
    akaike_weights: dict[str, float]
    f_test_results: dict[tuple[str, str], float]

    def summary(self) -> str:
        """Human-readable comparison summary."""
        lines = [
            "Model Comparison Summary",
            "=" * 70,
            f"  Best by AICc: {self.best_by_aicc}",
            f"  Best by AIC:  {self.best_by_aic}",
            f"  Best by BIC:  {self.best_by_bic}",
            "",
            f"  {'Model':<16} {'k':>4} {'AICc':>12} {'ΔAIC':>10} "
            f"{'Weight':>8} {'BIC':>12} {'D-W':>6} {'L-B p':>7}",
            "  " + "-" * 75,
        ]
        for c in self.candidates:
            lines.append(
                f"  {c.model_name:<16} {c.n_params:4d} "
                f"{c.aicc:12.2f} {self.delta_aic[c.model_name]:10.2f} "
                f"{self.akaike_weights[c.model_name]:8.3f} "
                f"{c.bic:12.2f} {c.durbin_watson:6.2f} "
                f"{c.ljung_box_p:7.3f}"
            )
        if self.f_test_results:
            lines.append("")
            lines.append("  F-test results (nested pairs):")
            for (m1, m2), p in self.f_test_results.items():
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                lines.append(f"    {m1} → {m2}: p = {p:.4f} {sig}")
        return "\n".join(lines)

    def to_frame(self) -> pd.DataFrame:
        """Return a DataFrame with per-model statistics."""
        rows = []
        for c in self.candidates:
            rows.append(
                {
                    "model": c.model_name,
                    "circuit": c.circuit_string,
                    "n_params": c.n_params,
                    "n_obs": c.n_observations,
                    "RSS": c.rss,
                    "log_likelihood": c.log_likelihood,
                    "AIC": c.aic,
                    "AICc": c.aicc,
                    "BIC": c.bic,
                    "delta_AIC": self.delta_aic[c.model_name],
                    "akaike_weight": self.akaike_weights[c.model_name],
                    "durbin_watson": c.durbin_watson,
                    "ljung_box_p": c.ljung_box_p,
                    "final_cost": c.identification_result.final_cost,
                }
            )
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Information criteria
# ---------------------------------------------------------------------------


def compute_aic(n: int, k: int, rss: float) -> float:
    """Akaike Information Criterion: AIC = n·ln(RSS/n) + 2k."""
    if n <= 0 or rss <= 0:
        return np.inf
    return float(n * np.log(rss / n) + 2 * k)


def compute_aicc(n: int, k: int, rss: float) -> float:
    """Corrected AIC for small samples: AICc = AIC + 2k(k+1)/(n-k-1)."""
    aic = compute_aic(n, k, rss)
    if n - k - 1 <= 0:
        return np.inf
    return float(aic + 2 * k * (k + 1) / (n - k - 1))


def compute_bic(n: int, k: int, rss: float) -> float:
    """Bayesian Information Criterion: BIC = n·ln(RSS/n) + k·ln(n)."""
    if n <= 0 or rss <= 0:
        return np.inf
    return float(n * np.log(rss / n) + k * np.log(n))


def compute_log_likelihood(n: int, rss: float) -> float:
    """Gaussian log-likelihood: LL = -n/2 · (1 + ln(2π) + ln(RSS/n))."""
    if n <= 0 or rss <= 0:
        return -np.inf
    return float(-n / 2 * (1 + np.log(2 * np.pi) + np.log(rss / n)))


# ---------------------------------------------------------------------------
# Residual diagnostics
# ---------------------------------------------------------------------------


def durbin_watson(residuals: FloatArray) -> float:
    """Durbin-Watson statistic for autocorrelation (ideal ≈ 2.0).

    D-W < 1.5 suggests positive autocorrelation (systematic model misfit).
    D-W > 2.5 suggests negative autocorrelation.
    """
    if residuals.size < 2:
        return 2.0  # neutral
    d = np.diff(residuals)
    ss_res = np.sum(residuals**2)
    if ss_res < 1e-30:
        return 2.0
    return float(np.sum(d**2) / ss_res)


def ljung_box_p_value(residuals: FloatArray, n_lags: int = 10) -> float:
    """Ljung-Box test p-value for residual whiteness.

    p > 0.05 means residuals appear white (good model fit).
    p < 0.05 means significant autocorrelation (model misfit).
    """
    n = residuals.size
    if n < n_lags + 2:
        n_lags = max(1, n - 2)
    if n < 3:
        return 1.0  # not enough data

    # Compute autocorrelation function
    mean = np.mean(residuals)
    centered = residuals - mean
    var = np.sum(centered**2) / n
    if var < 1e-30:
        return 1.0

    acf = np.zeros(n_lags)
    for lag in range(1, n_lags + 1):
        acf[lag - 1] = np.sum(centered[:-lag] * centered[lag:]) / (n * var)

    # Ljung-Box Q statistic
    q = n * (n + 2) * np.sum(acf**2 / np.arange(n - 1, n - n_lags - 1, -1))
    # p-value from chi-squared distribution
    p = 1.0 - _stats.chi2.cdf(q, df=n_lags)
    return float(p)


# ---------------------------------------------------------------------------
# F-test for nested models
# ---------------------------------------------------------------------------


def f_test_nested(
    rss_simple: float,
    rss_complex: float,
    k_simple: int,
    k_complex: int,
    n: int,
) -> float:
    """F-test p-value for comparing nested models.

    Tests whether the complex model provides a statistically significant
    improvement over the simple model.

    Parameters
    ----------
    rss_simple : float
        RSS of the simpler model (fewer parameters).
    rss_complex : float
        RSS of the more complex model (more parameters).
    k_simple : int
        Number of parameters in the simpler model.
    k_complex : int
        Number of parameters in the more complex model.
    n : int
        Number of observations.

    Returns
    -------
    float
        p-value. Small values (< 0.05) indicate the complex model is
        significantly better.
    """
    if k_complex <= k_simple:
        raise ValueError("Complex model must have more parameters")
    if n <= k_complex:
        return 1.0  # not enough data

    df1 = k_complex - k_simple
    df2 = n - k_complex

    if rss_complex <= 0 or df2 <= 0:
        return 1.0

    f_stat = ((rss_simple - rss_complex) / df1) / (rss_complex / df2)
    if f_stat < 0:
        return 1.0  # complex model is worse (shouldn't happen but be safe)

    p = 1.0 - _stats.f.cdf(f_stat, df1, df2)
    return float(p)


# ---------------------------------------------------------------------------
# Main comparison function
# ---------------------------------------------------------------------------


def compare_models(
    dataset: EISDataset,
    candidates: Sequence[Any],
    optimizer: Any | None = None,
    weights: FloatArray | None = None,
    relative: bool = True,
    max_nfev: int = 5000,
) -> ModelComparisonResult:
    """Fit multiple candidate ECM models and compare with AIC/BIC.

    Parameters
    ----------
    dataset : EISDataset
        Impedance dataset (filtered/weighted).
    candidates : Sequence[ECMModel]
        Candidate models to compare.  Each must implement ``ForwardModel``
        and have ``name``, ``circuit_string``, ``parameter_specs``, and
        ``n_params`` attributes.
    optimizer
        Optimizer instance (default: ``LeastSquaresOptimizer``).
    weights : FloatArray | None
        Optional per-point weights.
    relative : bool
        Whether to use relative residuals.
    max_nfev : int
        Maximum function evaluations per model.

    Returns
    -------
    ModelComparisonResult
        Full comparison with best model selection.
    """
    from .optimizers import LeastSquaresOptimizer

    if optimizer is None:
        optimizer = LeastSquaresOptimizer(relative=relative, max_nfev=max_nfev)

    if not candidates:
        raise ValueError("At least one candidate model is required")

    n_freq = dataset.freq_hz.size
    n_obs = 2 * n_freq  # real + imaginary parts

    fit_results: list[ModelFitResult] = []

    for model in candidates:
        specs = list(model.parameter_specs)
        k = model.n_params

        # Fit the model
        try:
            result = optimizer.fit(
                dataset=dataset,
                model=model,
                parameter_specs=specs,
                weights=weights,
            )
        except Exception:
            # If fitting fails, skip this model
            continue

        # Compute RSS from raw (unweighted) residuals for information criteria
        z_fit = model.simulate(
            dataset.freq_hz,
            np.array([result.theta_best[s.name] for s in specs]),
        )
        raw_residuals = dataset.z_obs - z_fit
        rss = float(np.sum(raw_residuals.real**2 + raw_residuals.imag**2))

        # Flatten residuals for autocorrelation tests (interleave real/imag)
        flat_residuals = np.concatenate([raw_residuals.real, raw_residuals.imag])

        # Compute information criteria
        aic = compute_aic(n_obs, k, rss)
        aicc = compute_aicc(n_obs, k, rss)
        bic = compute_bic(n_obs, k, rss)
        ll = compute_log_likelihood(n_obs, rss)

        # Residual diagnostics
        dw = durbin_watson(flat_residuals)
        lb_p = ljung_box_p_value(flat_residuals)

        fit_results.append(
            ModelFitResult(
                model_name=model.name,
                circuit_string=model.circuit_string,
                n_params=k,
                n_observations=n_obs,
                rss=rss,
                log_likelihood=ll,
                aic=aic,
                aicc=aicc,
                bic=bic,
                durbin_watson=dw,
                ljung_box_p=lb_p,
                identification_result=result,
            )
        )

    if not fit_results:
        raise RuntimeError("All candidate models failed to fit")

    # Sort by AICc
    fit_results.sort(key=lambda r: r.aicc)

    # Find best by each criterion
    best_aic = min(fit_results, key=lambda r: r.aic).model_name
    best_bic = min(fit_results, key=lambda r: r.bic).model_name
    best_aicc = fit_results[0].model_name

    # Delta AIC and Akaike weights
    min_aic = min(r.aic for r in fit_results)
    delta_aic = {r.model_name: r.aic - min_aic for r in fit_results}
    exp_delta = {name: np.exp(-0.5 * d) for name, d in delta_aic.items()}
    sum_exp = sum(exp_delta.values())
    akaike_weights = {
        name: float(w / sum_exp) if sum_exp > 0 else 0.0
        for name, w in exp_delta.items()
    }

    # F-tests for nested model pairs
    # Models are considered nested if the simpler one's circuit elements
    # are a subset. We test all pairs where k_i < k_j.
    f_test_results: dict[tuple[str, str], float] = {}
    for i, r1 in enumerate(fit_results):
        for r2 in fit_results[i + 1 :]:
            if r1.n_params < r2.n_params:
                simple, complex_ = r1, r2
            elif r2.n_params < r1.n_params:
                simple, complex_ = r2, r1
            else:
                continue
            try:
                p = f_test_nested(
                    simple.rss, complex_.rss,
                    simple.n_params, complex_.n_params,
                    n_obs,
                )
                f_test_results[(simple.model_name, complex_.model_name)] = p
            except ValueError:
                pass

    return ModelComparisonResult(
        candidates=tuple(fit_results),
        best_by_aic=best_aic,
        best_by_bic=best_bic,
        best_by_aicc=best_aicc,
        delta_aic=delta_aic,
        akaike_weights=akaike_weights,
        f_test_results=f_test_results,
    )
