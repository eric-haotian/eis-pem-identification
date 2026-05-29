"""JSON-friendly adapter for connecting frontend requests to the DFN model."""
# learning AI website www.haotianblog.com
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .seis_model import (
    SEIS_COMPONENT_CHANNELS,
    SEISComponentModel,
    SEISModel,
    all_seis_parameter_specs,
)


def simulate_dfn_from_frontend(request: Mapping[str, Any]) -> dict[str, Any]:
    """Run the DFN-like forward model from a JSON-style frontend request."""

    if not isinstance(request, Mapping):
        raise ValueError("request must be a mapping")
    freq_hz = _parse_frequency_grid(request)
    conditions = _parse_conditions(request)
    channels = _parse_response_channels(request)
    specs = all_seis_parameter_specs()
    parameter_names = tuple(spec.name for spec in specs)
    parameters = _parameter_values(specs, request.get("parameters", {}))
    theta = np.asarray([parameters[name] for name in parameter_names], dtype=float)

    spectra: list[dict[str, float | str]] = []
    for temperature_k, soc in conditions:
        if channels == ("cell",):
            impedance = SEISModel(
                temperature_k=temperature_k,
                soc=soc,
                parameter_names=parameter_names,
            ).simulate(freq_hz, theta)
            spectra.extend(
                _spectrum_records(freq_hz, impedance, temperature_k, soc, "cell")
            )
            continue

        component_model = SEISComponentModel(
            temperature_k=temperature_k,
            soc=soc,
            response_channel=channels[0],
            parameter_names=parameter_names,
        )
        components = component_model.simulate_components(freq_hz, theta)
        for channel in channels:
            spectra.extend(
                _spectrum_records(
                    freq_hz, components[channel], temperature_k, soc, channel
                )
            )

    return {
        "model": "DFN",
        "impedance_unit": "ohm*m^2",
        "frequency_count": int(freq_hz.size),
        "condition_count": len(conditions),
        "response_channels": list(channels),
        "parameter_count": len(parameter_names),
        "parameters": {name: float(parameters[name]) for name in parameter_names},
        "spectra": spectra,
    }


def dfn_frontend_response_to_frame(response: Mapping[str, Any]) -> pd.DataFrame:
    """Convert a frontend DFN response into a plotting/export dataframe."""

    if "spectra" not in response:
        raise ValueError("response must contain spectra")
    frame = pd.DataFrame(response["spectra"])
    required = {
        "temperature_K",
        "SOC",
        "response_channel",
        "freq_Hz",
        "Zreal_ohm_m2",
        "Zimag_ohm_m2",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"response spectra missing columns: {sorted(missing)}")
    return frame


def _parse_frequency_grid(request: Mapping[str, Any]) -> NDArray[np.float64]:
    if "freq_hz" in request:
        frequencies = np.asarray(request["freq_hz"], dtype=float)
    else:
        config = request.get("frequency", {})
        if not isinstance(config, Mapping):
            raise ValueError("frequency must be a mapping")
        min_hz = float(config.get("min_hz", 1e-3))
        max_hz = float(config.get("max_hz", 1e5))
        points = int(config.get("points", 100))
        if points < 2:
            raise ValueError("frequency points must be at least 2")
        if (
            not np.isfinite([min_hz, max_hz]).all()
            or min_hz <= 0
            or max_hz <= min_hz
        ):
            raise ValueError("frequency min_hz/max_hz must be finite positive bounds")
        frequencies = np.logspace(np.log10(min_hz), np.log10(max_hz), points)
    if frequencies.ndim != 1 or frequencies.size == 0:
        raise ValueError("freq_hz must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0):
        raise ValueError("freq_hz values must be finite and positive")
    return frequencies.astype(float, copy=False)


def _parse_conditions(request: Mapping[str, Any]) -> tuple[tuple[float, float], ...]:
    if "conditions" in request:
        raw_conditions = request["conditions"]
        if not isinstance(raw_conditions, Sequence) or isinstance(
            raw_conditions, (str, bytes)
        ):
            raise ValueError("conditions must be a sequence of mappings")
        conditions = tuple(_parse_condition(condition) for condition in raw_conditions)
    else:
        conditions = (
            (
                float(request.get("temperature_K", 298.15)),
                float(request.get("SOC", 1.0)),
            ),
        )
    if not conditions:
        raise ValueError("conditions cannot be empty")
    return conditions


def _parse_condition(condition: Any) -> tuple[float, float]:
    if not isinstance(condition, Mapping):
        raise ValueError("each condition must be a mapping")
    temperature_k = float(condition.get("temperature_K", 298.15))
    soc = float(condition.get("SOC", 1.0))
    if not np.isfinite(temperature_k) or temperature_k <= 0:
        raise ValueError("temperature_K must be finite and positive")
    if not np.isfinite(soc) or not 0 <= soc <= 1:
        raise ValueError("SOC must be between zero and one")
    return temperature_k, soc


def _parse_response_channels(request: Mapping[str, Any]) -> tuple[str, ...]:
    raw_channels = request.get("response_channels", ("cell",))
    if isinstance(raw_channels, str):
        channels = (raw_channels,)
    else:
        channels = tuple(raw_channels)
    if not channels or len(set(channels)) != len(channels):
        raise ValueError("response_channels must be unique and non-empty")
    unknown = set(channels).difference(SEIS_COMPONENT_CHANNELS)
    if unknown:
        raise ValueError(f"unknown response_channels: {sorted(unknown)}")
    return channels


def _parameter_values(specs: Sequence[Any], overrides: Any) -> dict[str, float]:
    if not isinstance(overrides, Mapping):
        raise ValueError("parameters must be a mapping")
    values = {spec.name: float(spec.initial_value) for spec in specs}
    known = set(values)
    unknown = set(overrides).difference(known)
    if unknown:
        raise ValueError(f"unknown DFN parameter overrides: {sorted(unknown)}")
    bounds = {spec.name: spec.bounds for spec in specs}
    for name, value in overrides.items():
        physical_value = float(value)
        lower, upper = bounds[name]
        if not np.isfinite(physical_value) or not lower <= physical_value <= upper:
            raise ValueError(
                f"DFN parameter {name} must be finite and within [{lower}, {upper}]"
            )
        values[name] = physical_value
    return values


def _spectrum_records(
    freq_hz: NDArray[np.float64],
    impedance: NDArray[np.complex128],
    temperature_k: float,
    soc: float,
    channel: str,
) -> list[dict[str, float | str]]:
    return [
        {
            "temperature_K": float(temperature_k),
            "SOC": float(soc),
            "response_channel": channel,
            "freq_Hz": float(frequency),
            "Zreal_ohm_m2": float(value.real),
            "Zimag_ohm_m2": float(value.imag),
        }
        for frequency, value in zip(freq_hz, impedance, strict=True)
    ]


# ---------------------------------------------------------------------------
# Robust identification frontend
# ---------------------------------------------------------------------------


def identify_parameters_robust(request: Mapping[str, Any]) -> dict[str, Any]:
    """Run the full robust identification pipeline from a JSON-style request.

    This is the primary entry point for real-data parameter identification.
    It chains: data quality → frequency filtering → adaptive identifiability
    selection → multi-start optimization → post-fit diagnostics.

    Parameters
    ----------
    request : Mapping[str, Any]
        JSON-style request with keys:

        **Required**:
        - ``freq_hz``: list of frequencies in Hz
        - ``z_obs_real``: list of Re(Z) observations
        - ``z_obs_imag``: list of Im(Z) observations
        - ``conditions``: list of ``{"temperature_K": ..., "SOC": ...}``

        **Optional**:
        - ``noise_level``: assumed noise (default: auto-estimated)
        - ``max_condition_number``: κ threshold (default: 1e4)
        - ``max_ci95``: CI₉₅ threshold for identifiability (default: 0.15)
        - ``n_starts``: optimizer multi-start count (default: 5)
        - ``alpha``: Tikhonov regularization strength (default: 0.0)
        - ``strategy``: "adaptive" or "classic" (default: "adaptive")
        - ``parameters``: dict of parameter value overrides
        - ``filter_config``: dict of FrequencyBandConfig overrides

    Returns
    -------
    dict[str, Any]
        JSON-serializable result with keys:
        - ``theta_best``: fitted parameter values
        - ``free_parameters``: list of estimated parameter names
        - ``fixed_parameters``: dict of fixed parameter name → value
        - ``final_cost``: optimization cost
        - ``quality_score``: pre-fit data quality score (0–1)
        - ``quality_warnings``: list of quality warning strings
        - ``n_points_used``: points after frequency filtering
        - ``n_points_total``: original point count
        - ``diagnostics``: per-parameter diagnostics (grade, CI₉₅, etc.)
        - ``metadata``: optimizer metadata
    """
    from .data_quality import assess_data_quality
    from .dataset import EISDataset
    from .diagnostics import compute_post_fit_diagnostics
    from .frequency_filter import (
        FrequencyBandAnalyzer,
        FrequencyBandConfig,
        weighted_dataset,
    )
    from .optimizers import AdaptiveLeastSquaresOptimizer
    from .robust import (
        IdentifiabilitySelector,
        IdentifiabilityStrategy,
        ReducedParameterModel,
    )
    from .seis_model import StackedSEISModel

    if not isinstance(request, Mapping):
        raise ValueError("request must be a mapping")

    # --- Parse inputs ---
    freq_hz = np.asarray(request["freq_hz"], dtype=float)
    z_real = np.asarray(request["z_obs_real"], dtype=float)
    z_imag = np.asarray(request["z_obs_imag"], dtype=float)
    z_obs = z_real + 1j * z_imag

    conditions = _parse_conditions(request)
    specs = list(all_seis_parameter_specs())
    parameter_names = tuple(spec.name for spec in specs)
    parameters = _parameter_values(specs, request.get("parameters", {}))
    theta_ref = np.array([parameters[name] for name in parameter_names], dtype=float)

    # --- Build model ---
    model = StackedSEISModel(conditions=conditions, parameter_names=parameter_names)
    dataset = EISDataset(freq_hz=freq_hz, z_obs=z_obs)

    # --- Options ---
    noise_level = request.get("noise_level", None)
    max_condition_number = float(request.get("max_condition_number", 1e4))
    max_ci95 = float(request.get("max_ci95", 0.15))
    n_starts = int(request.get("n_starts", 5))
    alpha = float(request.get("alpha", 0.0))
    strategy_str = str(request.get("strategy", "adaptive")).upper()
    strategy = IdentifiabilityStrategy[strategy_str]

    # --- Step 1: Data quality ---
    quality = assess_data_quality(dataset)
    # --- Step 2: Frequency filtering ---
    filter_overrides = request.get("filter_config", {})
    filter_config = FrequencyBandConfig(**filter_overrides) if filter_overrides else FrequencyBandConfig()
    analyzer = FrequencyBandAnalyzer(config=filter_config)
    filter_result = analyzer.analyze(dataset)
    
    # DO NOT drop points (which breaks multi-condition equal-block constraint).
    # Instead, we use the full dataset but pass the quality weights to the optimizer.
    weights = filter_result.weights

    # --- Step 3: Adaptive identifiability ---
    # Model uses the full dataset structure
    model = StackedSEISModel(
        conditions=conditions,
        parameter_names=parameter_names,
    )

    estimated_noise = noise_level if noise_level is not None else quality.estimated_noise_level
    selector = IdentifiabilitySelector(
        strategy=strategy,
        max_condition_number=max_condition_number,
        assumed_noise_level=estimated_noise,
        max_relative_ci95=max_ci95,
    )
    selection = selector.select(
        dataset=dataset,
        model=model,
        parameter_specs=specs,
        theta_reference=theta_ref,
    )

    # --- Step 4: Optimize ---
    reduced_model = ReducedParameterModel(full_model=model, selection=selection)
    spec_by_name = {s.name: s for s in specs}
    free_specs = [spec_by_name[name] for name in selection.free_names]

    optimizer = AdaptiveLeastSquaresOptimizer(
        relative=True,
        n_starts=n_starts,
        alpha=alpha,
        seed=42,
    )
    result = optimizer.fit(
        dataset=dataset,
        model=reduced_model,
        parameter_specs=free_specs,
        weights=weights,
    )

    # --- Step 5: Diagnostics ---
    # Compute diagnostics over the full grid. 
    # Points with weight=0 shouldn't heavily distort the CI calculation if residuals are appropriately weighted.
    theta_fitted = np.array([result.theta_best[s.name] for s in free_specs])
    diag = compute_post_fit_diagnostics(
        model=reduced_model,
        dataset=dataset,
        parameter_specs=free_specs,
        theta_fitted=theta_fitted,
        relative=True,
    )

    # --- Build response ---
    diagnostics_list = []
    for i, name in enumerate(diag.parameter_names):
        entry: dict[str, Any] = {
            "name": name,
            "value": float(diag.parameter_values[i]),
            "sensitivity": float(diag.sensitivity_norms[i]),
        }
        if diag.identifiability_grades is not None:
            entry["grade"] = diag.identifiability_grades.get(name, "?")
        if diag.parameter_ci95 is not None:
            entry["ci95_relative"] = float(diag.parameter_ci95[i])
        if diag.collinearity_indices is not None:
            entry["collinearity"] = float(diag.collinearity_indices[i])
        diagnostics_list.append(entry)

    return {
        "theta_best": {k: float(v) for k, v in result.theta_best.items()},
        "free_parameters": list(selection.free_names),
        "fixed_parameters": {k: float(v) for k, v in selection.fixed_values.items()},
        "final_cost": float(result.final_cost),
        "quality_score": float(quality.quality_score),
        "quality_warnings": list(quality.warnings),
        "n_points_used": int(np.sum(weights > 0)),
        "n_points_total": int(dataset.freq_hz.size),
        "rank": int(diag.rank),
        "effective_rank": diag.effective_rank,
        "condition_number": float(diag.condition_number),
        "diagnostics": diagnostics_list,
        "metadata": result.metadata,
    }


# ---------------------------------------------------------------------------
# Model selection frontend
# ---------------------------------------------------------------------------


def identify_with_model_selection(request: Mapping[str, Any]) -> dict[str, Any]:
    """Full pipeline: DRT → candidate models → fit → AIC/BIC → best model.

    This is the recommended entry point when the appropriate circuit model
    is unknown. It automatically determines model complexity from the data.

    Parameters
    ----------
    request : Mapping[str, Any]
        JSON-style request with keys:

        **Required**:
        - ``freq_hz``: list of frequencies in Hz
        - ``z_obs_real``: list of Re(Z) observations
        - ``z_obs_imag``: list of Im(Z) observations

        **Optional**:
        - ``candidate_models``: list of model names (default: auto from DRT)
        - ``n_starts``: optimizer multi-start count (default: 3)
        - ``max_nfev``: max function evaluations per model (default: 3000)
        - ``selection_criterion``: ``"aicc"`` (default), ``"aic"``, or ``"bic"``
        - ``filter_config``: dict of FrequencyBandConfig overrides
        - ``priors``: ``"lithium_ion"`` | ``"pemfc"`` | ``"none"`` (default: ``"lithium_ion"``)

    Returns
    -------
    dict[str, Any]
        JSON-serializable result with keys:
        - ``best_model``: name of the best model
        - ``best_circuit``: circuit string description
        - ``theta_best``: best model's fitted parameters
        - ``final_cost``: best model's cost
        - ``quality_score``: pre-fit data quality (0–1)
        - ``drt``: DRT analysis summary
        - ``comparison``: per-model AIC/BIC/weight table
        - ``prior_violations``: list of violated physics priors
    """
    from .data_quality import assess_data_quality
    from .dataset import EISDataset
    from .drt import DRTAnalyzer
    from .ecm_library import all_ecm_models, ecm_model_by_name, suggest_models_from_peaks
    from .frequency_filter import (
        FrequencyBandAnalyzer,
        FrequencyBandConfig,
        weighted_dataset,
    )
    from .model_selection import compare_models
    from .optimizers import LeastSquaresOptimizer
    from .physics_priors import check_prior_violations, lithium_ion_priors, pemfc_priors

    if not isinstance(request, Mapping):
        raise ValueError("request must be a mapping")

    # --- Parse inputs ---
    freq_hz = np.asarray(request["freq_hz"], dtype=float)
    z_real = np.asarray(request["z_obs_real"], dtype=float)
    z_imag = np.asarray(request["z_obs_imag"], dtype=float)
    z_obs = z_real + 1j * z_imag
    dataset = EISDataset(freq_hz=freq_hz, z_obs=z_obs)

    n_starts = int(request.get("n_starts", 3))
    max_nfev = int(request.get("max_nfev", 3000))
    criterion = str(request.get("selection_criterion", "aicc"))
    prior_set = str(request.get("priors", "lithium_ion"))

    # --- Step 1: Data quality ---
    quality = assess_data_quality(dataset)

    # --- Step 2: Frequency filtering ---
    filter_overrides = request.get("filter_config", {})
    filter_config = (
        FrequencyBandConfig(**filter_overrides) if filter_overrides
        else FrequencyBandConfig()
    )
    analyzer = FrequencyBandAnalyzer(config=filter_config)
    filter_result = analyzer.analyze(dataset)
    filtered_dataset, weights = weighted_dataset(dataset, filter_result)

    # --- Step 3: DRT analysis ---
    drt_analyzer = DRTAnalyzer()
    drt_result = drt_analyzer.analyze(filtered_dataset)

    # --- Step 4: Select candidate models ---
    explicit_models = request.get("candidate_models")
    if explicit_models:
        candidates = [ecm_model_by_name(name) for name in explicit_models]
    else:
        candidates = list(suggest_models_from_peaks(
            n_peaks=drt_result.n_peaks,
            has_diffusion_tail=drt_result.has_diffusion_tail,
            has_inductive=drt_result.has_inductive,
        ))
        # Always include at least 2 models for comparison
        if len(candidates) < 2:
            all_models = list(all_ecm_models())
            for m in all_models:
                if m.name not in {c.name for c in candidates}:
                    candidates.append(m)
                    if len(candidates) >= 3:
                        break

    # --- Step 5: Fit and compare ---
    optimizer = LeastSquaresOptimizer(relative=True, max_nfev=max_nfev)
    comparison = compare_models(
        dataset=filtered_dataset,
        candidates=candidates,
        optimizer=optimizer,
        weights=weights,
        relative=True,
    )

    # --- Step 6: Select best ---
    if criterion == "bic":
        best_name = comparison.best_by_bic
    elif criterion == "aic":
        best_name = comparison.best_by_aic
    else:
        best_name = comparison.best_by_aicc

    best_fit = next(c for c in comparison.candidates if c.model_name == best_name)

    # --- Step 7: Physics prior check ---
    if prior_set == "lithium_ion":
        priors = lithium_ion_priors()
    elif prior_set == "pemfc":
        priors = pemfc_priors()
    else:
        priors = ()

    violations: list[str] = []
    if priors:
        best_params = best_fit.identification_result.theta_best
        param_names = list(best_params.keys())
        theta_vals = np.array(list(best_params.values()))
        violations = check_prior_violations(priors, param_names, theta_vals)

    # --- Build response ---
    comparison_table: list[dict[str, Any]] = []
    for c in comparison.candidates:
        comparison_table.append({
            "model": c.model_name,
            "circuit": c.circuit_string,
            "n_params": c.n_params,
            "AICc": float(c.aicc),
            "BIC": float(c.bic),
            "delta_AIC": float(comparison.delta_aic[c.model_name]),
            "akaike_weight": float(comparison.akaike_weights[c.model_name]),
            "durbin_watson": float(c.durbin_watson),
            "ljung_box_p": float(c.ljung_box_p),
            "RSS": float(c.rss),
        })

    return {
        "best_model": best_name,
        "best_circuit": best_fit.circuit_string,
        "theta_best": {k: float(v) for k, v in best_fit.identification_result.theta_best.items()},
        "final_cost": float(best_fit.identification_result.final_cost),
        "quality_score": float(quality.quality_score),
        "quality_warnings": list(quality.warnings),
        "n_points_used": int(filtered_dataset.freq_hz.size),
        "n_points_total": int(dataset.freq_hz.size),
        "drt": {
            "n_peaks": drt_result.n_peaks,
            "peak_frequencies_hz": [
                float(1.0 / (2 * np.pi * t)) for t in drt_result.peak_taus
            ] if drt_result.n_peaks > 0 else [],
            "peak_resistances": [float(r) for r in drt_result.peak_resistances],
            "has_diffusion_tail": drt_result.has_diffusion_tail,
            "has_inductive": drt_result.has_inductive,
            "r_inf": float(drt_result.r_inf),
            "total_polarisation_R": float(drt_result.total_resistance()),
        },
        "comparison": comparison_table,
        "selection_criterion": criterion,
        "prior_violations": violations,
        "metadata": best_fit.identification_result.metadata,
    }
