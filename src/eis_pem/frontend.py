"""JSON-friendly adapter for connecting frontend requests to the DFN model."""

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
