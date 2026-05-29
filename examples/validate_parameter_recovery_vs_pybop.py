"""Validation benchmark for SEIS parameter recovery vs. Oxford PyBOP.

This script is intentionally diagnostic rather than promotional. It reports
the initial guess, bounds, fitted value, parameter error, residual error, and
local log-space sensitivity for each model. The two models are not physically
identical, so the output should be read as a reproducible sanity check under
separate synthetic-data closures, not as a proof of real-experiment superiority.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import numpy as np
import pandas as pd

from eis_pem import (
    DecoupledStackedSEISModel,
    IdentifiabilitySelector,
    SEISModel,
    LeastSquaresOptimizer,
    ReducedParameterModel,
    all_seis_parameter_specs,
    default_seis_parameter_specs,
    default_seis_theta,
    evaluate_local_identifiability,
    generate_synthetic_dataset,
)
from eis_pem.dataset import EISDataset
from eis_pem.parameters import ParameterSpec


PYBOP_LOCAL_DEFAULT = Path("~/Projects/pybop-local").expanduser()
PYBOP_CORE_PARAMETER_NAMES = (
    "Positive particle diffusion time scale [s]",
    "Negative particle diffusion time scale [s]",
    "Positive electrode charge transfer time scale [s]",
    "Negative electrode charge transfer time scale [s]",
    "Series resistance [Ohm]",
)
PYBOP_GROUPED_PARAMETER_NAMES = (
    "Minimum negative stoichiometry",
    "Maximum negative stoichiometry",
    "Minimum positive stoichiometry",
    "Maximum positive stoichiometry",
    "Positive particle diffusion time scale [s]",
    "Negative particle diffusion time scale [s]",
    "Positive electrode charge transfer time scale [s]",
    "Negative electrode charge transfer time scale [s]",
    "Positive electrode capacitance [F]",
    "Negative electrode capacitance [F]",
    "Positive electrode relative thickness",
    "Negative electrode relative thickness",
    "Series resistance [Ohm]",
)
PYBOP_INITIAL_FACTORS = (1.20, 0.80, 1.15, 0.85, 1.10, 0.90, 1.12, 0.88)
SEIS_MAX_CONDITIONS = tuple(
    (temperature_k, soc)
    for temperature_k in (278.15, 288.15, 298.15, 308.15, 318.15)
    for soc in (0.05, 0.25, 0.50, 0.75, 0.95)
)
SEIS_MAX_CHANNELS = ("neg", "pos", "sep")


def ensure_pybop_importable(pybop_local: Path) -> None:
    """Re-exec with the user-provided PyBOP virtualenv when needed."""

    try:
        import pybop  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    venv_python = pybop_local / ".venv" / "bin" / "python"
    if os.environ.get("EIS_PEM_SKIP_PYBOP_REEXEC") == "1" or not venv_python.exists():
        raise ModuleNotFoundError(
            "pybop is not importable. Run with the PyBOP environment Python or set "
            f"--pybop-local to a directory containing .venv/bin/python. Tried: {venv_python}"
        )

    env = os.environ.copy()
    env["EIS_PEM_SKIP_PYBOP_REEXEC"] = "1"
    existing_pythonpath = env.get("PYTHONPATH", "")
    pythonpath_parts = [str(SRC_PATH)]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    os.execvpe(str(venv_python), [str(venv_python), *sys.argv], env)


def make_perturbed_seis_specs() -> list[ParameterSpec]:
    """Return SEIS specs with non-truth initial values."""

    factors = {
        "Ds_neg_0": 0.35,
        "rs_neg": 2.50,
        "k_neg_0": 3.00,
        "rou_sei_neg_0": 0.40,
    }
    specs: list[ParameterSpec] = []
    for spec in default_seis_parameter_specs():
        initial = spec.initial_value * factors[spec.name]
        lower, upper = spec.bounds
        initial = float(np.clip(initial, lower * 1.001, upper * 0.999))
        specs.append(replace(spec, initial_value=initial))
    return specs


def make_lightly_perturbed_specs(specs: list[ParameterSpec]) -> list[ParameterSpec]:
    """Return specs with small deterministic offsets from their nominal truth."""

    fitted_specs: list[ParameterSpec] = []
    for index, spec in enumerate(specs):
        lower, upper = spec.bounds
        factor = 1.03 if index % 2 == 0 else 0.97
        value = spec.initial_value * factor
        if lower > 0:
            value = float(np.clip(value, lower * 1.001, upper * 0.999))
        else:
            margin = (upper - lower) * 1e-9
            value = float(np.clip(value, lower + margin, upper - margin))
        fitted_specs.append(replace(spec, initial_value=value))
    return fitted_specs


def run_forward_timer(callback: Any, n_runs: int) -> float:
    """Return average wall time in milliseconds."""

    callback()
    start = time.perf_counter()
    for _ in range(n_runs):
        callback()
    return 1000.0 * (time.perf_counter() - start) / n_runs


def relative_rmse_percent(prediction: np.ndarray, target: np.ndarray) -> float:
    scale = np.maximum(np.abs(target), 1e-12)
    return float(100.0 * np.sqrt(np.mean(np.abs((prediction - target) / scale) ** 2)))


def voltage_rmse_mv(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(1000.0 * np.sqrt(np.mean((prediction - target) ** 2)))


def sensitivity_rank_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    totals = frame.groupby("model")["sensitivity_norm"].transform(
        lambda values: float(np.nansum(values.to_numpy(dtype=float)))
    )
    frame["sensitivity_share_pct"] = np.where(
        totals > 0, 100.0 * frame["sensitivity_norm"] / totals, np.nan
    )
    frame["sensitivity_rank"] = (
        frame.groupby("model")["sensitivity_norm"]
        .rank(method="first", ascending=False, na_option="bottom")
        .astype(int)
    )
    return frame.sort_values(["model", "sensitivity_rank"]).reset_index(drop=True)


def run_seis_four_parameter_case(args: argparse.Namespace) -> dict[str, Any]:
    model = SEISModel()
    specs = make_perturbed_seis_specs()
    theta_true = default_seis_theta()
    freq_hz = np.logspace(
        np.log10(args.freq_min_hz), np.log10(args.freq_max_hz), args.freq_points
    )
    dataset = generate_synthetic_dataset(
        model=model,
        freq_hz=freq_hz,
        theta_true=theta_true,
        noise_level=args.seis_noise_level,
        seed=args.seed,
    )

    optimizer = LeastSquaresOptimizer(
        relative=True,
        n_starts=args.seis_starts,
        max_nfev=args.seis_max_nfev,
    )
    start = time.perf_counter()
    result = optimizer.fit(dataset=dataset, model=model, parameter_specs=specs)
    fit_seconds = time.perf_counter() - start

    theta_fit = np.array([result.theta_best[spec.name] for spec in specs], dtype=float)
    report = evaluate_local_identifiability(
        model=model,
        dataset=dataset,
        parameter_specs=specs,
        theta=theta_fit,
        relative=True,
    )
    forward_ms = run_forward_timer(
        lambda: model.simulate(freq_hz, theta_fit), args.forward_timer_runs
    )

    parameter_rows: list[dict[str, Any]] = []
    for spec, true_value, fit_value, sensitivity in zip(
        specs, theta_true, theta_fit, report.sensitivity_norms, strict=True
    ):
        lower, upper = spec.bounds
        parameter_rows.append(
            {
                "model": "SEISModel",
                "parameter": spec.name,
                "status": "estimated",
                "reason": "four_parameter_sanity_check",
                "unit": spec.unit,
                "initial_value": spec.initial_value,
                "lower_bound": lower,
                "upper_bound": upper,
                "true_value": true_value,
                "estimated_value": fit_value,
                "absolute_error": abs(fit_value - true_value),
                "relative_error_pct": 100.0 * abs(fit_value - true_value) / abs(true_value),
                "sensitivity_norm": sensitivity,
                "at_lower_bound": np.isclose(fit_value, lower, rtol=1e-3, atol=0.0),
                "at_upper_bound": np.isclose(fit_value, upper, rtol=1e-3, atol=0.0),
            }
        )

    fit_frame = pd.DataFrame(
        {
            "model": "SEISModel",
            "domain": "Frequency-domain EIS",
            "x": dataset.freq_hz,
            "x_unit": "Hz",
            "observed_real": dataset.z_obs.real,
            "observed_imag": dataset.z_obs.imag,
            "clean_real": dataset.z_true.real if dataset.z_true is not None else np.nan,
            "clean_imag": dataset.z_true.imag if dataset.z_true is not None else np.nan,
            "fit_real": result.z_fit.real,
            "fit_imag": result.z_fit.imag,
        }
    )

    summary = {
        "model": "SEISModel",
        "toolbox_version": "local eis_pem",
        "domain": "Frequency-domain EIS",
        "parameter_count": len(specs),
        "candidate_parameter_count": len(specs),
        "estimated_parameter_count": len(specs),
        "fixed_parameter_count": 0,
        "data_points": dataset.freq_hz.size,
        "noise": f"{args.seis_noise_level:.4g} relative complex Gaussian",
        "optimizer": f"scipy least_squares, n_starts={args.seis_starts}",
        "fit_seconds": fit_seconds,
        "forward_ms": forward_ms,
        "model_evaluations": result.metadata.get("nfev"),
        "final_cost": result.final_cost,
        "data_rmse": relative_rmse_percent(result.z_fit, dataset.z_obs),
        "data_rmse_unit": "% relative impedance",
        "clean_rmse": relative_rmse_percent(result.z_fit, dataset.z_true),
        "clean_rmse_unit": "% relative impedance",
        "mean_parameter_error_pct": float(
            np.mean([row["relative_error_pct"] for row in parameter_rows])
        ),
        "median_parameter_error_pct": float(
            np.median([row["relative_error_pct"] for row in parameter_rows])
        ),
        "jacobian_rank": report.rank,
        "jacobian_parameter_count": report.parameter_count,
        "jacobian_condition_number": report.condition_number,
        "notes": "Same-model synthetic EIS recovery; not a real-data validation.",
    }
    singular_values = pd.DataFrame(
        {
            "model": "SEISModel",
            "singular_value_index": np.arange(1, report.singular_values.size + 1),
            "singular_value": report.singular_values,
        }
    )
    return {
        "summary": summary,
        "parameters": parameter_rows,
        "fit_frame": fit_frame,
        "singular_values": singular_values,
    }


def run_seis_max_identifiable_case(args: argparse.Namespace) -> dict[str, Any]:
    specs = all_seis_parameter_specs()
    parameter_names = tuple(spec.name for spec in specs)
    theta_true = np.array([spec.initial_value for spec in specs], dtype=float)
    base_freq_hz = np.logspace(
        np.log10(args.freq_min_hz), np.log10(args.freq_max_hz), args.freq_points
    )
    freq_hz = np.tile(base_freq_hz, len(SEIS_MAX_CONDITIONS) * len(SEIS_MAX_CHANNELS))
    full_model = DecoupledStackedSEISModel(
        conditions=SEIS_MAX_CONDITIONS,
        response_channels=SEIS_MAX_CHANNELS,
        parameter_names=parameter_names,
    )
    structural_dataset = generate_synthetic_dataset(
        model=full_model,
        freq_hz=freq_hz,
        theta_true=theta_true,
        noise_level=0.0,
        seed=args.seed,
    )
    selector = IdentifiabilitySelector(protected_names=())
    selection = selector.select(
        structural_dataset,
        full_model,
        specs,
        theta_true,
    )
    reduced_model = ReducedParameterModel(full_model=full_model, selection=selection)
    spec_by_name = {spec.name: spec for spec in specs}
    fitted_specs = make_lightly_perturbed_specs(
        [spec_by_name[name] for name in selection.free_names]
    )

    generated = generate_synthetic_dataset(
        model=full_model,
        freq_hz=freq_hz,
        theta_true=theta_true,
        noise_level=args.seis_noise_level,
        seed=args.seed,
    )
    block_conditions = [
        (temperature_k, soc, channel_index)
        for temperature_k, soc in SEIS_MAX_CONDITIONS
        for channel_index, _ in enumerate(SEIS_MAX_CHANNELS)
    ]
    dataset = EISDataset(
        freq_hz=generated.freq_hz,
        z_obs=generated.z_obs,
        z_true=generated.z_true,
        context={
            "temperature_K": np.repeat(
                [item[0] for item in block_conditions], args.freq_points
            ),
            "SOC": np.repeat([item[1] for item in block_conditions], args.freq_points),
            "response_channel_index": np.repeat(
                [item[2] for item in block_conditions], args.freq_points
            ),
        },
    )

    optimizer = LeastSquaresOptimizer(
        relative=True,
        n_starts=1,
        max_nfev=args.seis_max_nfev,
    )
    start = time.perf_counter()
    result = optimizer.fit(
        dataset=dataset,
        model=reduced_model,
        parameter_specs=fitted_specs,
    )
    fit_seconds = time.perf_counter() - start

    theta_fit_free = np.array(
        [result.theta_best[name] for name in selection.free_names], dtype=float
    )
    theta_fit_full = reduced_model.expand_theta(theta_fit_free)
    report = evaluate_local_identifiability(
        model=reduced_model,
        dataset=dataset,
        parameter_specs=fitted_specs,
        theta=theta_fit_free,
        relative=True,
    )
    sensitivity_by_name = dict(
        zip(selection.free_names, report.sensitivity_norms, strict=True)
    )
    forward_ms = run_forward_timer(
        lambda: full_model.simulate(freq_hz, theta_fit_full), args.forward_timer_runs
    )

    truth_by_name = dict(zip(parameter_names, theta_true, strict=True))
    fit_by_name = dict(zip(parameter_names, theta_fit_full, strict=True))
    initial_by_name = {spec.name: spec.initial_value for spec in specs}
    free_errors: list[float] = []
    parameter_rows: list[dict[str, Any]] = []
    for spec in specs:
        true_value = truth_by_name[spec.name]
        fit_value = fit_by_name[spec.name]
        lower, upper = spec.bounds
        is_estimated = spec.name in selection.free_names
        relative_error = (
            100.0 * abs(fit_value - true_value) / abs(true_value)
            if is_estimated
            else np.nan
        )
        if is_estimated:
            free_errors.append(relative_error)
        parameter_rows.append(
            {
                "model": "SEISModel max-identifiable",
                "parameter": spec.name,
                "status": "estimated" if is_estimated else selection.statuses[spec.name],
                "reason": selection.reasons[spec.name],
                "unit": spec.unit,
                "initial_value": initial_by_name[spec.name],
                "lower_bound": lower,
                "upper_bound": upper,
                "true_value": true_value,
                "estimated_value": fit_value if is_estimated else np.nan,
                "absolute_error": abs(fit_value - true_value) if is_estimated else np.nan,
                "relative_error_pct": relative_error,
                "sensitivity_norm": sensitivity_by_name.get(spec.name, np.nan),
                "at_lower_bound": (
                    np.isclose(fit_value, lower, rtol=1e-3, atol=0.0)
                    if is_estimated
                    else False
                ),
                "at_upper_bound": (
                    np.isclose(fit_value, upper, rtol=1e-3, atol=0.0)
                    if is_estimated
                    else False
                ),
            }
        )

    fit_frame = pd.DataFrame(
        {
            "model": "SEISModel max-identifiable",
            "domain": "Decoupled multi-condition frequency-domain EIS",
            "x": dataset.freq_hz,
            "x_unit": "Hz",
            "observed_real": dataset.z_obs.real,
            "observed_imag": dataset.z_obs.imag,
            "clean_real": dataset.z_true.real if dataset.z_true is not None else np.nan,
            "clean_imag": dataset.z_true.imag if dataset.z_true is not None else np.nan,
            "fit_real": result.z_fit.real,
            "fit_imag": result.z_fit.imag,
        }
    )
    summary = {
        "model": "SEISModel max-identifiable",
        "toolbox_version": "local eis_pem",
        "domain": "Decoupled multi-condition frequency-domain EIS",
        "parameter_count": len(selection.free_names),
        "candidate_parameter_count": len(specs),
        "estimated_parameter_count": len(selection.free_names),
        "fixed_parameter_count": len(selection.fixed_values),
        "data_points": dataset.freq_hz.size,
        "noise": f"{args.seis_noise_level:.4g} relative complex Gaussian",
        "optimizer": "identifiability selector + scipy least_squares",
        "fit_seconds": fit_seconds,
        "forward_ms": forward_ms,
        "model_evaluations": result.metadata.get("nfev"),
        "final_cost": result.final_cost,
        "data_rmse": relative_rmse_percent(result.z_fit, dataset.z_obs),
        "data_rmse_unit": "% relative impedance",
        "clean_rmse": relative_rmse_percent(result.z_fit, dataset.z_true),
        "clean_rmse_unit": "% relative impedance",
        "mean_parameter_error_pct": float(np.mean(free_errors)),
        "median_parameter_error_pct": float(np.median(free_errors)),
        "jacobian_rank": report.rank,
        "jacobian_parameter_count": report.parameter_count,
        "jacobian_condition_number": report.condition_number,
        "notes": (
            f"{len(selection.free_names)}/{len(specs)} selected from decoupled "
            "SEIS spectra; fixed rows are not claimed as identified."
        ),
    }
    singular_values = pd.DataFrame(
        {
            "model": "SEISModel max-identifiable",
            "singular_value_index": np.arange(1, report.singular_values.size + 1),
            "singular_value": report.singular_values,
        }
    )
    selection_frame = selection.to_frame()
    selection_frame.insert(0, "model", "SEISModel max-identifiable")
    return {
        "summary": summary,
        "parameters": parameter_rows,
        "fit_frame": fit_frame,
        "singular_values": singular_values,
        "selection_frame": selection_frame,
    }


def run_seis_case(args: argparse.Namespace) -> dict[str, Any]:
    if args.seis_mode == "four-parameter":
        return run_seis_four_parameter_case(args)
    return run_seis_max_identifiable_case(args)


def pybop_default_value(parameter_values: Any, name: str) -> float:
    value = parameter_values[name]
    try:
        return float(value)
    except TypeError as exc:
        raise TypeError(f"PyBOP default for {name!r} is not numeric: {value!r}") from exc


def make_pybop_parameter_values(
    pybop: Any, model: Any, parameter_names: tuple[str, ...]
) -> tuple[Any, dict[str, float], dict[str, float], dict[str, tuple[float, float]]]:
    parameter_values = model.default_parameter_values.copy()
    true_values = {
        name: pybop_default_value(parameter_values, name) for name in parameter_names
    }
    initial_values: dict[str, float] = {}
    bounds: dict[str, tuple[float, float]] = {}
    for index, name in enumerate(parameter_names):
        factor = PYBOP_INITIAL_FACTORS[index % len(PYBOP_INITIAL_FACTORS)]
        true_value = true_values[name]
        if "stoichiometry" in name or "relative thickness" in name:
            lower = max(1e-4, true_value - 0.2)
            upper = min(0.9999, true_value + 0.2)
        else:
            lower = max(true_value * 0.5, 1e-12)
            upper = true_value * 1.5
        initial = float(np.clip(true_value * factor, lower * 1.001, upper * 0.999))
        initial_values[name] = initial
        bounds[name] = (lower, upper)
        parameter_values.update(
            {
                name: pybop.Parameter(
                    bounds=[lower, upper],
                    initial_value=initial,
                )
            }
        )
    return parameter_values, true_values, initial_values, bounds


def finite_difference_pybop_sensitivity(
    simulate_voltage: Any,
    theta: np.ndarray,
    bounds: list[tuple[float, float]],
    voltage_obs: np.ndarray,
    step_size: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Compute relative-voltage residual sensitivity in log10 parameter space."""

    search_point = np.log10(theta)
    reference = simulate_voltage(theta)
    scale = np.maximum(np.abs(voltage_obs), 1e-12)
    jacobian = np.empty((voltage_obs.size, theta.size), dtype=float)

    for index, (lower, upper) in enumerate(bounds):
        lower_log = np.log10(lower)
        upper_log = np.log10(upper)
        up = search_point.copy()
        down = search_point.copy()
        up[index] = min(up[index] + step_size, upper_log)
        down[index] = max(down[index] - step_size, lower_log)
        denominator = up[index] - down[index]
        if denominator <= 0:
            jacobian[:, index] = np.nan
            continue
        up_theta = np.power(10.0, up)
        down_theta = np.power(10.0, down)
        up_delta = (simulate_voltage(up_theta) - reference) / scale
        down_delta = (simulate_voltage(down_theta) - reference) / scale
        jacobian[:, index] = (up_delta - down_delta) / denominator

    finite = np.isfinite(jacobian).all(axis=0)
    usable = jacobian[:, finite]
    if usable.size:
        singular_values = np.linalg.svd(usable, compute_uv=False)
        tolerance = singular_values[0] * max(usable.shape) * np.finfo(float).eps
        rank = int(np.sum(singular_values > tolerance))
        condition_number = (
            float(singular_values[0] / singular_values[-1])
            if singular_values[-1] > 0
            else float("inf")
        )
    else:
        singular_values = np.array([], dtype=float)
        rank = 0
        condition_number = float("inf")
    return np.linalg.norm(jacobian, axis=0), singular_values, rank, condition_number


def run_pybop_case(args: argparse.Namespace) -> dict[str, Any]:
    import pybop

    requested_parameter_names = (
        PYBOP_CORE_PARAMETER_NAMES
        if args.pybop_mode == "core5"
        else PYBOP_GROUPED_PARAMETER_NAMES
    )
    model = pybop.lithium_ion.GroupedSPM()
    parameter_values, true_values, initial_values, bounds_map = make_pybop_parameter_values(
        pybop, model, requested_parameter_names
    )
    t_eval = np.arange(0.0, args.pybop_duration_s, args.pybop_dt_s)
    simulator = pybop.pybamm.Simulator(
        model=model,
        parameter_values=parameter_values,
        protocol=t_eval,
        output_variables=["Voltage [V]"],
    )
    true_inputs = {name: true_values[name] for name in requested_parameter_names}
    clean_solution = simulator.solve(inputs=true_inputs)
    voltage_clean = np.asarray(clean_solution["Voltage [V]"](t_eval), dtype=float)
    rng = np.random.default_rng(args.seed)
    voltage_obs = voltage_clean + rng.normal(0.0, args.pybop_voltage_noise_v, t_eval.size)

    dataset = pybop.Dataset({"Time [s]": t_eval, "Voltage [V]": voltage_obs})
    cost = pybop.SumSquaredError(dataset, target="Voltage [V]")
    problem = pybop.Problem(simulator, cost)
    np.random.seed(args.seed)
    optimizer = pybop.CMAES(
        problem,
        options=pybop.PintsOptions(
            max_iterations=args.pybop_iterations,
            max_unchanged_iterations=args.pybop_iterations,
            absolute_tolerance=1e-12,
            relative_tolerance=1e-12,
            verbose=False,
        ),
    )

    start = time.perf_counter()
    result = optimizer.run()
    fit_seconds = time.perf_counter() - start
    theta_fit = np.asarray(result.x, dtype=float)
    parameter_names = tuple(problem.parameters.names)

    def simulate_voltage(theta: np.ndarray) -> np.ndarray:
        inputs = {
            name: float(value)
            for name, value in zip(parameter_names, theta, strict=True)
        }
        solution = simulator.solve(inputs=inputs)
        return np.asarray(solution["Voltage [V]"](t_eval), dtype=float)

    voltage_fit = simulate_voltage(theta_fit)
    bounds = [bounds_map[name] for name in parameter_names]
    sensitivity_norms, singular_values, rank, condition_number = (
        finite_difference_pybop_sensitivity(
            simulate_voltage=simulate_voltage,
            theta=theta_fit,
            bounds=bounds,
            voltage_obs=voltage_obs,
        )
    )
    forward_ms = run_forward_timer(lambda: simulate_voltage(theta_fit), args.pybop_timer_runs)

    parameter_rows: list[dict[str, Any]] = []
    for name, fit_value, sensitivity in zip(
        parameter_names, theta_fit, sensitivity_norms, strict=True
    ):
        true_value = true_values[name]
        lower, upper = bounds_map[name]
        status = "estimated" if sensitivity > 1e-10 else "fitted_zero_sensitivity"
        parameter_rows.append(
            {
                "model": "Oxford PyBOP GroupedSPM",
                "parameter": name,
                "status": status,
                "reason": f"{args.pybop_mode}_synthetic_grouped_spm_parameter",
                "unit": infer_pybop_unit(name),
                "initial_value": initial_values[name],
                "lower_bound": lower,
                "upper_bound": upper,
                "true_value": true_value,
                "estimated_value": fit_value,
                "absolute_error": abs(fit_value - true_value),
                "relative_error_pct": 100.0 * abs(fit_value - true_value) / abs(true_value),
                "sensitivity_norm": sensitivity,
                "at_lower_bound": np.isclose(fit_value, lower, rtol=1e-3, atol=0.0),
                "at_upper_bound": np.isclose(fit_value, upper, rtol=1e-3, atol=0.0),
            }
        )

    fit_frame = pd.DataFrame(
        {
            "model": "Oxford PyBOP GroupedSPM",
            "domain": "Time-domain voltage",
            "x": t_eval,
            "x_unit": "s",
            "observed_real": voltage_obs,
            "observed_imag": np.nan,
            "clean_real": voltage_clean,
            "clean_imag": np.nan,
            "fit_real": voltage_fit,
            "fit_imag": np.nan,
        }
    )
    pybop_version = getattr(pybop, "__version__", "unknown")
    summary = {
        "model": "Oxford PyBOP GroupedSPM",
        "toolbox_version": f"pybop {pybop_version}",
        "domain": "Time-domain voltage",
        "parameter_count": len(parameter_names),
        "candidate_parameter_count": len(parameter_names),
        "estimated_parameter_count": int(np.sum(sensitivity_norms > 1e-10)),
        "fixed_parameter_count": 0,
        "data_points": t_eval.size,
        "noise": f"{args.pybop_voltage_noise_v:.4g} V Gaussian",
        "optimizer": f"PyBOP CMAES, max_iterations={args.pybop_iterations}",
        "fit_seconds": fit_seconds,
        "forward_ms": forward_ms,
        "model_evaluations": result.n_evaluations,
        "final_cost": result.best_cost,
        "data_rmse": voltage_rmse_mv(voltage_fit, voltage_obs),
        "data_rmse_unit": "mV",
        "clean_rmse": voltage_rmse_mv(voltage_fit, voltage_clean),
        "clean_rmse_unit": "mV",
        "mean_parameter_error_pct": float(
            np.mean([row["relative_error_pct"] for row in parameter_rows])
        ),
        "median_parameter_error_pct": float(
            np.median([row["relative_error_pct"] for row in parameter_rows])
        ),
        "jacobian_rank": rank,
        "jacobian_parameter_count": len(parameter_names),
        "jacobian_condition_number": condition_number,
        "notes": "Same-model synthetic voltage recovery using local ~/Projects/pybop-local environment.",
    }
    singular_values_frame = pd.DataFrame(
        {
            "model": "Oxford PyBOP GroupedSPM",
            "singular_value_index": np.arange(1, singular_values.size + 1),
            "singular_value": singular_values,
        }
    )
    return {
        "summary": summary,
        "parameters": parameter_rows,
        "fit_frame": fit_frame,
        "singular_values": singular_values_frame,
    }


def infer_pybop_unit(parameter_name: str) -> str:
    start = parameter_name.rfind("[")
    end = parameter_name.rfind("]")
    if start != -1 and end > start:
        return parameter_name[start + 1 : end]
    return "-"


def write_outputs(
    output_dir: Path,
    seis_case: dict[str, Any],
    pybop_case: dict[str, Any],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([seis_case["summary"], pybop_case["summary"]])
    parameters = sensitivity_rank_frame(
        pd.DataFrame([*seis_case["parameters"], *pybop_case["parameters"]])
    )
    observations = pd.concat(
        [seis_case["fit_frame"], pybop_case["fit_frame"]],
        ignore_index=True,
    )
    singular_values = pd.concat(
        [seis_case["singular_values"], pybop_case["singular_values"]],
        ignore_index=True,
    )

    paths = {
        "summary": output_dir / "validation_summary_vs_pybop.csv",
        "parameters": output_dir / "validation_parameter_recovery_vs_pybop.csv",
        "observations": output_dir / "validation_fit_traces_vs_pybop.csv",
        "singular_values": output_dir / "validation_singular_values_vs_pybop.csv",
    }
    if "selection_frame" in seis_case:
        paths["seis_selection"] = output_dir / "validation_seis_selection_vs_pybop.csv"
    summary.to_csv(paths["summary"], index=False)
    parameters.to_csv(paths["parameters"], index=False)
    observations.to_csv(paths["observations"], index=False)
    singular_values.to_csv(paths["singular_values"], index=False)
    if "selection_frame" in seis_case:
        seis_case["selection_frame"].to_csv(paths["seis_selection"], index=False)
    return paths


def print_report(paths: dict[str, Path]) -> None:
    summary = pd.read_csv(paths["summary"])
    parameters = pd.read_csv(paths["parameters"])

    print("\n================ VALIDATION SUMMARY ================")
    print(summary.to_markdown(index=False))
    print("\n================ PARAMETER RECOVERY ================")
    shown_parameters = parameters[parameters["status"] != "fixed_identifiability"].copy()
    columns = [
        "model",
        "parameter",
        "status",
        "initial_value",
        "true_value",
        "estimated_value",
        "relative_error_pct",
        "sensitivity_norm",
        "sensitivity_rank",
        "at_lower_bound",
        "at_upper_bound",
    ]
    print(shown_parameters[columns].to_markdown(index=False))
    print("\nOutput files:")
    for label, path in paths.items():
        print(f"- {label}: {path.relative_to(PROJECT_ROOT)}")
    print(
        "\nCaution: both recoveries use model-generated synthetic data. Use these CSVs "
        "to check numerical behavior and identifiability, not as a real-cell accuracy claim."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare SEIS and local Oxford PyBOP parameter recovery with sensitivity diagnostics."
    )
    parser.add_argument("--pybop-local", type=Path, default=PYBOP_LOCAL_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seis-mode",
        choices=("max-identifiable", "four-parameter"),
        default="max-identifiable",
    )
    parser.add_argument(
        "--pybop-mode",
        choices=("grouped13", "core5"),
        default="grouped13",
    )
    parser.add_argument("--freq-min-hz", type=float, default=1e-3)
    parser.add_argument("--freq-max-hz", type=float, default=1e5)
    parser.add_argument("--freq-points", type=int, default=60)
    parser.add_argument("--seis-noise-level", type=float, default=0.005)
    parser.add_argument("--seis-starts", type=int, default=6)
    parser.add_argument("--seis-max-nfev", type=int, default=1000)
    parser.add_argument("--pybop-voltage-noise-v", type=float, default=0.001)
    parser.add_argument("--pybop-duration-s", type=float, default=900.0)
    parser.add_argument("--pybop-dt-s", type=float, default=10.0)
    parser.add_argument("--pybop-iterations", type=int, default=60)
    parser.add_argument("--forward-timer-runs", type=int, default=20)
    parser.add_argument("--pybop-timer-runs", type=int, default=20)
    args = parser.parse_args()
    if args.freq_min_hz <= 0 or args.freq_max_hz <= args.freq_min_hz:
        parser.error("--freq-min-hz must be positive and below --freq-max-hz")
    if args.freq_points < 2:
        parser.error("--freq-points must be at least 2")
    if args.pybop_dt_s <= 0 or args.pybop_duration_s <= args.pybop_dt_s:
        parser.error("--pybop-dt-s must be positive and below --pybop-duration-s")
    return args


def main() -> None:
    args = parse_args()
    args.pybop_local = args.pybop_local.expanduser()
    ensure_pybop_importable(args.pybop_local)

    print("Running SEISModel validation case...")
    seis_case = run_seis_case(args)
    print("Running Oxford PyBOP validation case...")
    pybop_case = run_pybop_case(args)
    paths = write_outputs(args.output_dir, seis_case, pybop_case)
    print_report(paths)


if __name__ == "__main__":
    main()
