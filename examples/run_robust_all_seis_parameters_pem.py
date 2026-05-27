"""Fit an identifiability-selected SEIS parameter set with noisy synthetic data."""

from dataclasses import replace
from pathlib import Path

import numpy as np

from eis_pem import (
    DEFAULT_SELECTED_PARAMETER_NAMES,
    EISDataset,
    IdentifiabilitySelector,
    LeastSquaresOptimizer,
    ReducedParameterModel,
    StackedSEISModel,
    all_seis_parameter_specs,
    generate_synthetic_dataset,
    save_robust_selection_plots,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _perturbed_specs(specs):
    fitted_specs = []
    for index, spec in enumerate(specs):
        lower, upper = spec.bounds
        margin = (upper - lower) * 1e-12
        value = spec.initial_value * (1.03 if index % 2 == 0 else 0.97)
        value = float(np.clip(value, lower + margin, upper - margin))
        fitted_specs.append(replace(spec, initial_value=value))
    return fitted_specs


def main() -> None:
    all_specs = all_seis_parameter_specs()
    parameter_names = tuple(spec.name for spec in all_specs)
    theta_true = np.array([spec.initial_value for spec in all_specs], dtype=float)
    conditions = tuple(
        (temperature_k, soc)
        for temperature_k in (278.15, 288.15, 298.15, 308.15, 318.15)
        for soc in (0.05, 0.25, 0.50, 0.75, 0.95)
    )
    points_per_condition = 60
    freq_hz = np.tile(np.logspace(-3, 5, points_per_condition), len(conditions))
    full_model = StackedSEISModel(
        conditions=conditions, parameter_names=parameter_names
    )
    generated = generate_synthetic_dataset(
        model=full_model,
        freq_hz=freq_hz,
        theta_true=theta_true,
        noise_level=0.005,
        seed=42,
    )
    dataset = EISDataset(
        freq_hz=generated.freq_hz,
        z_obs=generated.z_obs,
        z_true=generated.z_true,
        context={
            "temperature_K": np.repeat(
                [temperature for temperature, _ in conditions], points_per_condition
            ),
            "SOC": np.repeat([soc for _, soc in conditions], points_per_condition),
        },
    )
    selector = IdentifiabilitySelector()
    selection = selector.select(dataset, full_model, all_specs, theta_true)
    reduced_model = ReducedParameterModel(full_model=full_model, selection=selection)
    spec_by_name = {spec.name: spec for spec in all_specs}
    fitted_specs = _perturbed_specs(
        [spec_by_name[name] for name in selection.free_names]
    )
    result = LeastSquaresOptimizer(relative=True, max_nfev=5000).fit(
        dataset=dataset,
        model=reduced_model,
        parameter_specs=fitted_specs,
    )

    data_dir = PROJECT_ROOT / "data"
    fit_path = result.export_fit_csv(
        data_dir / "robust_all_seis_fit_result.csv", impedance_unit="ohm_m2"
    )
    parameter_path = result.export_parameters_csv(
        data_dir / "robust_all_seis_parameters.csv",
        parameter_specs=all_specs,
        theta_true=theta_true,
        selection=selection,
    )
    selection_path = selection.export_csv(data_dir / "robust_all_seis_selection.csv")
    singular_values_path = selection.export_singular_values_csv(
        data_dir / "robust_all_seis_singular_values.csv"
    )
    plot_paths = save_robust_selection_plots(
        result=result,
        selection=selection,
        output_dir=PROJECT_ROOT / "outputs",
        impedance_unit="ohm*m^2",
    )

    truth_by_name = {
        spec.name: value for spec, value in zip(all_specs, theta_true, strict=True)
    }
    free_error = np.asarray(
        [
            abs(result.theta_best[name] - truth_by_name[name]) / truth_by_name[name]
            for name in selection.free_names
        ]
    )
    print("Robust all-parameter SEIS identification")
    print(f"Operating points: {len(conditions)} (5 temperatures x 5 SOC values)")
    print(f"Complex EIS points: {freq_hz.size}")
    print("Synthetic noise level: 0.5% relative complex Gaussian noise (seed=42)")
    print(
        f"Parameters: {len(parameter_names)} reported, "
        f"{len(selection.free_names)} estimated, {len(selection.fixed_values)} fixed"
    )
    print(
        "Condition number: "
        f"{selection.original_condition_number:.6e} -> "
        f"{selection.reduced_condition_number:.6e}"
    )
    print(f"Relative PEM cost: {result.final_cost:.8e}")
    print(
        f"Estimated parameter relative error: median={np.median(free_error):.4%}; "
        f"maximum={free_error.max():.4%}"
    )
    print("Protected target parameter relative errors:")
    for name in DEFAULT_SELECTED_PARAMETER_NAMES:
        error = abs(result.theta_best[name] - truth_by_name[name]) / truth_by_name[name]
        print(f"  {name:18s} {error:.4%} ({selection.statuses[name]})")
    print(
        "Fixed parameters are reported as nominal assumptions and do not receive "
        "synthetic recovery-error claims."
    )
    print(f"Fitted data written:       {fit_path}")
    print(f"Parameters written:        {parameter_path}")
    print(f"Selection audit written:   {selection_path}")
    print(f"Singular values written:   {singular_values_path}")
    for name, path in plot_paths.items():
        print(f"{name:24s} {path}")


if __name__ == "__main__":
    main()
