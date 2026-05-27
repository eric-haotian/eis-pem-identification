"""Fit all scalar SEIS physical inputs over a designed synthetic experiment."""

from dataclasses import replace
from pathlib import Path

import numpy as np

from eis_pem import (
    EISDataset,
    LeastSquaresOptimizer,
    StackedSEISModel,
    all_seis_parameter_specs,
    evaluate_local_identifiability,
    generate_synthetic_dataset,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _perturbed_initial_specs(specs):
    fitted_specs = []
    for index, spec in enumerate(specs):
        scale = 1.03 if index % 2 == 0 else 0.97
        value = spec.initial_value * scale
        lower, upper = spec.bounds
        value = float(
            np.clip(
                value, lower + 1e-12 * (upper - lower), upper - 1e-12 * (upper - lower)
            )
        )
        fitted_specs.append(replace(spec, initial_value=value))
    return fitted_specs


def main() -> None:
    true_specs = all_seis_parameter_specs()
    parameter_names = tuple(spec.name for spec in true_specs)
    theta_true = np.array([spec.initial_value for spec in true_specs], dtype=float)
    conditions = tuple(
        (temperature_k, soc)
        for temperature_k in (278.15, 288.15, 298.15, 308.15, 318.15)
        for soc in (0.05, 0.25, 0.50, 0.75, 0.95)
    )
    points_per_condition = 60
    base_freq_hz = np.logspace(-3, 5, points_per_condition)
    freq_hz = np.tile(base_freq_hz, len(conditions))
    model = StackedSEISModel(conditions=conditions, parameter_names=parameter_names)
    generated = generate_synthetic_dataset(
        model=model,
        freq_hz=freq_hz,
        theta_true=theta_true,
        noise_level=0.0,
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
    fitted_specs = _perturbed_initial_specs(true_specs)
    result = LeastSquaresOptimizer(relative=True, max_nfev=5000).fit(
        dataset=dataset,
        model=model,
        parameter_specs=fitted_specs,
    )
    report = evaluate_local_identifiability(
        model=model,
        dataset=dataset,
        parameter_specs=true_specs,
        theta=theta_true,
        relative=True,
    )

    data_dir = PROJECT_ROOT / "data"
    spectrum_path = dataset.to_csv(
        data_dir / "synthetic_all_seis_experiments.csv", impedance_unit="ohm_m2"
    )
    fit_path = result.export_fit_csv(
        data_dir / "all_seis_fit_result.csv", impedance_unit="ohm_m2"
    )
    parameter_path = result.export_parameters_csv(
        data_dir / "all_seis_identified_parameters.csv",
        parameter_specs=fitted_specs,
        theta_true=theta_true,
    )
    identifiability_path = report.export_csv(data_dir / "all_seis_identifiability.csv")
    singular_values_path = report.export_singular_values_csv(
        data_dir / "all_seis_singular_values.csv"
    )

    theta_fit = np.array([result.theta_best[name] for name in parameter_names])
    relative_error = np.abs(theta_fit - theta_true) / theta_true
    print("All SEIS scalar physical parameter identification")
    print(f"Operating points: {len(conditions)} (5 temperatures x 5 SOC values)")
    print(f"Complex EIS points: {freq_hz.size}")
    print("Synthetic noise level: 0 (structural identifiability assessment)")
    print(f"Relative PEM cost: {result.final_cost:.8e}")
    print(
        f"Local Jacobian rank: {report.rank}/{report.parameter_count}; "
        f"condition number: {report.condition_number:.6e}"
    )
    print(
        f"Parameter relative error: median={np.median(relative_error):.4%}; "
        f"maximum={relative_error.max():.4%}"
    )
    print(
        "Caution: near-zero fit residual does not establish unique parameter recovery "
        "when the condition number is large."
    )
    print(f"Synthetic data written:  {spectrum_path}")
    print(f"Fitted data written:     {fit_path}")
    print(f"Parameters written:      {parameter_path}")
    print(f"Identifiability written: {identifiability_path}")
    print(f"Singular values written: {singular_values_path}")


if __name__ == "__main__":
    main()
