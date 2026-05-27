"""Identify all selected SEIS physical parameters from a synthetic EIS spectrum."""

from pathlib import Path

import numpy as np

from eis_pem import (
    DifferentialEvolutionOptimizer,
    SEISModel,
    default_seis_parameter_specs,
    default_seis_theta,
    generate_synthetic_dataset,
    save_diagnostic_plots,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    freq_hz = np.logspace(-3, 5, 100)
    theta_true = default_seis_theta()
    parameter_specs = default_seis_parameter_specs()
    model = SEISModel(temperature_k=298.15, soc=1.0)
    dataset = generate_synthetic_dataset(
        model=model,
        freq_hz=freq_hz,
        theta_true=theta_true,
        noise_level=0.005,
        seed=42,
    )
    dataset_path = dataset.to_csv(
        PROJECT_ROOT / "data" / "synthetic_seis.csv", impedance_unit="ohm_m2"
    )
    result = DifferentialEvolutionOptimizer(relative=True).fit(
        dataset=dataset,
        model=model,
        parameter_specs=parameter_specs,
    )
    fit_path = result.export_fit_csv(
        PROJECT_ROOT / "data" / "seis_fit_result.csv", impedance_unit="ohm_m2"
    )
    parameter_path = result.export_parameters_csv(
        PROJECT_ROOT / "data" / "seis_identified_parameters.csv",
        parameter_specs=parameter_specs,
        theta_true=theta_true,
    )
    plot_paths = save_diagnostic_plots(
        dataset,
        result,
        PROJECT_ROOT / "outputs",
        filename_prefix="seis_",
        impedance_unit="ohm m^2",
    )

    print("SEIS DFN-like physical parameter identification")
    print("Fitted spectrum: Z_cell, temperature=298.15 K, SOC=1.0")
    print(f"Relative PEM cost: {result.final_cost:.8e}")
    print("Parameter               True          Identified    Relative error")
    for spec, true_value in zip(parameter_specs, theta_true, strict=True):
        fitted_value = result.theta_best[spec.name]
        relative_error = abs(fitted_value - true_value) / true_value
        print(
            f"{spec.name:<20} {true_value:>12.6g} {fitted_value:>12.6g} "
            f"{relative_error:>14.4%}"
        )
    print(f"Synthetic data written: {dataset_path}")
    print(f"Fitted data written:    {fit_path}")
    print(f"Parameters written:     {parameter_path}")
    for path in plot_paths.values():
        print(f"Plot written:           {path}")


if __name__ == "__main__":
    main()
