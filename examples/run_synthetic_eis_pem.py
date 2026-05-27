"""Run the deterministic synthetic Randles-model parameter recovery example."""

from pathlib import Path

import numpy as np

from eis_pem import (
    DifferentialEvolutionOptimizer,
    ParameterSpec,
    RandlesModel,
    generate_synthetic_dataset,
    save_diagnostic_plots,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    freq_hz = np.logspace(-2, 5, 100)
    theta_true = np.array([0.01, 0.05, 2000.0, 0.8, 1e-6])
    model = RandlesModel()
    dataset = generate_synthetic_dataset(
        model=model,
        freq_hz=freq_hz,
        theta_true=theta_true,
        noise_level=0.005,
        seed=42,
    )
    parameter_specs = [
        ParameterSpec("Rs", 0.02, (1e-4, 1e-1), "ohm", log_transform=True),
        ParameterSpec("Rct", 0.1, (1e-4, 1.0), "ohm", log_transform=True),
        ParameterSpec("Qdl", 1000.0, (10.0, 1e5), "F", log_transform=True),
        ParameterSpec("alpha", 0.9, (0.5, 1.0), "-", log_transform=False),
        ParameterSpec("L", 2e-6, (1e-10, 1e-4), "H", log_transform=True),
    ]

    dataset_path = dataset.to_csv(PROJECT_ROOT / "data" / "synthetic_eis.csv")
    result = DifferentialEvolutionOptimizer().fit(
        dataset=dataset,
        model=model,
        parameter_specs=parameter_specs,
    )
    fit_path = result.export_fit_csv(PROJECT_ROOT / "data" / "fit_result.csv")
    plot_paths = save_diagnostic_plots(dataset, result, PROJECT_ROOT / "outputs")

    print("Synthetic EIS PEM identification")
    print(f"Final PEM cost: {result.final_cost:.8e}")
    print("Parameter      True          Identified    Relative error")
    for spec, true_value in zip(parameter_specs, theta_true, strict=True):
        fitted_value = result.theta_best[spec.name]
        relative_error = abs(fitted_value - true_value) / true_value
        print(
            f"{spec.name:<10} {true_value:>12.6g} {fitted_value:>12.6g} "
            f"{relative_error:>14.4%}"
        )
    print(f"Data written: {dataset_path}")
    print(f"Fit written:  {fit_path}")
    for path in plot_paths.values():
        print(f"Plot written: {path}")


if __name__ == "__main__":
    main()
