from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from eis_pem import (
    DecoupledStackedSEISModel,
    IdentifiabilitySelector,
    LeastSquaresOptimizer,
    all_seis_parameter_specs,
    evaluate_local_identifiability,
    generate_synthetic_dataset,
    generate_synthetic_parameter_measurements,
    save_joint_identification_plots,
)


CONDITIONS = tuple(
    (temperature_k, soc)
    for temperature_k in (278.15, 288.15, 298.15, 308.15, 318.15)
    for soc in (0.05, 0.25, 0.50, 0.75, 0.95)
)
CHANNELS = ("neg", "pos", "sep")


def _perturbed_specs(specs):
    result = []
    for index, spec in enumerate(specs):
        lower, upper = spec.bounds
        margin = (upper - lower) * 1e-12
        value = np.clip(
            spec.initial_value * (1.03 if index % 2 == 0 else 0.97),
            lower + margin,
            upper - margin,
        )
        result.append(replace(spec, initial_value=float(value)))
    return result


def test_parameter_measurement_dataset_exports_and_has_zero_true_residual(
    tmp_path: Path,
) -> None:
    specs = all_seis_parameter_specs()
    theta = np.array([spec.initial_value for spec in specs], dtype=float)
    measurements = generate_synthetic_parameter_measurements(
        parameter_specs=specs,
        measured_names=("epse_sep", "epse_neg", "dlnf_ce"),
        theta_true=theta,
        noise_level=0.0,
        seed=42,
    )

    residuals = measurements.relative_residuals(
        tuple(spec.name for spec in specs), theta
    )
    path = measurements.export_csv(tmp_path / "measurements.csv")
    frame = pd.read_csv(path)

    np.testing.assert_allclose(residuals, 0.0)
    assert list(frame["parameter"]) == ["epse_sep", "epse_neg", "dlnf_ce"]
    assert {"observed_value", "true_value", "relative_noise_std"}.issubset(
        frame.columns
    )


def test_joint_decoupled_pem_freely_recovers_all_parameters_with_auxiliary_measurements(
    tmp_path: Path,
) -> None:
    specs = all_seis_parameter_specs()
    names = tuple(spec.name for spec in specs)
    theta_true = np.array([spec.initial_value for spec in specs], dtype=float)
    freq_hz = np.tile(np.logspace(-3, 5, 60), len(CONDITIONS) * len(CHANNELS))
    model = DecoupledStackedSEISModel(
        conditions=CONDITIONS,
        response_channels=CHANNELS,
        parameter_names=names,
    )
    structural_data = generate_synthetic_dataset(
        model, freq_hz, theta_true, noise_level=0.0
    )
    baseline_selection = IdentifiabilitySelector().select(
        structural_data, model, specs, theta_true
    )
    measurements = generate_synthetic_parameter_measurements(
        parameter_specs=specs,
        measured_names=baseline_selection.fixed_names,
        theta_true=theta_true,
        noise_level=0.005,
        seed=142,
        replicates=9,
    )
    dataset = generate_synthetic_dataset(
        model=model,
        freq_hz=freq_hz,
        theta_true=theta_true,
        noise_level=0.005,
        seed=42,
    )

    report = evaluate_local_identifiability(
        model=model,
        dataset=structural_data,
        parameter_specs=specs,
        theta=theta_true,
        auxiliary_measurements=measurements,
    )
    result = LeastSquaresOptimizer(relative=True, max_nfev=5000).fit(
        dataset=dataset,
        model=model,
        parameter_specs=_perturbed_specs(specs),
        auxiliary_measurements=measurements,
    )
    theta_fit = np.array([result.theta_best[name] for name in names])
    relative_error = np.abs(theta_fit - theta_true) / theta_true

    assert len(baseline_selection.fixed_names) > 0
    assert report.condition_number <= 1e4
    assert set(result.theta_best) == set(names)
    assert np.all(relative_error < 0.10)
    parameter_path = result.export_parameters_csv(
        tmp_path / "joint_all_seis_parameters.csv",
        parameter_specs=specs,
        theta_true=theta_true,
    )
    measurement_path = measurements.export_csv(
        tmp_path / "joint_all_seis_auxiliary_measurements.csv"
    )
    report_path = report.export_csv(tmp_path / "joint_all_seis_identifiability.csv")
    singular_values_path = report.export_singular_values_csv(
        tmp_path / "joint_all_seis_singular_values.csv"
    )
    plot_paths = save_joint_identification_plots(
        result=result,
        report=report,
        output_dir=tmp_path / "outputs",
    )

    assert len(pd.read_csv(parameter_path)) == 48
    assert len(pd.read_csv(measurement_path)) == len(baseline_selection.fixed_names) * 9
    assert len(pd.read_csv(report_path)) == 48
    assert singular_values_path.stat().st_size > 0
    assert set(plot_paths) == {"singular_values", "residuals"}
    assert all(path.stat().st_size > 0 for path in plot_paths.values())
