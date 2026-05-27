"""Freely fit all SEIS parameters using decoupled spectra and scalar calibrations."""

from dataclasses import replace
from pathlib import Path

import numpy as np

from eis_pem import (
    DecoupledStackedSEISModel,
    EISDataset,
    IdentifiabilitySelector,
    LeastSquaresOptimizer,
    all_seis_parameter_specs,
    evaluate_local_identifiability,
    generate_synthetic_dataset,
    generate_synthetic_parameter_measurements,
    save_joint_identification_plots,
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
    specs = all_seis_parameter_specs()
    parameter_names = tuple(spec.name for spec in specs)
    theta_true = np.array([spec.initial_value for spec in specs], dtype=float)
    conditions = tuple(
        (temperature_k, soc)
        for temperature_k in (278.15, 288.15, 298.15, 308.15, 318.15)
        for soc in (0.05, 0.25, 0.50, 0.75, 0.95)
    )
    response_channels = ("neg", "pos", "sep")
    points_per_channel = 60
    base_freq_hz = np.logspace(-3, 5, points_per_channel)
    freq_hz = np.tile(base_freq_hz, len(conditions) * len(response_channels))
    model = DecoupledStackedSEISModel(
        conditions=conditions,
        response_channels=response_channels,
        parameter_names=parameter_names,
    )

    structural_dataset = generate_synthetic_dataset(
        model=model,
        freq_hz=freq_hz,
        theta_true=theta_true,
        noise_level=0.0,
        seed=42,
    )
    eis_only_selection = IdentifiabilitySelector().select(
        structural_dataset, model, specs, theta_true
    )
    measurements = generate_synthetic_parameter_measurements(
        parameter_specs=specs,
        measured_names=eis_only_selection.fixed_names,
        theta_true=theta_true,
        noise_level=0.005,
        seed=142,
        replicates=9,
    )

    generated = generate_synthetic_dataset(
        model=model,
        freq_hz=freq_hz,
        theta_true=theta_true,
        noise_level=0.005,
        seed=42,
    )
    block_conditions = [
        (temperature_k, soc, channel_index)
        for temperature_k, soc in conditions
        for channel_index, _ in enumerate(response_channels)
    ]
    dataset = EISDataset(
        freq_hz=generated.freq_hz,
        z_obs=generated.z_obs,
        z_true=generated.z_true,
        context={
            "temperature_K": np.repeat(
                [item[0] for item in block_conditions], points_per_channel
            ),
            "SOC": np.repeat(
                [item[1] for item in block_conditions], points_per_channel
            ),
            "response_channel_index": np.repeat(
                [item[2] for item in block_conditions], points_per_channel
            ),
        },
    )
    result = LeastSquaresOptimizer(relative=True, max_nfev=5000).fit(
        dataset=dataset,
        model=model,
        parameter_specs=_perturbed_specs(specs),
        auxiliary_measurements=measurements,
    )
    report = evaluate_local_identifiability(
        model=model,
        dataset=structural_dataset,
        parameter_specs=specs,
        theta=theta_true,
        auxiliary_measurements=measurements,
    )

    data_dir = PROJECT_ROOT / "data"
    observations_path = dataset.to_csv(
        data_dir / "joint_decoupled_seis_observations.csv",
        impedance_unit="ohm_m2",
    )
    fit_path = result.export_fit_csv(
        data_dir / "joint_all_seis_fit_result.csv", impedance_unit="ohm_m2"
    )
    parameters_path = result.export_parameters_csv(
        data_dir / "joint_all_seis_parameters.csv",
        parameter_specs=specs,
        theta_true=theta_true,
    )
    measurement_path = measurements.export_csv(
        data_dir / "joint_all_seis_auxiliary_measurements.csv"
    )
    eis_selection_path = eis_only_selection.export_csv(
        data_dir / "joint_decoupled_eis_only_selection.csv"
    )
    report_path = report.export_csv(data_dir / "joint_all_seis_identifiability.csv")
    singular_path = report.export_singular_values_csv(
        data_dir / "joint_all_seis_singular_values.csv"
    )
    plot_paths = save_joint_identification_plots(
        result=result, report=report, output_dir=PROJECT_ROOT / "outputs"
    )

    theta_fit = np.array([result.theta_best[name] for name in parameter_names])
    relative_error = np.abs(theta_fit - theta_true) / theta_true
    worst = np.argsort(relative_error)[::-1][:5]
    print("Joint decoupled SEIS all-parameter identification")
    print(f"Free fitted parameters: {len(specs)}/{len(specs)}")
    print(
        f"Decoupled spectra: {len(conditions)} operating points x "
        f"{len(response_channels)} channels x {points_per_channel} frequencies"
    )
    print("Channel mapping: 0=neg, 1=pos, 2=sep")
    print(
        f"Auxiliary calibration: {len(eis_only_selection.fixed_names)} parameters x "
        f"9 replicate observations = {len(measurements.parameter_names)} readings"
    )
    print(
        "Parameters requiring auxiliary calibration after decoupled-EIS-only diagnosis: "
        + ", ".join(eis_only_selection.fixed_names)
    )
    print(f"Joint Jacobian rank: {report.rank}/{report.parameter_count}")
    print(f"Joint Jacobian condition number: {report.condition_number:.6e}")
    print(f"Joint relative PEM cost: {result.final_cost:.8e}")
    print(
        f"All-parameter relative error: median={np.median(relative_error):.4%}; "
        f"maximum={relative_error.max():.4%}"
    )
    print(
        "Worst errors: "
        + ", ".join(
            f"{parameter_names[index]}={relative_error[index]:.4%}" for index in worst
        )
    )
    print(f"Decoupled observations written: {observations_path}")
    print(f"Fitted data written:            {fit_path}")
    print(f"Parameters written:             {parameters_path}")
    print(f"Auxiliary data written:         {measurement_path}")
    print(f"EIS-only audit written:         {eis_selection_path}")
    print(f"Identifiability written:        {report_path}")
    print(f"Singular values written:        {singular_path}")
    for name, path in plot_paths.items():
        print(f"{name:25s} {path}")


if __name__ == "__main__":
    main()
