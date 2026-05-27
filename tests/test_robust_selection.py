from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eis_pem import (
    DEFAULT_SELECTED_PARAMETER_NAMES,
    EISDataset,
    IdentificationResult,
    IdentifiabilitySelector,
    LeastSquaresOptimizer,
    ReducedParameterModel,
    StackedSEISModel,
    all_seis_parameter_specs,
    generate_synthetic_dataset,
    save_robust_selection_plots,
)


CONDITIONS = tuple(
    (temperature_k, soc)
    for temperature_k in (278.15, 288.15, 298.15, 308.15, 318.15)
    for soc in (0.05, 0.25, 0.50, 0.75, 0.95)
)


@pytest.fixture(scope="module")
def robust_case():
    specs = all_seis_parameter_specs()
    names = tuple(spec.name for spec in specs)
    theta_true = np.array([spec.initial_value for spec in specs], dtype=float)
    freq_hz = np.tile(np.logspace(-3, 5, 60), len(CONDITIONS))
    model = StackedSEISModel(conditions=CONDITIONS, parameter_names=names)
    generated = generate_synthetic_dataset(
        model=model,
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
            "temperature_K": np.repeat([condition[0] for condition in CONDITIONS], 60),
            "SOC": np.repeat([condition[1] for condition in CONDITIONS], 60),
        },
    )
    selection = IdentifiabilitySelector().select(dataset, model, specs, theta_true)
    return specs, theta_true, freq_hz, model, dataset, selection


def _perturbed_specs(specs):
    fitted_specs = []
    for index, spec in enumerate(specs):
        lower, upper = spec.bounds
        margin = (upper - lower) * 1e-12
        initial = np.clip(
            spec.initial_value * (1.03 if index % 2 == 0 else 0.97),
            lower + margin,
            upper - margin,
        )
        fitted_specs.append(replace(spec, initial_value=float(initial)))
    return fitted_specs


def test_selector_reduces_ill_conditioning_without_fixing_protected_parameters(
    robust_case,
) -> None:
    specs, _, _, _, _, selection = robust_case
    protected_names = set(DEFAULT_SELECTED_PARAMETER_NAMES)

    assert selection.original_condition_number > 1e8
    assert selection.reduced_condition_number <= 1e4
    assert protected_names.issubset(selection.free_names)
    assert protected_names.isdisjoint(selection.fixed_values)
    assert len(selection.free_names) < len(specs)
    assert {"epse_sep", "dlnf_ce"}.intersection(selection.fixed_values)
    for name in selection.free_names:
        if name not in protected_names:
            assert selection.ci95_relative[name] <= 0.10 + 1e-12


def test_reduced_model_expands_free_parameters_to_the_complete_forward_model(
    robust_case,
) -> None:
    specs, theta_true, freq_hz, model, _, selection = robust_case
    reduced_model = ReducedParameterModel(full_model=model, selection=selection)
    physical_by_name = dict(zip((spec.name for spec in specs), theta_true, strict=True))
    theta_free = np.array(
        [physical_by_name[name] for name in selection.free_names], dtype=float
    )

    np.testing.assert_allclose(
        reduced_model.simulate(freq_hz, theta_free),
        model.simulate(freq_hz, theta_true),
        rtol=1e-12,
        atol=1e-18,
    )


def test_selection_and_full_parameter_exports_explain_fixed_parameters(
    robust_case, tmp_path: Path
) -> None:
    specs, theta_true, _, model, dataset, selection = robust_case
    reduced_model = ReducedParameterModel(full_model=model, selection=selection)
    physical_by_name = dict(zip((spec.name for spec in specs), theta_true, strict=True))
    theta_free = np.array(
        [physical_by_name[name] for name in selection.free_names], dtype=float
    )
    z_fit = reduced_model.simulate(dataset.freq_hz, theta_free)
    result = IdentificationResult(
        theta_best={name: physical_by_name[name] for name in selection.free_names},
        final_cost=0.0,
        z_fit=z_fit,
        residuals=dataset.z_obs - z_fit,
        dataset=dataset,
    )

    parameters_path = result.export_parameters_csv(
        tmp_path / "parameters.csv",
        parameter_specs=specs,
        theta_true=theta_true,
        selection=selection,
    )
    selection_path = selection.export_csv(tmp_path / "selection.csv")
    singular_path = selection.export_singular_values_csv(
        tmp_path / "singular_values.csv"
    )

    parameters = pd.read_csv(parameters_path)
    fixed = parameters[parameters["status"] == "fixed_identifiability"]
    assert len(parameters) == 48
    assert {
        "status",
        "reason",
        "reference_value",
        "identified_value",
        "ci95_relative",
    }.issubset(parameters.columns)
    assert not fixed.empty
    assert fixed["relative_error"].isna().all()
    assert (fixed["identified_value"] == fixed["reference_value"]).all()
    assert set(pd.read_csv(selection_path)["parameter"]) == {
        spec.name for spec in specs
    }
    assert set(pd.read_csv(singular_path)["stage"]) == {"full", "selected"}


def test_noisy_robust_fit_recovers_free_parameters_and_writes_diagnostics(
    robust_case, tmp_path: Path
) -> None:
    specs, theta_true, freq_hz, model, _, selection = robust_case
    reduced_model = ReducedParameterModel(full_model=model, selection=selection)
    spec_by_name = {spec.name: spec for spec in specs}
    free_specs = _perturbed_specs([spec_by_name[name] for name in selection.free_names])
    truth_by_name = dict(zip((spec.name for spec in specs), theta_true, strict=True))
    protected_names = set(DEFAULT_SELECTED_PARAMETER_NAMES)
    result_42 = None
    dataset_42 = None

    for seed in (42, 43, 44):
        dataset = generate_synthetic_dataset(
            model=model,
            freq_hz=freq_hz,
            theta_true=theta_true,
            noise_level=0.005,
            seed=seed,
        )
        result = LeastSquaresOptimizer(relative=True, max_nfev=5000).fit(
            dataset=dataset,
            model=reduced_model,
            parameter_specs=free_specs,
        )
        relative_error = np.array(
            [
                abs(result.theta_best[name] - truth_by_name[name]) / truth_by_name[name]
                for name in selection.free_names
            ]
        )

        assert np.all(relative_error < 0.35)
        for name in protected_names:
            position = selection.free_names.index(name)
            assert relative_error[position] < 0.25
        if seed == 42:
            result_42 = result
            dataset_42 = dataset

    assert result_42 is not None
    assert dataset_42 is not None
    fit_path = result_42.export_fit_csv(
        tmp_path / "data" / "robust_all_seis_fit_result.csv",
        impedance_unit="ohm_m2",
    )
    parameters_path = result_42.export_parameters_csv(
        tmp_path / "data" / "robust_all_seis_parameters.csv",
        parameter_specs=specs,
        theta_true=theta_true,
        selection=selection,
    )
    selection_path = selection.export_csv(
        tmp_path / "data" / "robust_all_seis_selection.csv"
    )
    singular_path = selection.export_singular_values_csv(
        tmp_path / "data" / "robust_all_seis_singular_values.csv"
    )
    plot_paths = save_robust_selection_plots(
        result=result_42,
        selection=selection,
        output_dir=tmp_path / "outputs",
        impedance_unit="ohm*m^2",
    )

    assert fit_path.stat().st_size > 0
    assert parameters_path.stat().st_size > 0
    assert selection_path.stat().st_size > 0
    assert singular_path.stat().st_size > 0
    assert set(plot_paths) == {
        "parameter_uncertainty",
        "singular_values",
        "residuals",
    }
    assert all(path.stat().st_size > 0 for path in plot_paths.values())
