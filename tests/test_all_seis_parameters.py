import numpy as np
from dataclasses import replace
from pathlib import Path

from eis_pem.costs import EISPredictionErrorCost
from eis_pem.dataset import EISDataset, generate_synthetic_dataset
from eis_pem.diagnostics import evaluate_local_identifiability
from eis_pem.optimizers import LeastSquaresOptimizer
from eis_pem.seis_model import (
    SEISModel,
    StackedSEISModel,
    all_seis_parameter_specs,
    default_seis_theta,
    default_seis_parameter_specs,
)


EXPECTED_ALL_PARAMETER_NAMES = [
    "alpha_a_neg",
    "alpha_a_pos",
    "sigma_neg",
    "sigma_pos",
    "epse_neg",
    "epse_pos",
    "epse_sep",
    "epsf_neg",
    "epsf_pos",
    "brug_neg",
    "brug_pos",
    "brug_sep",
    "Ds_neg_0",
    "Ds_pos_0",
    "rs_neg",
    "rs_pos",
    "L_neg",
    "L_pos",
    "L_sep",
    "k_neg_0",
    "k_pos_0",
    "Cdl_neg",
    "Cdl_pos",
    "alpha_dl_neg",
    "alpha_dl_pos",
    "rou_sei_neg_0",
    "epse_sei_neg",
    "rou_sei_pos_0",
    "epse_sei_pos",
    "kappa_0",
    "dlnf_ce",
    "t_plus",
    "ce_0",
    "s0_neg",
    "s100_neg",
    "s0_pos",
    "s100_pos",
    "cs_max_neg",
    "cs_max_pos",
    "Ea_Ds_neg",
    "Ea_Ds_pos",
    "Ea_k_neg",
    "Ea_k_pos",
    "Ea_De",
    "Ea_kappa",
    "Ea_rou_sei",
    "R_contact",
    "L_ind",
]


def test_all_seis_parameter_specs_expose_all_independent_numeric_inputs() -> None:
    specs = all_seis_parameter_specs()

    assert [spec.name for spec in specs] == EXPECTED_ALL_PARAMETER_NAMES
    assert len(specs) == 48
    assert {spec.name for spec in default_seis_parameter_specs()}.issubset(
        {spec.name for spec in specs}
    )


def test_all_parameter_default_vector_reproduces_four_parameter_default_model() -> None:
    all_specs = all_seis_parameter_specs()
    all_model = SEISModel(parameter_names=tuple(spec.name for spec in all_specs))
    reference_model = SEISModel()
    freq_hz = np.logspace(-3, 5, 50)

    z_all = all_model.simulate(
        freq_hz, np.array([spec.initial_value for spec in all_specs], dtype=float)
    )
    z_reference = reference_model.simulate(freq_hz, default_seis_theta())

    np.testing.assert_allclose(z_all, z_reference, rtol=1e-12, atol=1e-18)


def test_stacked_model_runs_multiple_operating_conditions_through_same_parameters() -> (
    None
):
    specs = default_seis_parameter_specs()
    parameter_names = tuple(spec.name for spec in specs)
    stacked = StackedSEISModel(
        conditions=((278.15, 0.2), (298.15, 0.5), (318.15, 0.8)),
        parameter_names=parameter_names,
    )
    segment_freq_hz = np.logspace(-3, 5, 20)
    stacked_freq_hz = np.tile(segment_freq_hz, len(stacked.conditions))
    theta = default_seis_theta()

    spectrum = stacked.simulate(stacked_freq_hz, theta)
    expected = np.concatenate(
        [
            SEISModel(temperature_k=temperature, soc=soc).simulate(
                segment_freq_hz, theta
            )
            for temperature, soc in stacked.conditions
        ]
    )

    np.testing.assert_allclose(spectrum, expected)


def test_all_parameter_multicondition_workflow_reduces_cost_and_reports_rank(
    tmp_path: Path,
) -> None:
    specs = all_seis_parameter_specs()
    parameter_names = tuple(spec.name for spec in specs)
    conditions = tuple(
        (temperature, soc)
        for temperature in (278.15, 298.15, 318.15)
        for soc in (0.1, 0.5, 0.9)
    )
    model = StackedSEISModel(conditions=conditions, parameter_names=parameter_names)
    segment_freq = np.logspace(-3, 5, 40)
    freq_hz = np.tile(segment_freq, len(conditions))
    theta_true = np.array([spec.initial_value for spec in specs])
    generated = generate_synthetic_dataset(model, freq_hz, theta_true, noise_level=0.0)
    dataset = EISDataset(
        freq_hz=generated.freq_hz,
        z_obs=generated.z_obs,
        z_true=generated.z_true,
        context={
            "temperature_K": np.repeat([c[0] for c in conditions], segment_freq.size),
            "SOC": np.repeat([c[1] for c in conditions], segment_freq.size),
        },
    )
    fitted_specs = []
    for i, spec in enumerate(specs):
        scale = 1.03 if i % 2 == 0 else 0.97
        value = spec.initial_value * scale
        lower, upper = spec.bounds
        value = float(
            np.clip(
                value, lower + 1e-12 * (upper - lower), upper - 1e-12 * (upper - lower)
            )
        )
        fitted_specs.append(replace(spec, initial_value=value))
    cost = EISPredictionErrorCost(dataset, model, relative=True)
    initial_theta = np.array([spec.initial_value for spec in fitted_specs])

    result = LeastSquaresOptimizer(relative=True, max_nfev=2000).fit(
        dataset, model, fitted_specs
    )
    report = evaluate_local_identifiability(model, dataset, specs, theta_true)

    assert result.final_cost < cost(initial_theta) * 1e-8
    assert set(result.theta_best) == set(EXPECTED_ALL_PARAMETER_NAMES)
    assert report.parameter_count == len(specs)
    assert report.rank <= report.parameter_count
    assert np.isfinite(report.condition_number)
    spectrum_path = dataset.to_csv(
        tmp_path / "all_synthetic.csv", impedance_unit="ohm_m2"
    )
    fit_path = result.export_fit_csv(tmp_path / "all_fit.csv", impedance_unit="ohm_m2")
    report_path = report.export_csv(tmp_path / "all_identifiability.csv")
    reloaded = EISDataset.from_csv(
        spectrum_path,
        impedance_unit="ohm_m2",
        context_columns=("temperature_K", "SOC"),
    )
    assert reloaded.context is not None
    assert (
        "temperature_K"
        in np.genfromtxt(fit_path, delimiter=",", max_rows=1, dtype=str).tolist()
    )
    assert report_path.exists()
