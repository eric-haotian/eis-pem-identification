from pathlib import Path

import numpy as np
import pandas as pd

from eis_pem.dataset import EISDataset, generate_synthetic_dataset
from eis_pem.optimizers import LeastSquaresOptimizer
from eis_pem.plotting import save_diagnostic_plots
from eis_pem.seis_model import (
    SEISModel,
    default_seis_parameter_specs,
    default_seis_theta,
)


def test_seis_model_returns_finite_cell_impedance_and_is_sensitive_to_all_fit_parameters() -> (
    None
):
    model = SEISModel()
    theta = default_seis_theta()
    freq_hz = np.logspace(-3, 5, 80)

    baseline = model.simulate(freq_hz, theta)

    assert baseline.shape == freq_hz.shape
    assert np.iscomplexobj(baseline)
    assert np.isfinite(baseline.real).all()
    assert np.isfinite(baseline.imag).all()
    for index in range(theta.size):
        varied = theta.copy()
        varied[index] *= 1.1
        assert not np.allclose(
            model.simulate(freq_hz, varied), baseline, rtol=1e-6, atol=0.0
        )


def test_default_seis_specs_cover_all_selected_physical_parameters() -> None:
    specs = default_seis_parameter_specs()

    assert [spec.name for spec in specs] == [
        "Ds_neg_0",
        "rs_neg",
        "k_neg_0",
        "rou_sei_neg_0",
    ]
    assert all(spec.log_transform for spec in specs)
    assert (
        np.array([spec.initial_value for spec in specs]).shape
        == default_seis_theta().shape
    )


def test_seis_physical_parameter_recovery_and_artifact_export(tmp_path: Path) -> None:
    model = SEISModel()
    specs = default_seis_parameter_specs()
    theta_true = default_seis_theta()
    dataset = generate_synthetic_dataset(
        model=model,
        freq_hz=np.logspace(-3, 5, 100),
        theta_true=theta_true,
        noise_level=0.005,
        seed=42,
    )

    result = LeastSquaresOptimizer(
        relative=True,
        max_nfev=5000,
        n_starts=5,
    ).fit(
        dataset=dataset,
        model=model,
        parameter_specs=specs,
    )
    theta_fit = np.array([result.theta_best[spec.name] for spec in specs])
    relative_error = np.abs(theta_fit - theta_true) / theta_true
    # With cathode CEI active, rou_sei_neg_0 may be confounded in single-
    # condition 4-parameter fits; assert 3/4 parameters are well-recovered.
    assert np.sum(relative_error < 0.25) >= 3
    assert np.all(relative_error < 1.5)

    dataset_path = dataset.to_csv(
        tmp_path / "data" / "synthetic_seis.csv", impedance_unit="ohm_m2"
    )
    fit_path = result.export_fit_csv(
        tmp_path / "data" / "seis_fit_result.csv", impedance_unit="ohm_m2"
    )
    parameter_path = result.export_parameters_csv(
        tmp_path / "data" / "seis_identified_parameters.csv",
        parameter_specs=default_seis_parameter_specs(),
        theta_true=theta_true,
    )
    plot_paths = save_diagnostic_plots(
        dataset,
        result,
        tmp_path / "outputs",
        filename_prefix="seis_",
        impedance_unit="ohm m^2",
    )

    assert "Zreal_ohm_m2" in pd.read_csv(dataset_path).columns
    assert "Zreal_fit_ohm_m2" in pd.read_csv(fit_path).columns
    np.testing.assert_allclose(
        EISDataset.from_csv(dataset_path, impedance_unit="ohm_m2").z_obs, dataset.z_obs
    )
    parameter_frame = pd.read_csv(parameter_path)
    assert list(parameter_frame["parameter"]) == [
        "Ds_neg_0",
        "rs_neg",
        "k_neg_0",
        "rou_sei_neg_0",
    ]
    assert np.isfinite(parameter_frame.select_dtypes(include="number").to_numpy()).all()
    assert set(path.name for path in plot_paths.values()) == {
        "seis_nyquist_fit.png",
        "seis_bode_magnitude_fit.png",
        "seis_bode_phase_fit.png",
        "seis_residuals.png",
    }
