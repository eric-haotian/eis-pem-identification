from pathlib import Path

import numpy as np
import pandas as pd

from eis_pem.dataset import EISDataset, generate_synthetic_dataset
from eis_pem.optimizers import LeastSquaresOptimizer
from eis_pem.plotting import save_diagnostic_plots
from eis_pem.seis_model import SEISModel, default_seis_parameter_specs, default_seis_theta


def test_synthetic_recovery_exports_required_artifacts(tmp_path: Path) -> None:
    model = SEISModel()
    specs = default_seis_parameter_specs()
    theta_true = default_seis_theta()
    freq_hz = np.logspace(-3, 5, 100)
    dataset = generate_synthetic_dataset(
        model=model,
        freq_hz=freq_hz,
        theta_true=theta_true,
        noise_level=0.005,
        seed=42,
    )
    synthetic_path = tmp_path / "data" / "synthetic_eis.csv"
    dataset.to_csv(synthetic_path, impedance_unit="ohm_m2")

    result = LeastSquaresOptimizer(relative=True, max_nfev=5000, n_starts=5).fit(
        dataset=dataset,
        model=model,
        parameter_specs=specs,
    )

    theta_fit = np.array([result.theta_best[spec.name] for spec in specs])
    relative_error = np.abs(theta_fit - theta_true) / theta_true
    assert np.sum(relative_error < 0.25) >= 3
    assert np.all(relative_error < 1.5)

    fit_path = tmp_path / "data" / "fit_result.csv"
    result.export_fit_csv(fit_path, impedance_unit="ohm_m2")
    figure_paths = save_diagnostic_plots(
        dataset, result, tmp_path / "outputs", impedance_unit="ohm m^2"
    )

    expected_synthetic_columns = [
        "freq_Hz",
        "Zreal_ohm_m2",
        "Zimag_ohm_m2",
        "Zreal_true_ohm_m2",
        "Zimag_true_ohm_m2",
    ]
    expected_fit_columns = [
        "freq_Hz",
        "Zreal_obs_ohm_m2",
        "Zimag_obs_ohm_m2",
        "Zreal_fit_ohm_m2",
        "Zimag_fit_ohm_m2",
        "Zreal_true_ohm_m2",
        "Zimag_true_ohm_m2",
        "residual_real_ohm_m2",
        "residual_imag_ohm_m2",
    ]
    synthetic_frame = pd.read_csv(synthetic_path)
    fit_frame = pd.read_csv(fit_path)
    assert list(synthetic_frame.columns) == expected_synthetic_columns
    assert list(fit_frame.columns) == expected_fit_columns
    assert np.isfinite(synthetic_frame.to_numpy()).all()
    assert np.isfinite(fit_frame.to_numpy()).all()
    assert EISDataset.from_csv(synthetic_path, impedance_unit="ohm_m2").z_true is not None

    assert set(figure_paths) == {
        "nyquist_fit",
        "bode_magnitude_fit",
        "bode_phase_fit",
        "residuals",
    }
    for path in figure_paths.values():
        assert path.exists()
        assert path.stat().st_size > 0
