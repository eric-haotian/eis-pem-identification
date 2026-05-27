from pathlib import Path

import numpy as np
import pandas as pd

from eis_pem.dataset import EISDataset, generate_synthetic_dataset
from eis_pem.forward_models import RandlesModel
from eis_pem.optimizers import DifferentialEvolutionOptimizer
from eis_pem.parameters import ParameterSpec
from eis_pem.plotting import save_diagnostic_plots


def randles_parameter_specs() -> list[ParameterSpec]:
    return [
        ParameterSpec("Rs", 0.02, (1e-4, 1e-1), "ohm", log_transform=True),
        ParameterSpec("Rct", 0.1, (1e-4, 1.0), "ohm", log_transform=True),
        ParameterSpec("Qdl", 1000.0, (10.0, 1e5), "F", log_transform=True),
        ParameterSpec("alpha", 0.9, (0.5, 1.0), "-", log_transform=False),
        ParameterSpec("L", 2e-6, (1e-10, 1e-4), "H", log_transform=True),
    ]


def test_synthetic_recovery_exports_required_artifacts(tmp_path: Path) -> None:
    model = RandlesModel()
    theta_true = np.array([0.01, 0.05, 2000.0, 0.8, 1e-6])
    freq_hz = np.logspace(-2, 5, 100)
    dataset = generate_synthetic_dataset(
        model=model,
        freq_hz=freq_hz,
        theta_true=theta_true,
        noise_level=0.005,
        seed=42,
    )
    synthetic_path = tmp_path / "data" / "synthetic_eis.csv"
    dataset.to_csv(synthetic_path)

    result = DifferentialEvolutionOptimizer().fit(
        dataset=dataset,
        model=model,
        parameter_specs=randles_parameter_specs(),
    )

    theta_fit = np.array(
        [result.theta_best[name] for name in ["Rs", "Rct", "Qdl", "alpha", "L"]]
    )
    relative_error = np.abs(theta_fit - theta_true) / theta_true
    assert relative_error[0] < 0.05
    assert relative_error[1] < 0.40
    assert relative_error[2] < 0.40
    assert relative_error[3] < 0.05
    assert relative_error[4] < 0.15

    fit_path = tmp_path / "data" / "fit_result.csv"
    result.export_fit_csv(fit_path)
    figure_paths = save_diagnostic_plots(dataset, result, tmp_path / "outputs")

    expected_synthetic_columns = [
        "freq_Hz",
        "Zreal_ohm",
        "Zimag_ohm",
        "Zreal_true_ohm",
        "Zimag_true_ohm",
    ]
    expected_fit_columns = [
        "freq_Hz",
        "Zreal_obs_ohm",
        "Zimag_obs_ohm",
        "Zreal_fit_ohm",
        "Zimag_fit_ohm",
        "Zreal_true_ohm",
        "Zimag_true_ohm",
        "residual_real_ohm",
        "residual_imag_ohm",
    ]
    synthetic_frame = pd.read_csv(synthetic_path)
    fit_frame = pd.read_csv(fit_path)
    assert list(synthetic_frame.columns) == expected_synthetic_columns
    assert list(fit_frame.columns) == expected_fit_columns
    assert np.isfinite(synthetic_frame.to_numpy()).all()
    assert np.isfinite(fit_frame.to_numpy()).all()
    assert EISDataset.from_csv(synthetic_path).z_true is not None

    assert set(figure_paths) == {
        "nyquist_fit",
        "bode_magnitude_fit",
        "bode_phase_fit",
        "residuals",
    }
    for path in figure_paths.values():
        assert path.exists()
        assert path.stat().st_size > 0
