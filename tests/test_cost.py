import numpy as np
import pytest

from eis_pem.costs import EISPredictionErrorCost
from eis_pem.dataset import EISDataset
from eis_pem.forward_models import RandlesModel


def test_noiseless_true_parameters_have_zero_prediction_error() -> None:
    model = RandlesModel()
    theta_true = np.array([0.01, 0.05, 2000.0, 0.8, 1e-6])
    freq_hz = np.logspace(-2, 5, 20)
    z_true = model.simulate(freq_hz, theta_true)
    dataset = EISDataset(freq_hz=freq_hz, z_obs=z_true, z_true=z_true)

    cost = EISPredictionErrorCost(dataset=dataset, model=model)

    assert cost(theta_true) == pytest.approx(0.0, abs=1e-18)
    assert cost(np.array([0.02, 0.05, 2000.0, 0.8, 1e-6])) > 0.0


def test_weighted_relative_residuals_use_sqrt_weight_scaling() -> None:
    model = RandlesModel()
    theta_true = np.array([0.01, 0.05, 2000.0, 0.8, 1e-6])
    theta_wrong = np.array([0.02, 0.04, 2200.0, 0.75, 2e-6])
    freq_hz = np.array([0.01, 1.0, 100.0])
    z_obs = model.simulate(freq_hz, theta_true)
    weights = np.array([1.0, 4.0, 9.0])
    dataset = EISDataset(freq_hz=freq_hz, z_obs=z_obs, z_true=z_obs)
    cost = EISPredictionErrorCost(
        dataset=dataset,
        model=model,
        weights=weights,
        relative=True,
        eps=1e-12,
    )

    expected = (z_obs - model.simulate(freq_hz, theta_wrong)) / np.maximum(
        np.abs(z_obs), 1e-12
    )
    expected *= np.sqrt(weights)

    np.testing.assert_allclose(cost.residuals(theta_wrong), expected)
    expected_cost = np.sum(expected.real**2 + expected.imag**2)
    assert cost(theta_wrong) == pytest.approx(expected_cost)
    assert cost(theta_wrong) == pytest.approx(cost(theta_wrong))
