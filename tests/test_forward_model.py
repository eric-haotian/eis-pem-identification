import numpy as np
import pytest

from eis_pem.forward_models import RandlesModel
from eis_pem.parameters import ParameterSpec


def test_randles_model_returns_finite_complex_spectrum() -> None:
    model = RandlesModel()
    freq_hz = np.logspace(-2, 5, 100)

    z_sim = model.simulate(freq_hz, np.array([0.01, 0.05, 2000.0, 0.8, 1e-6]))

    assert z_sim.shape == freq_hz.shape
    assert np.iscomplexobj(z_sim)
    assert np.all(np.isfinite(z_sim.real))
    assert np.all(np.isfinite(z_sim.imag))


def test_randles_model_matches_analytic_formula() -> None:
    model = RandlesModel()
    freq_hz = np.array([1.0])
    theta = np.array([0.01, 0.05, 2000.0, 0.8, 1e-6])
    expected = (
        theta[0]
        + 1j * 2 * np.pi * freq_hz * theta[4]
        + theta[1]
        / (1.0 + (1j * 2 * np.pi * freq_hz * theta[1] * theta[2]) ** theta[3])
    )

    np.testing.assert_allclose(model.simulate(freq_hz, theta), expected)


@pytest.mark.parametrize(
    "theta",
    [
        np.array([0.0, 0.05, 2000.0, 0.8, 1e-6]),
        np.array([0.01, -0.05, 2000.0, 0.8, 1e-6]),
        np.array([0.01, 0.05, np.nan, 0.8, 1e-6]),
    ],
)
def test_randles_model_rejects_invalid_parameters(theta: np.ndarray) -> None:
    with pytest.raises(ValueError):
        RandlesModel().simulate(np.array([1.0]), theta)


def test_log_parameter_spec_round_trips_physical_values() -> None:
    spec = ParameterSpec(
        name="Rs",
        initial_value=0.01,
        bounds=(1e-4, 1e-1),
        unit="ohm",
        log_transform=True,
    )

    assert spec.optimization_bounds == pytest.approx((-4.0, -1.0))
    assert spec.from_optimization(spec.to_optimization(0.01)) == pytest.approx(0.01)
