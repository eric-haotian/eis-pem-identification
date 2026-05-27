import numpy as np

from eis_pem.dataset import generate_synthetic_dataset
from eis_pem.forward_models import RandlesModel
from eis_pem.optimizers import LeastSquaresOptimizer
from eis_pem.parameters import ParameterSpec


def test_least_squares_optimizer_recovers_randles_parameters_in_log_space() -> None:
    model = RandlesModel()
    theta_true = np.array([0.01, 0.05, 2000.0, 0.8, 1e-6])
    dataset = generate_synthetic_dataset(
        model=model,
        freq_hz=np.logspace(-2, 5, 100),
        theta_true=theta_true,
        noise_level=0.0,
    )
    specs = [
        ParameterSpec("Rs", 0.02, (1e-4, 1e-1), "ohm", True),
        ParameterSpec("Rct", 0.1, (1e-4, 1.0), "ohm", True),
        ParameterSpec("Qdl", 1000.0, (10.0, 1e5), "F", True),
        ParameterSpec("alpha", 0.9, (0.5, 1.0), "-", False),
        ParameterSpec("L", 2e-6, (1e-10, 1e-4), "H", True),
    ]

    result = LeastSquaresOptimizer().fit(dataset, model, specs)

    estimate = np.array([result.theta_best[spec.name] for spec in specs])
    np.testing.assert_allclose(estimate, theta_true, rtol=1e-5)
