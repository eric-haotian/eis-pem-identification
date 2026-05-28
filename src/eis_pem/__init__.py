"""Minimal prediction-error-method tooling for synthetic EIS identification."""

from .costs import EISPredictionErrorCost
from .dataset import EISDataset, generate_synthetic_dataset
from .diagnostics import IdentifiabilityReport, evaluate_local_identifiability
from .forward_models import ForwardModel, RandlesModel
from .frontend import dfn_frontend_response_to_frame, simulate_dfn_from_frontend
from .measurements import (
    ParameterMeasurementDataset,
    generate_synthetic_parameter_measurements,
)
from .optimizers import DifferentialEvolutionOptimizer, LeastSquaresOptimizer
from .parameters import ParameterSpec
from .plotting import (
    save_diagnostic_plots,
    save_joint_identification_plots,
    save_robust_selection_plots,
)
from .robust import IdentifiabilitySelector, ParameterSelection, ReducedParameterModel
from .results import IdentificationResult
from .seis_model import (
    DEFAULT_SELECTED_PARAMETER_NAMES,
    SEIS_COMPONENT_CHANNELS,
    DecoupledStackedSEISModel,
    SEISComponentModel,
    SEISModel,
    StackedSEISModel,
    all_seis_parameter_specs,
    default_seis_parameter_specs,
    default_seis_theta,
)

__all__ = [
    "DifferentialEvolutionOptimizer",
    "LeastSquaresOptimizer",
    "EISDataset",
    "EISPredictionErrorCost",
    "IdentifiabilitySelector",
    "IdentifiabilityReport",
    "ForwardModel",
    "dfn_frontend_response_to_frame",
    "IdentificationResult",
    "ParameterSpec",
    "ParameterMeasurementDataset",
    "ParameterSelection",
    "RandlesModel",
    "ReducedParameterModel",
    "SEISModel",
    "StackedSEISModel",
    "DEFAULT_SELECTED_PARAMETER_NAMES",
    "SEIS_COMPONENT_CHANNELS",
    "SEISComponentModel",
    "DecoupledStackedSEISModel",
    "all_seis_parameter_specs",
    "default_seis_parameter_specs",
    "default_seis_theta",
    "evaluate_local_identifiability",
    "generate_synthetic_dataset",
    "generate_synthetic_parameter_measurements",
    "save_diagnostic_plots",
    "save_joint_identification_plots",
    "save_robust_selection_plots",
    "simulate_dfn_from_frontend",
]
