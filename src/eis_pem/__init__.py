"""Prediction-error-method tooling for battery EIS parameter identification."""
# learning AI website www.haotianblog.com
from .costs import EISPredictionErrorCost
from .dataset import EISDataset, generate_synthetic_dataset
from .diagnostics import (
    IdentifiabilityReport,
    compute_post_fit_diagnostics,
    evaluate_local_identifiability,
)
from .forward_models import ForwardModel, RandlesModel
from .frontend import (
    dfn_frontend_response_to_frame,
    identify_parameters_robust,
    identify_with_model_selection,
    simulate_dfn_from_frontend,
)
from .ecm_library import (
    SingleRCModel,
    SingleRCPEModel,
    TwoRCPEModel,
    RandlesWarburgModel,
    TwoRCPEWarburgModel,
    ThreeRCPEModel,
    all_ecm_models,
    ecm_model_by_name,
    suggest_models_from_peaks,
)
from .drt import DRTAnalyzer, DRTResult
from .model_selection import (
    ModelComparisonResult,
    ModelFitResult,
    compare_models,
    compute_aic,
    compute_aicc,
    compute_bic,
)
from .physics_priors import (
    BoundPrior,
    OrderingPrior,
    PhysicsPrior,
    RatioPrior,
    SumPrior,
    check_prior_violations,
    lithium_ion_priors,
    pemfc_priors,
    prior_penalty,
)
from .measurements import (
    ParameterMeasurementDataset,
    generate_synthetic_parameter_measurements,
)
from .data_quality import DataQualityReport, assess_data_quality
from .frequency_filter import (
    FrequencyBandAnalyzer,
    FrequencyBandConfig,
    FrequencyFilterResult,
    filter_dataset,
    weighted_dataset,
)
from .optimizers import (
    AdaptiveLeastSquaresOptimizer,
    DifferentialEvolutionOptimizer,
    HybridOptimizer,
    LeastSquaresOptimizer,
)
from .parameters import ParameterSpec
from .plotting import (
    save_diagnostic_plots,
    save_joint_identification_plots,
    save_robust_selection_plots,
)
from .robust import (
    IdentifiabilitySelector,
    IdentifiabilityStrategy,
    ParameterSelection,
    ReducedParameterModel,
)
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
    # Optimizers
    "AdaptiveLeastSquaresOptimizer",
    "DifferentialEvolutionOptimizer",
    "HybridOptimizer",
    "LeastSquaresOptimizer",
    # Data containers
    "EISDataset",
    "EISPredictionErrorCost",
    "IdentificationResult",
    "ParameterSpec",
    "ParameterMeasurementDataset",
    # Models — physics-based
    "ForwardModel",
    "RandlesModel",
    "SEISModel",
    "StackedSEISModel",
    "SEISComponentModel",
    "DecoupledStackedSEISModel",
    # Models — ECM library
    "SingleRCModel",
    "SingleRCPEModel",
    "TwoRCPEModel",
    "RandlesWarburgModel",
    "TwoRCPEWarburgModel",
    "ThreeRCPEModel",
    "all_ecm_models",
    "ecm_model_by_name",
    "suggest_models_from_peaks",
    # DRT analysis
    "DRTAnalyzer",
    "DRTResult",
    # Model selection
    "ModelComparisonResult",
    "ModelFitResult",
    "compare_models",
    "compute_aic",
    "compute_aicc",
    "compute_bic",
    # Physics priors
    "PhysicsPrior",
    "BoundPrior",
    "RatioPrior",
    "OrderingPrior",
    "SumPrior",
    "prior_penalty",
    "check_prior_violations",
    "lithium_ion_priors",
    "pemfc_priors",
    # Robust selection
    "IdentifiabilitySelector",
    "IdentifiabilityStrategy",
    "ParameterSelection",
    "ReducedParameterModel",
    # Diagnostics
    "IdentifiabilityReport",
    "evaluate_local_identifiability",
    "compute_post_fit_diagnostics",
    # Data quality & frequency filtering
    "DataQualityReport",
    "assess_data_quality",
    "FrequencyBandAnalyzer",
    "FrequencyBandConfig",
    "FrequencyFilterResult",
    "filter_dataset",
    "weighted_dataset",
    # SEIS model helpers
    "DEFAULT_SELECTED_PARAMETER_NAMES",
    "SEIS_COMPONENT_CHANNELS",
    "all_seis_parameter_specs",
    "default_seis_parameter_specs",
    "default_seis_theta",
    # Generation & plotting
    "generate_synthetic_dataset",
    "generate_synthetic_parameter_measurements",
    "save_diagnostic_plots",
    "save_joint_identification_plots",
    "save_robust_selection_plots",
    # Frontend
    "dfn_frontend_response_to_frame",
    "identify_parameters_robust",
    "identify_with_model_selection",
    "simulate_dfn_from_frontend",
]
