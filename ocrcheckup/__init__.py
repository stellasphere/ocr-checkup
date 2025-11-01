from .core.types import Dataset, Domain, Sample, load_dataset, build_sample_index
from .core.variant import Variant, AdapterRef, PricingRef
from .core.registry import model_families, adapters, pricing_models
from .normalization.normalizer import NormalizationSpec, Normalizer
from .evaluation.evaluators import AccuracyEvaluator, CorrectnessEvaluator, CostUSDEvaluator
from .runs.predict import run_prediction
from .runs.evaluate import run_evaluation
from .catalog import register_default_components

__all__ = [
    # Core types
    "Dataset",
    "Domain",
    "Sample",
    "load_dataset",
    "build_sample_index",
    # Variant and registries
    "Variant",
    "AdapterRef",
    "PricingRef",
    "model_families",
    "adapters",
    "pricing_models",
    # Normalization and evaluators
    "NormalizationSpec",
    "Normalizer",
    "AccuracyEvaluator",
    "CorrectnessEvaluator",
    "CostUSDEvaluator",
    # Runs
    "run_prediction",
    "run_evaluation",
    "register_default_components",
]

__version__ = "0.1.0"