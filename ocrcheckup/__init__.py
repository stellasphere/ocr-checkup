from .core.types import Dataset, Domain, Sample, load_dataset, build_sample_index
from .core.variant import Variant, AdapterRef, PricingRef
from .core.registry import model_families, adapters, pricing_models
from .normalization.normalizer import NormalizationSpec, Normalizer
from .evaluation.evaluators import (
    AccuracyEvaluator,
    CorrectnessEvaluator,
    SpeedEvaluator,
    CostUSDEvaluator,
)
from .runs.predict import run_prediction
from .runs.evaluate import run_evaluation
from .runs.summarize import summarize_evaluation, summarize_by_family
from .runs.leaderboard import build_leaderboard, write_leaderboard

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
    "SpeedEvaluator",
    "CostUSDEvaluator",
    # Runs
    "run_prediction",
    "run_evaluation",
    # Summarization and leaderboard
    "summarize_evaluation",
    "summarize_by_family",
    "build_leaderboard",
    "write_leaderboard",
]

__version__ = "0.1.0"
