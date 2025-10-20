from __future__ import annotations

from pathlib import Path

from ocrcheckup.core.types import load_dataset
from ocrcheckup.core.registry import model_families, adapters, pricing_models
from ocrcheckup.core.variant import Variant, AdapterRef, PricingRef
from ocrcheckup.families.examples.tesseract import family as tesseract_family
from ocrcheckup.adapters.examples.mock import adapter as mock_adapter
from ocrcheckup.pricing.builtins.local import pricing as local_pricing
from ocrcheckup.normalization.normalizer import NormalizationSpec, Normalizer
from ocrcheckup.evaluation.evaluators import AccuracyEvaluator, CorrectnessEvaluator, CostUSDEvaluator
from ocrcheckup.runs.predict import run_prediction
from ocrcheckup.runs.evaluate import run_evaluation


if __name__ == "__main__":
    # Register components
    model_families.register(tesseract_family)
    adapters.register(mock_adapter)
    pricing_models.register(local_pricing)

    # Construct variant
    variant = Variant(
        name="mock-tesseract-local",
        description="Mock adapter using tesseract family fields",
        family_id="tesseract",
        fields={"lang": "eng"},
        adapter=AdapterRef(id="mock", config=None),
        pricing=PricingRef(id="local", config=None),
    )

    # Load dataset (expects datasets/10x10.json)
    dataset = load_dataset(Path("datasets/10x10.json"))

    # Run prediction -> writes runs/<run_id>.jsonl
    run_id = run_prediction(dataset, variant, seed=42)

    # Evaluate -> writes evals/<evaluation_id>.jsonl
    spec = NormalizationSpec(
        unicode_form="NFC",
        lowercase=True,
        collapse_whitespace=True,
        remove_punctuation=True,
        spec_version="1",
    )
    normalizer = Normalizer(spec)
    evaluators = [AccuracyEvaluator(), CorrectnessEvaluator(), CostUSDEvaluator()]
    evaluation_id = run_evaluation(
        dataset,
        variant,
        run_id,
        out_path=Path("evals") / f"{run_id}.jsonl",
        normalizer=normalizer,
        evaluators=evaluators,
    )

    print({"run_id": run_id, "evaluation_id": evaluation_id})
