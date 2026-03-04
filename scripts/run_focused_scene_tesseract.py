from __future__ import annotations

from pathlib import Path

from ocrcheckup.core.types import load_dataset
from ocrcheckup.core.registry import model_families, adapters, pricing_models
from ocrcheckup.core.variant import Variant, AdapterRef, PricingRef
from ocrcheckup.families.tesseract import tesseract_family
from ocrcheckup.adapters.tesseract_pytesseract import adapter as tess_adapter
from ocrcheckup.pricing.builtins.local import pricing as local_pricing
from ocrcheckup.normalization.normalizer import NormalizationSpec, Normalizer
from ocrcheckup.evaluation.evaluators import (
    AccuracyEvaluator,
    CorrectnessEvaluator,
    CostUSDEvaluator,
    SpeedEvaluator,
)
from ocrcheckup.runs.predict import run_prediction
from ocrcheckup.runs.evaluate import run_evaluation


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    manifest = repo_root / "datasets" / "focused_scene.json"

    # Register components
    model_families.register(tesseract_family)
    adapters.register(tess_adapter)
    pricing_models.register(local_pricing)

    # Construct variant (real adapter)
    variant = Variant(
        name="tesseract-eng",
        family_id="tesseract",
        fields={"lang": "eng", "psm": 6},
        adapter=AdapterRef(id="tesseract-pytesseract"),
        pricing=PricingRef(id="local"),
    )

    dataset = load_dataset(manifest)

    run_id = run_prediction(dataset, variant, seed=42, out_dir=repo_root / "results" / "runs")

    spec = NormalizationSpec(
        unicode_form="NFC",
        lowercase=True,
        collapse_whitespace=True,
        remove_punctuation=True,
        spec_version="1",
    )
    normalizer = Normalizer(spec)
    evaluators = [AccuracyEvaluator(), CorrectnessEvaluator(), SpeedEvaluator(), CostUSDEvaluator()]
    eval_out = repo_root / "results" / "evals" / f"{run_id}.jsonl"
    evaluation_id = run_evaluation(
        dataset,
        variant,
        run_id,
        out_path=eval_out,
        normalizer=normalizer,
        evaluators=evaluators,
    )
    print({"run_id": run_id, "evaluation_id": evaluation_id, "eval_path": str(eval_out)})


if __name__ == "__main__":
    main()
