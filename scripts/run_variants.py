from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from ocrcheckup.adapters.cloud import (
    AnthropicAdapter,
    GeminiAdapter,
    MistralAdapter,
    OpenAIAdapter,
    RoboflowAdapter,
)
from ocrcheckup.adapters.local import (
    EasyOCRAdapter,
    IdeficsAdapter,
    MoondreamAdapter,
    QwenAdapter,
    TrOCRAdapter,
)
from ocrcheckup.adapters.tesseract_pytesseract import adapter as tesseract_adapter
from ocrcheckup.core.registry import adapters, pricing_models
from ocrcheckup.core.types import load_dataset
from ocrcheckup.core.variant import Variant
from ocrcheckup.evaluation.evaluators import AccuracyEvaluator, CorrectnessEvaluator, CostUSDEvaluator
from ocrcheckup.normalization.normalizer import NormalizationSpec, Normalizer
from ocrcheckup.pricing.builtins.flat_per_image import pricing as flat_per_image_pricing
from ocrcheckup.pricing.builtins.local import pricing as local_pricing
from ocrcheckup.pricing.providers import (
    anthropic_pricing,
    gemini_pricing,
    mistral_pricing,
    openai_pricing,
    roboflow_pricing,
)
from ocrcheckup.runs.evaluate import run_evaluation
from ocrcheckup.runs.predict import run_prediction
from ocrcheckup.utils.discovery import discover_families_and_variants


def register_components() -> None:
    # Adapters
    adapters.register(OpenAIAdapter())
    adapters.register(AnthropicAdapter())
    adapters.register(GeminiAdapter())
    adapters.register(MistralAdapter())
    adapters.register(RoboflowAdapter())
    adapters.register(tesseract_adapter)
    adapters.register(EasyOCRAdapter())
    adapters.register(QwenAdapter())
    adapters.register(TrOCRAdapter())
    adapters.register(IdeficsAdapter())
    adapters.register(MoondreamAdapter())

    # Pricing models
    for pricing in [
        openai_pricing,
        anthropic_pricing,
        gemini_pricing,
        mistral_pricing,
        roboflow_pricing,
        local_pricing,
        flat_per_image_pricing,
    ]:
        pricing_models.register(pricing)


def collect_variants(all_variants: List[Variant], include_cloud: bool, include_local: bool) -> List[Variant]:
    selected: List[Variant] = []
    for v in all_variants:
        # Simple heuristic: local pricing means local variant
        is_local = v.pricing.id == "local"
        if is_local and include_local:
            selected.append(v)
        elif not is_local and include_cloud:
            selected.append(v)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OCRCheckup variants.")
    parser.add_argument("--dataset", default="datasets/focused_scene.json", help="Path to dataset manifest.")
    parser.add_argument(
        "--include-cloud",
        action="store_true",
        help="Include cloud variants (requires API credentials).",
    )
    parser.add_argument(
        "--include-local",
        action="store_true",
        help="Include local variants.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for prediction runs.")
    args = parser.parse_args()

    if not args.include_cloud and not args.include_local:
        args.include_local = True

    register_components()
    # Discovery registers families and returns all variants
    discovered_variants = discover_families_and_variants()

    dataset = load_dataset(Path(args.dataset))
    variants = collect_variants(discovered_variants, include_cloud=args.include_cloud, include_local=args.include_local)

    normalizer = Normalizer(
        NormalizationSpec(
            unicode_form="NFC",
            lowercase=True,
            collapse_whitespace=True,
            remove_punctuation=True,
            spec_version="1",
        )
    )
    evaluators = [AccuracyEvaluator(), CorrectnessEvaluator(), CostUSDEvaluator()]

    print(f"Found {len(variants)} variants to run.")
    for variant in variants:
        print(f"Running variant: {variant.name} ({variant.family_id})")
        try:
            run_id = run_prediction(dataset, variant, seed=args.seed)
        except Exception as exc:
            print(f"  Skipped {variant.name}: {exc}")
            continue

        evaluation_path = Path("results") / "evals" / f"{run_id}.jsonl"
        evaluation_id = run_evaluation(
            dataset,
            variant,
            run_id,
            out_path=evaluation_path,
            normalizer=normalizer,
            evaluators=evaluators,
        )
        print(f"  Completed run_id={run_id}, evaluation_id={evaluation_id}")


if __name__ == "__main__":
    main()
