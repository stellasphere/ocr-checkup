from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv

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
from ocrcheckup.core.registry import adapters, model_families, pricing_models
from ocrcheckup.core.types import load_dataset
from ocrcheckup.core.variant import Variant
from ocrcheckup.evaluation.evaluators import (
    AccuracyEvaluator,
    CorrectnessEvaluator,
    CostUSDEvaluator,
    SpeedEvaluator,
)
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
from ocrcheckup.runs.leaderboard import build_leaderboard, write_leaderboard
from ocrcheckup.runs.predict import find_run_by_variant_id, run_prediction
from ocrcheckup.runs.summarize import summarize_by_family, summarize_evaluation
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


def collect_variants(
    all_variants: List[Variant], include_cloud: bool, include_local: bool
) -> List[Variant]:
    selected: List[Variant] = []
    for v in all_variants:
        is_local = v.pricing.id == "local"
        if is_local and include_local:
            selected.append(v)
        elif not is_local and include_cloud:
            selected.append(v)
    return selected


def _compute_variant_id(variant: Variant) -> str:
    """Compute variant_id without running prediction (for dedup checks)."""
    family = model_families.get(variant.family_id)
    canonical_fields = family.validate_fields(variant.fields)
    adapter = adapters.get(variant.adapter.id)
    pricing = pricing_models.get(variant.pricing.id)
    adapter_hash_cfg = adapter.hash_config(variant.adapter.config)
    pricing_hash_cfg = pricing.hash_config(variant.pricing.config)
    variant.compute_variant_id(
        family_schema_version=family.family_schema_version,
        canonical_fields=canonical_fields,
        adapter_hash_cfg=adapter_hash_cfg,
        pricing_hash_cfg=pricing_hash_cfg,
    )
    return variant.variant_id


def _collect_all_eval_summaries(
    evals_dir: Path,
    runs_dir: Path,
) -> List[dict]:
    """Build summaries from ALL existing evaluation files (for historical leaderboard)."""
    summaries = []
    for eval_file in sorted(evals_dir.glob("*.jsonl")):
        run_id = eval_file.stem
        meta_path = runs_dir / f"{run_id}.meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            variant_data = meta.get("variant", {})
            summary = summarize_evaluation(
                eval_file,
                variant_name=variant_data.get("name", run_id),
                variant_id=meta.get("variant_id", ""),
                family_id=variant_data.get("family_id", "unknown"),
            )
            summaries.append(summary)
        except Exception as exc:
            print(f"  Warning: could not summarize {eval_file.name}: {exc}")
    return summaries


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run OCRCheckup variants.")
    parser.add_argument(
        "--dataset",
        default="datasets/focused_scene.json",
        help="Path to dataset manifest.",
    )
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
    parser.add_argument(
        "--seed", type=int, default=42, help="Shuffle seed for prediction runs."
    )
    parser.add_argument(
        "--cost-profile",
        default=None,
        help="Cost profile for local models (e.g., 'free', 't4-gcp', 'a100-40gb-gcp').",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run even if a run already exists for a variant.",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Re-evaluate existing prediction runs without re-predicting.",
    )
    parser.add_argument(
        "--leaderboard-only",
        action="store_true",
        help="Build leaderboard from all existing evaluation files without running anything.",
    )
    parser.add_argument(
        "--leaderboard-out",
        default="results/leaderboard.json",
        help="Path for leaderboard JSON output.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Max retries per sample on adapter failure.",
    )
    parser.add_argument(
        "--requests-per-minute",
        type=int,
        default=None,
        help="Rate limit for API requests (requests per minute).",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Variant names to exclude from the run.",
    )
    args = parser.parse_args()

    if not args.include_cloud and not args.include_local:
        args.include_local = True

    register_components()
    discovered_variants = discover_families_and_variants()

    dataset = load_dataset(Path(args.dataset))
    runs_dir = Path("results") / "runs"
    evals_dir = Path("results") / "evals"

    # --leaderboard-only: build from all existing evals and exit
    if args.leaderboard_only:
        print("Building leaderboard from all existing evaluations...")
        summaries = _collect_all_eval_summaries(evals_dir, runs_dir)
        if not summaries:
            print("No evaluation files with metadata found.")
            return
        family_summaries = summarize_by_family(summaries)
        leaderboard = build_leaderboard(
            summaries,
            family_summaries,
            dataset_id=dataset.dataset_id,
            cost_profile=args.cost_profile,
        )
        write_leaderboard(leaderboard, args.leaderboard_out)
        print(f"Leaderboard written to {args.leaderboard_out} ({len(summaries)} variants)")
        _print_leaderboard(family_summaries)
        return

    variants = collect_variants(
        discovered_variants,
        include_cloud=args.include_cloud,
        include_local=args.include_local,
    )

    if args.exclude:
        excluded = set(args.exclude)
        variants = [v for v in variants if v.name not in excluded]

    normalizer = Normalizer(
        NormalizationSpec(
            unicode_form="NFC",
            lowercase=True,
            collapse_whitespace=True,
            remove_punctuation=True,
            spec_version="1",
        )
    )
    evaluators = [
        AccuracyEvaluator(),
        CorrectnessEvaluator(),
        SpeedEvaluator(),
        CostUSDEvaluator(cost_profile=args.cost_profile),
    ]

    completed: List[Tuple[str, Variant]] = []

    print(f"Found {len(variants)} variants to run.")
    for variant in variants:
        print(f"\nVariant: {variant.name} ({variant.family_id})")

        # Compute variant_id for deduplication
        variant_id = _compute_variant_id(variant)

        if args.evaluate_only:
            # Find existing run by variant_id
            run_id = find_run_by_variant_id(variant_id, runs_dir)
            if run_id is None:
                print(f"  No existing run found for variant_id={variant_id}, skipping.")
                continue
            print(f"  Re-evaluating existing run_id={run_id}")
        else:
            # Check for existing run (deduplication)
            if not args.force:
                existing_run_id = find_run_by_variant_id(variant_id, runs_dir)
                if existing_run_id is not None:
                    print(f"  Skipping: run already exists (run_id={existing_run_id}). Use --force to re-run.")
                    # Still re-evaluate with current settings
                    run_id = existing_run_id
                    evaluation_path = evals_dir / f"{run_id}.jsonl"
                    evaluation_id = run_evaluation(
                        dataset,
                        variant,
                        run_id,
                        out_path=evaluation_path,
                        normalizer=normalizer,
                        evaluators=evaluators,
                        runs_dir=runs_dir,
                    )
                    print(f"  Re-evaluated: evaluation_id={evaluation_id}")
                    completed.append((run_id, variant))
                    continue

            try:
                run_id = run_prediction(
                    dataset,
                    variant,
                    seed=args.seed,
                    max_retries=args.max_retries,
                    requests_per_minute=args.requests_per_minute,
                )
            except Exception as exc:
                print(f"  Skipped {variant.name}: {exc}")
                continue

        evaluation_path = evals_dir / f"{run_id}.jsonl"
        evaluation_id = run_evaluation(
            dataset,
            variant,
            run_id,
            out_path=evaluation_path,
            normalizer=normalizer,
            evaluators=evaluators,
            runs_dir=runs_dir,
        )
        print(f"  Completed: run_id={run_id}, evaluation_id={evaluation_id}")
        completed.append((run_id, variant))

    # Summarize and build leaderboard
    if completed:
        print(f"\nSummarizing {len(completed)} variant(s)...")
        variant_summaries = []
        for run_id, variant in completed:
            eval_path = evals_dir / f"{run_id}.jsonl"
            summary = summarize_evaluation(
                eval_path,
                variant_name=variant.name,
                variant_id=variant.variant_id or "",
                family_id=variant.family_id,
            )
            variant_summaries.append(summary)

        family_summaries = summarize_by_family(variant_summaries)
        leaderboard = build_leaderboard(
            variant_summaries,
            family_summaries,
            dataset_id=dataset.dataset_id,
            cost_profile=args.cost_profile,
        )
        write_leaderboard(leaderboard, args.leaderboard_out)
        print(f"\nLeaderboard written to {args.leaderboard_out}")
        _print_leaderboard(family_summaries)


def _print_leaderboard(family_summaries: List[dict]) -> None:
    print("\n--- Leaderboard (by family, accuracy) ---")
    for i, fam in enumerate(family_summaries, 1):
        acc = fam.get("mean_accuracy")
        acc_str = f"{acc:.1f}%" if acc is not None else "N/A"
        print(f"  {i}. {fam['display_name']}: {acc_str} (variant: {fam['best_variant_name']})")


if __name__ == "__main__":
    main()
