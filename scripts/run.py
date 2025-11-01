from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from ocrcheckup import register_default_components
from ocrcheckup.core.types import Dataset, load_dataset
from ocrcheckup.core.variant import Variant
from ocrcheckup.normalization.normalizer import NormalizationSpec, Normalizer
from ocrcheckup.evaluation.evaluators import (
    AccuracyEvaluator,
    CorrectnessEvaluator,
    CostUSDEvaluator,
)
from ocrcheckup.runs.predict import run_prediction
from ocrcheckup.runs.evaluate import run_evaluation
from ocrcheckup.variants import discover_variants


DEFAULT_DATASET_PATH = Path("datasets") / "focused_scene.json"


def _load_variants() -> Iterable[Variant]:
    return discover_variants()


def _load_dataset(path: Path) -> Dataset:
    if not path.exists():
        raise FileNotFoundError(f"Dataset manifest not found at {path}")
    return load_dataset(path)


def _clone_variant(variant: Variant) -> Variant:
    return variant.model_copy(deep=True)


def _default_normalizer() -> Normalizer:
    spec = NormalizationSpec(
        unicode_form="NFC",
        lowercase=True,
        collapse_whitespace=True,
        remove_punctuation=True,
        spec_version="1",
    )
    return Normalizer(spec)


def _maybe_skip(run_path: Path, skip_existing: bool) -> bool:
    return skip_existing and run_path.exists()


def run_all(dataset_path: Path, *, seed: int, skip_existing: bool) -> None:
    register_default_components()
    dataset = _load_dataset(dataset_path)
    variants = list(_load_variants())
    if not variants:
        raise RuntimeError("No variants discovered. Ensure ocrcheckup.variants has modules with VARIANTS defined.")

    results_root = Path("results") / dataset.dataset_id
    runs_dir = results_root / "runs"
    evals_dir = results_root / "evals"
    runs_dir.mkdir(parents=True, exist_ok=True)
    evals_dir.mkdir(parents=True, exist_ok=True)

    normalizer = _default_normalizer()
    evaluators = [AccuracyEvaluator(), CorrectnessEvaluator(), CostUSDEvaluator()]

    for variant in variants:
        variant_instance = _clone_variant(variant)
        predicted_path = runs_dir / f"{variant_instance.name}.jsonl"
        if _maybe_skip(predicted_path, skip_existing):
            print(f"Skipping {variant_instance.name}: run exists at {predicted_path}")
            continue

        run_id = run_prediction(
            dataset,
            variant_instance,
            seed=seed,
            out_dir=runs_dir,
            out_path=predicted_path,
        )
        evaluation_path = evals_dir / f"{run_id}.jsonl"
        evaluation_id = run_evaluation(
            dataset,
            variant_instance,
            run_id,
            out_path=evaluation_path,
            normalizer=normalizer,
            evaluators=evaluators,
        )
        print(
            {
                "variant": variant_instance.name,
                "run_id": run_id,
                "evaluation_id": evaluation_id,
                "prediction_path": str(predicted_path),
                "evaluation_path": str(evaluation_path),
            }
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OCRCheckup benchmarks for all variants")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to dataset manifest (default: datasets/focused_scene.json)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for shuffling samples")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip running variants that already have prediction logs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_all(
        dataset_path=args.dataset,
        seed=args.seed,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
