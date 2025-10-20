import json
from pathlib import Path

from ocrcheckup.core.types import Dataset, Domain, Sample
from ocrcheckup.core.types import build_sample_index
from ocrcheckup.core.variant import Variant, AdapterRef, PricingRef
from ocrcheckup.core.registry import model_families, adapters, pricing_models
from ocrcheckup.families.examples.tesseract import family as tesseract_family
from ocrcheckup.adapters.examples.mock import adapter as mock_adapter
from ocrcheckup.pricing.builtins.local import pricing as local_pricing
from ocrcheckup.normalization.normalizer import NormalizationSpec, Normalizer
from ocrcheckup.evaluation.evaluators import AccuracyEvaluator, CorrectnessEvaluator, CostUSDEvaluator
from ocrcheckup.runs.predict import run_prediction
from ocrcheckup.runs.evaluate import run_evaluation


def test_pipeline_smoke(tmp_path: Path):
    # Minimal in-memory dataset
    samples = [
        Sample(sample_id="s1", image_uri="/tmp/s1.png", ground_truth_raw="Hello, World!"),
        Sample(sample_id="s2", image_uri="/tmp/s2.png", ground_truth_raw="OCR Checkup"),
    ]
    domain = Domain(domain_id="d1", name="demo", samples=samples)
    dataset = Dataset(dataset_id="ds1", name="Demo", domains=[domain])
    build_sample_index(dataset)  # exercise

    # Register components
    model_families.register(tesseract_family)
    adapters.register(mock_adapter)
    pricing_models.register(local_pricing)

    # Variant
    variant = Variant(
        name="mock-variant",
        family_id="tesseract",
        fields={"lang": "eng"},
        adapter=AdapterRef(id="mock"),
        pricing=PricingRef(id="local"),
    )

    # Prediction
    runs_path = tmp_path / "runs.jsonl"
    run_id = run_prediction(dataset, variant, seed=0, out_path=runs_path)

    # Evaluation
    spec = NormalizationSpec(
        unicode_form="NFC",
        lowercase=True,
        collapse_whitespace=True,
        remove_punctuation=True,
        spec_version="1",
    )
    normalizer = Normalizer(spec)
    evaluators = [AccuracyEvaluator(), CorrectnessEvaluator(), CostUSDEvaluator()]
    evals_path = tmp_path / "evals.jsonl"
    evaluation_id = run_evaluation(
        dataset,
        variant,
        run_id,
        out_path=evals_path,
        normalizer=normalizer,
        evaluators=evaluators,
        pred_path=runs_path,
    )

    assert runs_path.exists() and evals_path.exists()
    # Verify at least one evaluation record and expected metric keys
    lines = evals_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(samples)
    rec = json.loads(lines[0])
    assert set(["accuracy", "correctness", "cost_usd"]) <= set(rec["metrics"].keys()) 