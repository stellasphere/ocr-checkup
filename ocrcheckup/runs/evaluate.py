from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic import BaseModel

from ocrcheckup.core.ids import new_evaluation_id
from ocrcheckup.core.jsonl import iterate_jsonl, jsonl_writer
from ocrcheckup.core.types import Dataset, build_sample_index
from ocrcheckup.core.variant import Variant
from ocrcheckup.evaluation.evaluators import (
    AccuracyEvaluator,
    CorrectnessEvaluator,
    CostUSDEvaluator,
    Evaluator,
)
from ocrcheckup.normalization.normalizer import Normalizer


class EvaluationRun(BaseModel):
    evaluation_id: str
    run_id: str
    normalizer: Normalizer
    evaluators: List[Evaluator]


def run_evaluation(
    dataset: Dataset,
    variant: Variant,
    run_id: str,
    *,
    out_path: Path,
    normalizer: Normalizer,
    evaluators: List[Evaluator],
    pred_path: Path | str | None = None,
) -> str:
    evaluation_id = new_evaluation_id()
    erun = EvaluationRun(
        evaluation_id=evaluation_id,
        run_id=run_id,
        normalizer=normalizer,
        evaluators=evaluators,
    )

    index = build_sample_index(dataset)

    pred_path = Path(pred_path) if pred_path is not None else (Path("runs") / f"{run_id}.jsonl")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure required metrics include cost
    metric_names = {ev.metric_name for ev in evaluators}
    assert "cost_usd" in metric_names, "CostUSD evaluator is required per TDD"

    with jsonl_writer(out_path) as write:
        for rec in iterate_jsonl(pred_path):
            sample_id = rec["sample_id"]
            _, sample = index[sample_id]
            gt_raw = sample.ground_truth_raw
            pred_raw = rec.get("prediction_raw", "")

            gt_norm = normalizer.normalize(gt_raw)
            pred_norm = normalizer.normalize(pred_raw)

            metrics = {}
            for ev in evaluators:
                metrics[ev.metric_name] = ev.evaluate(
                    gt_raw, pred_raw, normalizer=normalizer, prediction_record=rec, variant=variant
                )

            out_rec = {
                "evaluation_id": erun.evaluation_id,
                "run_id": run_id,
                "variant_id": rec.get("variant_id"),
                "sample_id": sample_id,
                "gt_norm": gt_norm,
                "pred_norm": pred_norm,
                "metrics": metrics,
            }
            write(out_rec)

    return erun.evaluation_id
