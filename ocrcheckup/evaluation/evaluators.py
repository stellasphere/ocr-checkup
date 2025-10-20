from __future__ import annotations

from typing import Any, Protocol

from ocrcheckup.core.variant import Variant
from ocrcheckup.normalization.normalizer import Normalizer


class Evaluator(Protocol):
    metric_name: str

    def evaluate(
        self,
        gt_raw: str,
        pred_raw: str,
        *,
        normalizer: Normalizer,
        prediction_record: dict,
        variant: Variant,
    ) -> Any: ...


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr.append(
                min(
                    curr[-1] + 1,
                    prev[j] + 1,
                    prev[j - 1] + cost,
                )
            )
        prev = curr
    return prev[-1]


class AccuracyEvaluator:
    metric_name = "accuracy"

    def evaluate(
        self,
        gt_raw: str,
        pred_raw: str,
        *,
        normalizer: Normalizer,
        prediction_record: dict,
        variant: Variant,
    ) -> float:
        gt = normalizer.normalize(gt_raw)
        pred = normalizer.normalize(pred_raw)
        if not gt and not pred:
            return 100.0
        if not gt and pred:
            return 0.0
        dist = _edit_distance(gt, pred)
        return 100.0 * (1.0 - (dist / max(1, len(gt))))


class CorrectnessEvaluator:
    metric_name = "correctness"

    def evaluate(
        self,
        gt_raw: str,
        pred_raw: str,
        *,
        normalizer: Normalizer,
        prediction_record: dict,
        variant: Variant,
    ) -> int:
        gt = normalizer.normalize(gt_raw)
        pred = normalizer.normalize(pred_raw)
        return 1 if gt == pred else 0


class CostUSDEvaluator:
    metric_name = "cost_usd"

    def evaluate(
        self,
        gt_raw: str,
        pred_raw: str,
        *,
        normalizer: Normalizer,
        prediction_record: dict,
        variant: Variant,
    ) -> float | None:
        # Pricing is evaluated using the prediction_record metadata and variant pricing config
        from ocrcheckup.core.registry import pricing_models

        pricing = pricing_models.get(variant.pricing.id)
        return pricing.quote(prediction_record, variant, variant.pricing.config)


__all__ = [
    "Evaluator",
    "AccuracyEvaluator",
    "CorrectnessEvaluator",
    "CostUSDEvaluator",
]
