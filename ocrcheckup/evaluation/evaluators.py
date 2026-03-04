from __future__ import annotations

from typing import Any, Protocol

import Levenshtein

from ocrcheckup.core.time import parse_iso
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
        return 100.0 * Levenshtein.ratio(gt, pred)


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


class SpeedEvaluator:
    metric_name = "elapsed_seconds"

    def evaluate(
        self,
        gt_raw: str,
        pred_raw: str,
        *,
        normalizer: Normalizer,
        prediction_record: dict,
        variant: Variant,
    ) -> float | None:
        started_at = prediction_record.get("started_at")
        ended_at = prediction_record.get("ended_at")
        if not started_at or not ended_at:
            return None
        start = parse_iso(started_at)
        end = parse_iso(ended_at)
        return (end - start).total_seconds()


class CostUSDEvaluator:
    metric_name = "cost_usd"

    def __init__(self, cost_profile: str | None = None) -> None:
        self._cost_profile = cost_profile

    def evaluate(
        self,
        gt_raw: str,
        pred_raw: str,
        *,
        normalizer: Normalizer,
        prediction_record: dict,
        variant: Variant,
    ) -> float | None:
        from ocrcheckup.core.registry import pricing_models

        # For local pricing with a cost profile, compute from elapsed time
        if variant.pricing.id == "local" and self._cost_profile is not None:
            from ocrcheckup.evaluation.cost_profiles import (
                compute_local_cost,
                get_cost_profile,
            )

            usd_per_hour = get_cost_profile(self._cost_profile)
            started_at = prediction_record.get("started_at")
            ended_at = prediction_record.get("ended_at")
            if not started_at or not ended_at:
                return 0.0
            elapsed = (parse_iso(ended_at) - parse_iso(started_at)).total_seconds()
            return compute_local_cost(elapsed, usd_per_hour)

        pricing = pricing_models.get(variant.pricing.id)
        return pricing.quote(prediction_record, variant, variant.pricing.config)


__all__ = [
    "Evaluator",
    "AccuracyEvaluator",
    "CorrectnessEvaluator",
    "SpeedEvaluator",
    "CostUSDEvaluator",
]
