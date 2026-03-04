from __future__ import annotations

import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from ocrcheckup.core.jsonl import iterate_jsonl
from ocrcheckup.core.registry import model_families


def summarize_evaluation(
    eval_path: Path | str,
    *,
    variant_name: str,
    variant_id: str,
    family_id: str,
) -> Dict[str, Any]:
    """
    Aggregate per-sample evaluation records into a per-variant summary.

    Returns a dict with aggregate metrics including per-domain breakdown.
    """
    eval_path = Path(eval_path)

    accuracies: List[float] = []
    correctnesses: List[int] = []
    elapsed_list: List[float] = []
    cost_list: List[float] = []
    per_domain: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {"accuracy": [], "elapsed_seconds": [], "cost_usd": []}
    )
    sample_count = 0
    error_count = 0

    for rec in iterate_jsonl(eval_path):
        sample_count += 1

        # Skip errored samples from metric aggregation
        if rec.get("is_error"):
            error_count += 1
            continue

        metrics = rec.get("metrics", {})
        domain_id = rec.get("domain_id", "unknown")

        acc = metrics.get("accuracy")
        if acc is not None:
            accuracies.append(float(acc))
            per_domain[domain_id]["accuracy"].append(float(acc))

        corr = metrics.get("correctness")
        if corr is not None:
            correctnesses.append(int(corr))

        elapsed = metrics.get("elapsed_seconds")
        if elapsed is not None:
            elapsed_list.append(float(elapsed))
            per_domain[domain_id]["elapsed_seconds"].append(float(elapsed))

        cost = metrics.get("cost_usd")
        if cost is not None:
            cost_list.append(float(cost))
            per_domain[domain_id]["cost_usd"].append(float(cost))

    mean_accuracy = statistics.mean(accuracies) if accuracies else None
    mean_correctness = statistics.mean(correctnesses) if correctnesses else None
    mean_elapsed = statistics.mean(elapsed_list) if elapsed_list else None
    mean_cost = statistics.mean(cost_list) if cost_list else None
    total_cost = sum(cost_list) if cost_list else None

    # Derived efficiency metrics
    speed_efficiency = None
    if mean_accuracy is not None and mean_elapsed is not None and mean_elapsed > 0:
        speed_efficiency = mean_accuracy / mean_elapsed

    cost_efficiency = None
    if mean_accuracy is not None and mean_cost is not None and mean_cost > 0:
        cost_efficiency = mean_accuracy / mean_cost

    # Per-domain summaries
    domain_summaries = {}
    for domain_id, domain_metrics in sorted(per_domain.items()):
        domain_summaries[domain_id] = {
            "sample_count": len(domain_metrics["accuracy"]),
            "mean_accuracy": statistics.mean(domain_metrics["accuracy"]) if domain_metrics["accuracy"] else None,
            "mean_elapsed_seconds": statistics.mean(domain_metrics["elapsed_seconds"]) if domain_metrics["elapsed_seconds"] else None,
            "mean_cost_usd": statistics.mean(domain_metrics["cost_usd"]) if domain_metrics["cost_usd"] else None,
        }

    return {
        "variant_id": variant_id,
        "variant_name": variant_name,
        "family_id": family_id,
        "sample_count": sample_count,
        "error_count": error_count,
        "success_count": sample_count - error_count,
        "mean_accuracy": mean_accuracy,
        "mean_correctness": mean_correctness,
        "mean_elapsed_seconds": mean_elapsed,
        "mean_cost_usd": mean_cost,
        "total_cost_usd": total_cost,
        "speed_efficiency": speed_efficiency,
        "cost_efficiency": cost_efficiency,
        "per_domain": domain_summaries,
    }


def summarize_by_family(
    variant_summaries: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Group variant summaries by family_id.

    For each family, pick the best variant (highest mean_accuracy)
    and produce a collapsed family-level summary for the leaderboard.
    """
    by_family: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for vs in variant_summaries:
        by_family[vs["family_id"]].append(vs)

    family_summaries: List[Dict[str, Any]] = []
    for family_id, variants in by_family.items():
        best = max(variants, key=lambda v: v.get("mean_accuracy") or 0.0)

        # Look up display_name from registry if available
        display_name = family_id
        try:
            fam = model_families.get(family_id)
            display_name = getattr(fam, "display_name", family_id)
        except KeyError:
            pass

        family_summaries.append(
            {
                "family_id": family_id,
                "display_name": display_name,
                "best_variant_id": best["variant_id"],
                "best_variant_name": best["variant_name"],
                "variant_count": len(variants),
                "mean_accuracy": best["mean_accuracy"],
                "mean_correctness": best["mean_correctness"],
                "mean_elapsed_seconds": best["mean_elapsed_seconds"],
                "mean_cost_usd": best["mean_cost_usd"],
                "total_cost_usd": best["total_cost_usd"],
                "speed_efficiency": best["speed_efficiency"],
                "cost_efficiency": best["cost_efficiency"],
                "per_domain": best.get("per_domain", {}),
            }
        )

    family_summaries.sort(
        key=lambda f: f.get("mean_accuracy") or 0.0, reverse=True
    )
    return family_summaries
