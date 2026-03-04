from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from ocrcheckup.core.time import utc_now_iso


def build_leaderboard(
    variant_summaries: List[Dict[str, Any]],
    family_summaries: List[Dict[str, Any]],
    *,
    dataset_id: str,
    cost_profile: str | None = None,
) -> Dict[str, Any]:
    """Build a leaderboard-ready JSON structure."""
    return {
        "generated_at": utc_now_iso(),
        "dataset_id": dataset_id,
        "cost_profile": cost_profile,
        "leaderboard": {
            "by_family": family_summaries,
            "by_variant": sorted(
                variant_summaries,
                key=lambda v: v.get("mean_accuracy") or 0.0,
                reverse=True,
            ),
        },
    }


def write_leaderboard(
    leaderboard: Dict[str, Any],
    out_path: Path | str,
) -> None:
    """Write leaderboard JSON to disk."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(leaderboard, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
