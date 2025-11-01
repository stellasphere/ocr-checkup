from __future__ import annotations

from typing import Any, Dict, Optional

from ocrcheckup.core.variant import Variant


class PerPagePricing:
    id = "per-page"
    description = "Pricing based on pages processed metadata"

    def quote(self, prediction_record: dict, variant: Variant, config: Optional[Dict[str, Any]]) -> float | None:
        if not config or "usd" not in config:
            raise ValueError("per-page pricing requires config with {'usd': float}")
        metadata = prediction_record.get("metadata") or {}
        usage = metadata.get("usage") or {}
        pages = usage.get("pages_processed")
        if pages is None:
            return None
        rate = float(config["usd"])
        total = float(pages) * rate
        rounding = config.get("round")
        if rounding is not None:
            total = round(total, int(rounding))
        return total

    def hash_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not config:
            return {}
        hashed: Dict[str, Any] = {"usd": float(config.get("usd", 0.0))}
        if "round" in config and config["round"] is not None:
            hashed["round"] = int(config["round"])
        return hashed


pricing = PerPagePricing()


__all__ = ["pricing", "PerPagePricing"]
