from __future__ import annotations

from typing import Any, Dict, Optional

from ocrcheckup.core.variant import Variant


class ElapsedSecondsPricing:
    id = "elapsed-seconds"
    description = "Pricing based on elapsed_seconds metadata"

    def quote(self, prediction_record: dict, variant: Variant, config: Optional[Dict[str, Any]]) -> float | None:
        if not config or "usd_per_second" not in config:
            raise ValueError("elapsed-seconds pricing requires config with {'usd_per_second': float}")
        metadata = prediction_record.get("metadata") or {}
        seconds = metadata.get("elapsed_seconds")
        if seconds is None:
            return None
        rate = float(config["usd_per_second"])
        total = float(seconds) * rate
        rounding = config.get("round")
        if rounding is not None:
            total = round(total, int(rounding))
        return total

    def hash_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not config:
            return {}
        hashed: Dict[str, Any] = {"usd_per_second": float(config.get("usd_per_second", 0.0))}
        if "round" in config and config["round"] is not None:
            hashed["round"] = int(config["round"])
        return hashed


pricing = ElapsedSecondsPricing()


__all__ = ["pricing", "ElapsedSecondsPricing"]
