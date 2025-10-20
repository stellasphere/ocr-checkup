from __future__ import annotations

from typing import Any, Dict, Optional

from ocrcheckup.pricing.base import PricingModel
from ocrcheckup.core.variant import Variant


class FlatPerImagePricing:
    id = "flat-per-image"
    description = "Flat USD per sample"

    def quote(self, prediction_record: dict, variant: Variant, config: Optional[Dict[str, Any]]) -> float | None:
        if not config or "usd" not in config:
            raise ValueError("flat-per-image pricing requires config with {'usd': float}")
        return float(config["usd"])

    def hash_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not config:
            return {}
        return {"usd": float(config.get("usd", 0.0))}


pricing = FlatPerImagePricing()
