from __future__ import annotations

from typing import Dict, Optional

from ocrcheckup.core.variant import Variant
from ocrcheckup.pricing.base import PricingModel


class RoboflowPricing:
    id = "roboflow"
    description = "Roboflow workflow pricing"

    def quote(
        self,
        prediction_record: dict,
        variant: Variant,
        config: Optional[Dict[str, float]],
    ) -> float | None:
        if not config:
            raise ValueError("RoboflowPricing requires a config dictionary.")

        if "usd_per_call" in config:
            return float(config["usd_per_call"])

        raise ValueError("RoboflowPricing config must include 'usd_per_call'.")

    def hash_config(self, config: Optional[Dict[str, float]]) -> Dict[str, float]:
        if not config:
            return {}
        sanitized: Dict[str, float] = {}
        if "usd_per_call" in config:
            sanitized["usd_per_call"] = float(config["usd_per_call"])
        return sanitized


pricing: PricingModel = RoboflowPricing()
