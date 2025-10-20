from __future__ import annotations

from typing import Any, Dict, Optional

from ocrcheckup.pricing.base import PricingModel
from ocrcheckup.core.variant import Variant


class LocalPricing:
    id = "local"
    description = "No cost (local inference)"

    def quote(self, prediction_record: dict, variant: Variant, config: Optional[Dict[str, Any]]) -> float | None:
        return 0.0

    def hash_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        return {}


pricing = LocalPricing()
