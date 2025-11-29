from __future__ import annotations

from typing import Dict, Optional

from ocrcheckup.pricing.base import PricingModel
from ocrcheckup.core.variant import Variant


class MistralPricing:
    id = "mistral"
    description = "Mistral OCR pricing (USD per processed page)"

    RATES: Dict[str, float] = {
        "mistral-ocr-2503": 0.002,
    }

    def quote(
        self,
        prediction_record: dict,
        variant: Variant,
        config: Optional[Dict[str, float]],
    ) -> float | None:
        metadata = prediction_record.get("metadata", {})
        version = variant.fields.get("version")
        if not version:
            raise ValueError("MistralPricing requires variant.fields['version'].")

        rate = self.RATES.get(version)
        if rate is None:
            return None

        pages = metadata.get("pages_processed", 0)
        return float(pages) * rate

    def hash_config(self, config: Optional[Dict[str, float]]) -> Dict[str, float]:
        return {}


pricing: PricingModel = MistralPricing()


