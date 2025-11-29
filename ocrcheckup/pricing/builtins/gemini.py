from __future__ import annotations

from typing import Dict, Optional

from ocrcheckup.pricing.base import PricingModel
from ocrcheckup.core.variant import Variant


class GeminiPricing:
    id = "gemini"
    description = "Google Gemini pricing (USD per 1K tokens)"

    RATES: Dict[str, Dict[str, float]] = {
        "gemini-1.5-pro": {"input": 3.50, "output": 10.50},
        "gemini-1.5-flash": {"input": 0.35, "output": 1.05},
        "gemini-1.5-flash-8b": {"input": 0.20, "output": 0.60},
        "gemini-2.5-pro-preview-03-25": {"input": 4.00, "output": 12.00},
        "gemini-2.0-flash": {"input": 0.25, "output": 0.75},
        "gemini-2.0-flash-lite": {"input": 0.05, "output": 0.15},
    }

    def quote(
        self,
        prediction_record: dict,
        variant: Variant,
        config: Optional[Dict[str, float]],
    ) -> float | None:
        metadata = prediction_record.get("metadata", {})
        model_version = variant.fields.get("model_version")
        if not model_version:
            raise ValueError("GeminiPricing requires variant.fields['model_version'].")

        usage = self.RATES.get(model_version)
        if usage is None:
            return None

        input_tokens = metadata.get("input_tokens", 0)
        output_tokens = metadata.get("output_tokens", 0)
        return (input_tokens / 1000 * usage["input"]) + (output_tokens / 1000 * usage["output"])

    def hash_config(self, config: Optional[Dict[str, float]]) -> Dict[str, float]:
        return {}


pricing: PricingModel = GeminiPricing()


