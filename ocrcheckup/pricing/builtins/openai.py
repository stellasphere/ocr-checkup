from __future__ import annotations

from typing import Dict, Optional

from ocrcheckup.pricing.base import PricingModel
from ocrcheckup.core.variant import Variant


class OpenAIPricing:
    id = "openai"
    description = "OpenAI snapshot pricing (USD per 1K tokens)"

    RATES: Dict[str, Dict[str, float]] = {
        "gpt-4o-2024-05-13": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini-2024-07-18": {"input": 0.15, "output": 0.60},
        "gpt-4.5-preview-2025-02-27": {"input": 5.00, "output": 15.00},
    }

    def quote(
        self,
        prediction_record: dict,
        variant: Variant,
        config: Optional[Dict[str, float]],
    ) -> float | None:
        metadata = prediction_record.get("metadata", {})
        snapshot = variant.fields.get("snapshot")
        if not snapshot:
            raise ValueError("OpenAIPricing requires variant.fields['snapshot'].")

        usage = self.RATES.get(snapshot)
        if usage is None:
            return None

        input_tokens = metadata.get("input_tokens", 0)
        output_tokens = metadata.get("output_tokens", 0)
        return (input_tokens / 1000 * usage["input"]) + (output_tokens / 1000 * usage["output"])

    def hash_config(self, config: Optional[Dict[str, float]]) -> Dict[str, float]:
        return {}


pricing: PricingModel = OpenAIPricing()


