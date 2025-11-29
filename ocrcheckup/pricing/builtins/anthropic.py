from __future__ import annotations

from typing import Dict, Optional

from ocrcheckup.pricing.base import PricingModel
from ocrcheckup.core.variant import Variant


class AnthropicPricing:
    id = "anthropic"
    description = "Anthropic Claude pricing (USD per 1K tokens)"

    RATES: Dict[str, Dict[str, float]] = {
        "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
        "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
        "claude-3-5-sonnet-20240620": {"input": 3.00, "output": 15.00},
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
        "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
        "claude-3-7-sonnet-20250219": {"input": 4.00, "output": 20.00},
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
            raise ValueError("AnthropicPricing requires variant.fields['version'].")

        usage = self.RATES.get(version)
        if usage is None:
            return None

        input_tokens = metadata.get("input_tokens", 0)
        output_tokens = metadata.get("output_tokens", 0)
        return (input_tokens / 1000 * usage["input"]) + (output_tokens / 1000 * usage["output"])

    def hash_config(self, config: Optional[Dict[str, float]]) -> Dict[str, float]:
        return {}


pricing: PricingModel = AnthropicPricing()


