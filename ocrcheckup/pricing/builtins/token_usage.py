from __future__ import annotations

from typing import Any, Dict, Optional

from ocrcheckup.core.variant import Variant
from ocrcheckup.pricing.base import PricingModel


class TokenUsagePricing:
    """Pricing model that multiplies recorded token usage with per-million rates."""

    id = "token-usage"
    description = "Token-based usage pricing"

    def quote(self, prediction_record: dict, variant: Variant, config: Optional[Dict[str, Any]]) -> float | None:
        if not config:
            raise ValueError("token-usage pricing requires config with per-million rates")
        metadata = prediction_record.get("metadata") or {}
        usage = metadata.get("usage") or {}
        if not usage:
            return None

        input_tokens = usage.get("prompt_tokens")
        if input_tokens is None:
            input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("completion_tokens")
        if output_tokens is None:
            output_tokens = usage.get("output_tokens")
        if output_tokens is None:
            output_tokens = usage.get("candidates_tokens")

        input_rate = float(config.get("input_per_million", 0.0))
        output_rate = float(config.get("output_per_million", 0.0))
        per_request = float(config.get("per_request", 0.0))
        minimum = float(config.get("minimum", 0.0))

        total = per_request
        if input_tokens is not None:
            total += (float(input_tokens) / 1_000_000.0) * input_rate
        if output_tokens is not None:
            total += (float(output_tokens) / 1_000_000.0) * output_rate

        if total < minimum:
            total = minimum

        rounding = config.get("round")
        if rounding is not None:
            total = round(total, int(rounding))
        return total

    def hash_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not config:
            return {}
        hashed: Dict[str, Any] = {}
        for key in ("input_per_million", "output_per_million", "per_request", "minimum", "round"):
            if key in config and config[key] is not None:
                hashed[key] = float(config[key]) if key != "round" else int(config[key])
        return hashed


pricing = TokenUsagePricing()


__all__ = ["pricing", "TokenUsagePricing"]
