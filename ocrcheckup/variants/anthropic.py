from __future__ import annotations

from ocrcheckup.core.variant import AdapterRef, PricingRef, Variant
from ocrcheckup.utils.prompts import DEFAULT_OCR_INSTRUCTION


TOKEN_PRICING = {
    "input_per_million": 8.0,
    "output_per_million": 24.0,
    "per_request": 0.0,
    "round": 6,
}


def _variant(name: str, family_id: str, max_tokens: int = 1024) -> Variant:
    return Variant(
        name=name,
        family_id=family_id,
        fields={
            "prompt": DEFAULT_OCR_INSTRUCTION,
            "max_output_tokens": max_tokens,
        },
        adapter=AdapterRef(id="anthropic-claude-messages", config={"rpm": 60}),
        pricing=PricingRef(id="token-usage", config=TOKEN_PRICING.copy()),
    )


VARIANTS = [
    _variant("anthropic-claude-3-opus", "anthropic-claude-3-opus"),
    _variant("anthropic-claude-3-sonnet", "anthropic-claude-3-sonnet"),
    _variant("anthropic-claude-3-haiku", "anthropic-claude-3-haiku", max_tokens=768),
    _variant("anthropic-claude-3.5-sonnet", "anthropic-claude-3.5-sonnet"),
    _variant("anthropic-claude-3.5-sonnet-v2", "anthropic-claude-3.5-sonnet-v2"),
    _variant("anthropic-claude-3.5-haiku", "anthropic-claude-3.5-haiku", max_tokens=768),
    _variant("anthropic-claude-3.7-sonnet", "anthropic-claude-3.7-sonnet", max_tokens=1536),
]


__all__ = ["VARIANTS"]
