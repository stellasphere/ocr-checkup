from __future__ import annotations

from ocrcheckup.core.variant import AdapterRef, PricingRef, Variant
from ocrcheckup.utils.prompts import DEFAULT_OCR_INSTRUCTION


TOKEN_PRICING = {
    "input_per_million": 5.0,
    "output_per_million": 15.0,
    "per_request": 0.0,
    "round": 6,
}


VARIANTS = [
    Variant(
        name="openai-gpt-4o-high-detail",
        family_id="openai-gpt-4o",
        fields={
            "prompt": DEFAULT_OCR_INSTRUCTION,
            "detail": "high",
        },
        adapter=AdapterRef(id="openai-vision-chat", config={"rpm": 240}),
        pricing=PricingRef(id="token-usage", config=TOKEN_PRICING.copy()),
        description="GPT-4o with high-detail vision rendering",
    ),
    Variant(
        name="openai-gpt-4o-auto-detail",
        family_id="openai-gpt-4o",
        fields={
            "prompt": DEFAULT_OCR_INSTRUCTION,
            "detail": "auto",
        },
        adapter=AdapterRef(id="openai-vision-chat", config={"rpm": 240}),
        pricing=PricingRef(id="token-usage", config=TOKEN_PRICING.copy()),
        description="GPT-4o with automatic detail level to benchmark quality trade-offs",
    ),
    Variant(
        name="openai-gpt-4o-mini-high-detail",
        family_id="openai-gpt-4o-mini",
        fields={
            "prompt": DEFAULT_OCR_INSTRUCTION,
            "detail": "high",
        },
        adapter=AdapterRef(id="openai-vision-chat", config={"rpm": 360}),
        pricing=PricingRef(id="token-usage", config=TOKEN_PRICING.copy()),
        description="GPT-4o mini with high-detail rendering for cost/performance comparison",
    ),
    Variant(
        name="openai-gpt-4.5-preview-high-detail",
        family_id="openai-gpt-4.5-preview",
        fields={
            "prompt": DEFAULT_OCR_INSTRUCTION,
            "detail": "high",
        },
        adapter=AdapterRef(id="openai-vision-chat", config={"rpm": 240}),
        pricing=PricingRef(id="token-usage", config=TOKEN_PRICING.copy()),
        description="GPT-4.5 preview in high-detail mode",
    ),
    Variant(
        name="openai-o1-reasoning",
        family_id="openai-o1",
        fields={
            "prompt": DEFAULT_OCR_INSTRUCTION,
            "detail": "high",
            "max_output_tokens": 1024,
        },
        adapter=AdapterRef(id="openai-vision-chat", config={"rpm": 180}),
        pricing=PricingRef(id="token-usage", config=TOKEN_PRICING.copy()),
        description="OpenAI o1 model emphasising longer reasoning budget",
    ),
]


__all__ = ["VARIANTS"]
