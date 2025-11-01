from __future__ import annotations

from ocrcheckup.core.variant import AdapterRef, PricingRef, Variant


PER_PAGE_PRICING = {
    "usd": 0.12,
    "round": 4,
}


VARIANTS = [
    Variant(
        name="mistral-ocr-markdown-parsed",
        family_id="mistral-ocr-2503",
        fields={
            "parse_markdown": True,
            "max_pages": 1,
        },
        adapter=AdapterRef(id="mistral-ocr-api", config={"rpm": 360}),
        pricing=PricingRef(id="per-page", config=PER_PAGE_PRICING.copy()),
    ),
    Variant(
        name="mistral-ocr-raw-markdown",
        family_id="mistral-ocr-2503",
        fields={
            "parse_markdown": False,
            "max_pages": 1,
        },
        adapter=AdapterRef(id="mistral-ocr-api", config={"rpm": 360}),
        pricing=PricingRef(id="per-page", config=PER_PAGE_PRICING.copy()),
    ),
]


__all__ = ["VARIANTS"]
