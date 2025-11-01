from __future__ import annotations

from ocrcheckup.core.variant import AdapterRef, PricingRef, Variant


VARIANTS = [
    Variant(
        name="trocr-base-printed",
        family_id="microsoft-trocr-base-printed",
        fields={
            "max_new_tokens": 128,
        },
        adapter=AdapterRef(id="trocr-transformers", config={}),
        pricing=PricingRef(id="local"),
    ),
    Variant(
        name="trocr-base-printed-long",
        family_id="microsoft-trocr-base-printed",
        fields={
            "max_new_tokens": 192,
        },
        adapter=AdapterRef(id="trocr-transformers", config={}),
        pricing=PricingRef(id="local"),
    ),
]


__all__ = ["VARIANTS"]
