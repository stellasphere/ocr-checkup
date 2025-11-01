from __future__ import annotations

from ocrcheckup.core.variant import AdapterRef, PricingRef, Variant


VARIANTS = [
    Variant(
        name="easyocr-english",
        family_id="easyocr",
        fields={
            "languages": ["en"],
            "paragraph": False,
        },
        adapter=AdapterRef(id="easyocr-reader", config={}),
        pricing=PricingRef(id="local"),
    ),
    Variant(
        name="easyocr-english-spanish",
        family_id="easyocr",
        fields={
            "languages": ["en", "es"],
            "paragraph": True,
        },
        adapter=AdapterRef(id="easyocr-reader", config={"gpu": True}),
        pricing=PricingRef(id="local"),
    ),
]


__all__ = ["VARIANTS"]
