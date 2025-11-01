from __future__ import annotations

from ocrcheckup.core.variant import AdapterRef, PricingRef, Variant


VARIANTS = [
    Variant(
        name="tesseract-eng-psm6",
        family_id="tesseract",
        fields={
            "lang": "eng",
            "psm": 6,
        },
        adapter=AdapterRef(id="tesseract-pytesseract", config={}),
        pricing=PricingRef(id="local"),
    ),
    Variant(
        name="tesseract-eng-psm11",
        family_id="tesseract",
        fields={
            "lang": "eng",
            "psm": 11,
        },
        adapter=AdapterRef(id="tesseract-pytesseract", config={}),
        pricing=PricingRef(id="local"),
        description="PSM 11 (sparse text) for comparison on cluttered scenes",
    ),
]


__all__ = ["VARIANTS"]
