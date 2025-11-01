from __future__ import annotations

from ocrcheckup.core.variant import AdapterRef, PricingRef, Variant
from ocrcheckup.utils.prompts import DEFAULT_OCR_INSTRUCTION


VARIANTS = [
    Variant(
        name="moondream2-default",
        family_id="moondream2",
        fields={
            "prompt": DEFAULT_OCR_INSTRUCTION,
            "revision": "2025-03-27",
        },
        adapter=AdapterRef(id="moondream-transformers", config={}),
        pricing=PricingRef(id="local"),
    ),
]


__all__ = ["VARIANTS"]
