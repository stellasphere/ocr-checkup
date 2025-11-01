from __future__ import annotations

from ocrcheckup.core.variant import AdapterRef, PricingRef, Variant
from ocrcheckup.utils.prompts import DEFAULT_OCR_INSTRUCTION


VARIANTS = [
    Variant(
        name="idefics2-8b-standard",
        family_id="idefics2-8b",
        fields={
            "prompt": DEFAULT_OCR_INSTRUCTION,
            "max_new_tokens": 128,
        },
        adapter=AdapterRef(id="idefics2-transformers", config={"torch_dtype": "float16"}),
        pricing=PricingRef(id="local"),
    ),
    Variant(
        name="idefics2-8b-extended",
        family_id="idefics2-8b",
        fields={
            "prompt": DEFAULT_OCR_INSTRUCTION,
            "max_new_tokens": 192,
        },
        adapter=AdapterRef(id="idefics2-transformers", config={"torch_dtype": "bfloat16"}),
        pricing=PricingRef(id="local"),
    ),
]


__all__ = ["VARIANTS"]
