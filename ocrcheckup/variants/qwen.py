from __future__ import annotations

from ocrcheckup.core.variant import AdapterRef, PricingRef, Variant
from ocrcheckup.utils.prompts import DEFAULT_OCR_INSTRUCTION


VARIANTS = [
    Variant(
        name="qwen2.5-vl-7b-default",
        family_id="qwen2.5-vl-7b",
        fields={
            "prompt": DEFAULT_OCR_INSTRUCTION,
            "max_new_tokens": 512,
        },
        adapter=AdapterRef(id="qwen-vision-transformers", config={"torch_dtype": "float16"}),
        pricing=PricingRef(id="local"),
    ),
    Variant(
        name="qwen2.5-vl-7b-long",
        family_id="qwen2.5-vl-7b",
        fields={
            "prompt": DEFAULT_OCR_INSTRUCTION,
            "max_new_tokens": 640,
        },
        adapter=AdapterRef(id="qwen-vision-transformers", config={"torch_dtype": "bfloat16"}),
        pricing=PricingRef(id="local"),
        description="Extends decoding budget to capture longer reasoning chains",
    ),
]


__all__ = ["VARIANTS"]
