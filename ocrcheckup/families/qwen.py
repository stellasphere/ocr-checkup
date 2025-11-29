from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from ocrcheckup.core.variant import AdapterRef, PricingRef, Variant
from ocrcheckup.families.base import BaseFamily
from ocrcheckup.prompts import PROMPTS

DEFAULT_PROMPT = PROMPTS["classic"]


class QwenFields(BaseModel):
    model_version: str = Field(default="Qwen/Qwen2.5-VL-7B-Instruct")
    prompt: str | None = None


class QwenFamily(BaseFamily):
    family_id = "qwen"
    display_name = "Qwen2.5-VL"
    description = "Local Qwen2.5-VL model"
    family_schema_version = "1"
    fields_schema = QwenFields


qwen_family = QwenFamily()

families = [qwen_family]

variants: List[Variant] = [
    Variant(
        name="qwen2_5-vl",
        family_id=qwen_family.family_id,
        fields={
            "model_version": "Qwen/Qwen2.5-VL-7B-Instruct",
            "prompt": DEFAULT_PROMPT,
        },
        adapter=AdapterRef(id="qwen"),
        pricing=PricingRef(id="local"),
    ),
]

