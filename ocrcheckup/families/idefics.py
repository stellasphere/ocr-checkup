from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from ocrcheckup.core.variant import AdapterRef, PricingRef, Variant
from ocrcheckup.families.base import BaseFamily
from ocrcheckup.prompts import PROMPTS

DEFAULT_PROMPT = PROMPTS["classic"]


class IdeficsFields(BaseModel):
    model_version: str = Field(default="HuggingFaceM4/idefics2-8b")
    prompt: str | None = None


class IdeficsFamily(BaseFamily):
    family_id = "idefics2"
    display_name = "Idefics2"
    description = "Local Idefics2 model"
    family_schema_version = "1"
    fields_schema = IdeficsFields


idefics_family = IdeficsFamily()

families = [idefics_family]

variants: List[Variant] = [
    Variant(
        name="idefics2",
        family_id=idefics_family.family_id,
        fields={
            "model_version": "HuggingFaceM4/idefics2-8b",
            "prompt": DEFAULT_PROMPT,
        },
        adapter=AdapterRef(id="idefics"),
        pricing=PricingRef(id="local"),
    ),
]

