from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from ocrcheckup.core.variant import AdapterRef, PricingRef, Variant
from ocrcheckup.families.base import BaseFamily
from ocrcheckup.prompts import PROMPTS

DEFAULT_PROMPT = PROMPTS["classic"]


class MoondreamFields(BaseModel):
    model_version: str = Field(default="vikhyatk/moondream2")
    prompt: str | None = None


class MoondreamFamily(BaseFamily):
    family_id = "moondream2"
    display_name = "Moondream2"
    description = "Local Moondream2 model"
    family_schema_version = "1"
    fields_schema = MoondreamFields


moondream_family = MoondreamFamily()

families = [moondream_family]

variants: List[Variant] = [
    Variant(
        name="moondream2",
        family_id=moondream_family.family_id,
        fields={
            "model_version": "vikhyatk/moondream2",
            "prompt": DEFAULT_PROMPT,
        },
        adapter=AdapterRef(id="moondream"),
        pricing=PricingRef(id="local"),
    ),
]

