from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from ocrcheckup.core.variant import AdapterRef, PricingRef, Variant
from ocrcheckup.families.base import BaseFamily
from ocrcheckup.prompts import PROMPTS

DEFAULT_PROMPT = PROMPTS["classic"]


class Florence2LargeFields(BaseModel):
    version: str = Field(default="florence-2-large")
    prompt: str | None = None


class Florence2BaseFields(BaseModel):
    version: str = Field(default="florence-2-base")
    prompt: str | None = None


class Florence2Large(BaseFamily):
    family_id = "florence-2-large"
    display_name = "Florence 2 Large"
    description = "Florence 2 Large OCR model"
    family_schema_version = "1"
    fields_schema = Florence2LargeFields


class Florence2Base(BaseFamily):
    family_id = "florence-2-base"
    display_name = "Florence 2 Base"
    description = "Florence 2 Base OCR model"
    family_schema_version = "1"
    fields_schema = Florence2BaseFields


florence2_large = Florence2Large()
florence2_base = Florence2Base()

families = [florence2_large, florence2_base]

variants: List[Variant] = [
    Variant(
        name="florence-2-large-classic",
        family_id=florence2_large.family_id,
        fields={
            "version": "florence-2-large-roboflow-hosted",
            "prompt": DEFAULT_PROMPT,
        },
        adapter=AdapterRef(id="roboflow"),
        pricing=PricingRef(id="roboflow", config={"usd_per_call": 0.002}),
    ),
]

