from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from ocrcheckup.core.variant import AdapterRef, PricingRef, Variant
from ocrcheckup.families.base import BaseFamily


class EasyOCRFields(BaseModel):
    langs: list[str] = Field(default_factory=lambda: ["en"])
    gpu: bool | None = None


class EasyOCR(BaseFamily):
    family_id = "easyocr"
    display_name = "EasyOCR"
    description = "Local EasyOCR adapter"
    family_schema_version = "1"
    fields_schema = EasyOCRFields


easyocr_family = EasyOCR()

families = [easyocr_family]

variants: List[Variant] = [
    Variant(
        name="easyocr-en",
        family_id=easyocr_family.family_id,
        fields={"langs": ["en"], "gpu": False},
        adapter=AdapterRef(id="easyocr"),
        pricing=PricingRef(id="local"),
    ),
]

