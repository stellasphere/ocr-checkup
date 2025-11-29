from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from ocrcheckup.core.variant import AdapterRef, PricingRef, Variant
from ocrcheckup.families.base import BaseFamily


class TesseractFields(BaseModel):
    lang: str = Field(default="eng")
    psm: int | None = None


class Tesseract(BaseFamily):
    family_id = "tesseract"
    display_name = "Tesseract OCR"
    description = "Local Tesseract OCR"
    family_schema_version = "1"
    fields_schema = TesseractFields


tesseract_family = Tesseract()

families = [tesseract_family]

variants: List[Variant] = [
    Variant(
        name="tesseract-eng-psm6",
        family_id=tesseract_family.family_id,
        fields={"lang": "eng", "psm": 6},
        adapter=AdapterRef(id="tesseract-pytesseract"),
        pricing=PricingRef(id="local"),
    ),
]

