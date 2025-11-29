from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from ocrcheckup.core.variant import AdapterRef, PricingRef, Variant
from ocrcheckup.families.base import BaseFamily


class MistralOCRFields(BaseModel):
    version: str = Field(default="mistral-ocr-2503")


class MistralOCR(BaseFamily):
    family_id = "mistral-ocr"
    display_name = "Mistral OCR"
    description = "Mistral OCR API"
    family_schema_version = "1"
    fields_schema = MistralOCRFields


mistral_ocr = MistralOCR()

families = [mistral_ocr]

variants: List[Variant] = [
    Variant(
        name="mistral-ocr-classic",
        family_id=mistral_ocr.family_id,
        fields={
            "version": "mistral-ocr-2503",
        },
        adapter=AdapterRef(id="mistral"),
        pricing=PricingRef(id="mistral"),
    ),
]

