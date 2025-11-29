from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from ocrcheckup.core.variant import AdapterRef, PricingRef, Variant
from ocrcheckup.families.base import BaseFamily


class TrOCRFields(BaseModel):
    model_version: str = Field(default="microsoft/trocr-base-printed")


class TrOCRFamily(BaseFamily):
    family_id = "trocr"
    display_name = "TrOCR"
    description = "Local TrOCR model"
    family_schema_version = "1"
    fields_schema = TrOCRFields


trocr_family = TrOCRFamily()

families = [trocr_family]

variants: List[Variant] = [
    Variant(
        name="trocr-base",
        family_id=trocr_family.family_id,
        fields={"model_version": "microsoft/trocr-base-printed"},
        adapter=AdapterRef(id="trocr"),
        pricing=PricingRef(id="local"),
    ),
]

