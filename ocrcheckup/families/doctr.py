from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from ocrcheckup.core.variant import AdapterRef, PricingRef, Variant
from ocrcheckup.families.base import BaseFamily


class DocTRFields(BaseModel):
    version: str = Field(default="doctr")


class DocTR(BaseFamily):
    family_id = "doctr"
    display_name = "DocTR"
    description = "DocTR OCR model"
    family_schema_version = "1"
    fields_schema = DocTRFields


doctr = DocTR()

families = [doctr]

variants: List[Variant] = [
    Variant(
        name="doctr-classic",
        family_id=doctr.family_id,
        fields={
            "version": "roboflow-hosted",
        },
        adapter=AdapterRef(id="roboflow"),
        pricing=PricingRef(id="roboflow", config={"usd_per_call": 0.001}),
    ),
]

