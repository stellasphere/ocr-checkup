from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ocrcheckup.families.base import ModelFamily


class DocTRFields(BaseModel):
    model: Literal["roboflow-doctr"] = Field(
        default="roboflow-doctr",
        description="Identifier used for documentation purposes.",
    )


class DocTRFamily(ModelFamily):
    family_id = "doctr-roboflow-hosted"
    display_name = "DocTR (Roboflow Hosted)"
    description = "Roboflow-hosted DocTR OCR API."
    family_schema_version = "2"
    fields_schema = DocTRFields

    def validate_fields(self, fields: dict) -> dict:
        return DocTRFields.model_validate(fields).model_dump(
            mode="json", by_alias=True, exclude_none=True
        )


family = DocTRFamily()


__all__ = ["DocTRFields", "DocTRFamily", "family"]
