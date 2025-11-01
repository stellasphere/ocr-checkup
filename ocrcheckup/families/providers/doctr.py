from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from ocrcheckup.families.base import ModelFamily


class DocTRFields(BaseModel):
    api_url: str = Field(
        default="https://infer.roboflow.com",
        description="Endpoint for the Roboflow DocTR hosted inference API.",
    )
    project: Optional[str] = Field(
        default=None,
        description="Optional project identifier to include in logging.",
    )


class DocTRFamily:
    family_id = "doctr-roboflow"
    display_name = "DocTR (Roboflow Hosted)"
    description = "Roboflow-hosted DocTR OCR API."
    family_schema_version = "1"
    fields_schema = DocTRFields

    def validate_fields(self, fields: dict) -> dict:
        return DocTRFields.model_validate(fields).model_dump(
            mode="json", by_alias=True, exclude_none=True
        )


family = DocTRFamily()


__all__ = [
    "DocTRFields",
    "DocTRFamily",
    "family",
]
