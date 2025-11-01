from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from ocrcheckup.families.base import ModelFamily


class EasyOCRFields(BaseModel):
    languages: List[str] = Field(
        default_factory=lambda: ["en"],
        description="Language codes to load in the EasyOCR reader.",
    )
    paragraph: bool = Field(
        default=False,
        description="Merge results into paragraphs before concatenation.",
    )


class EasyOCRFamily(ModelFamily):
    family_id = "easyocr"
    display_name = "EasyOCR"
    description = "EasyOCR reader for local OCR inference."
    family_schema_version = "2"
    fields_schema = EasyOCRFields

    def validate_fields(self, fields: dict) -> dict:
        return EasyOCRFields.model_validate(fields).model_dump(
            mode="json", by_alias=True, exclude_none=True
        )


family = EasyOCRFamily()


__all__ = ["EasyOCRFields", "EasyOCRFamily", "family"]
