from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from ocrcheckup.families.base import ModelFamily


class EasyOCRFields(BaseModel):
    languages: List[str] = Field(
        default_factory=lambda: ["en"],
        description="Collection of ISO language codes to load in the reader.",
    )
    gpu: Optional[bool] = Field(
        default=None,
        description="Override for GPU usage. Auto-detected when omitted.",
    )
    detail: int = Field(
        default=0,
        ge=0,
        le=2,
        description="Detail level parameter passed to EasyOCR reader.",
    )
    paragraph: bool = Field(
        default=False,
        description="If true, merge results into paragraphs before concatenation.",
    )


class EasyOCRFamily:
    family_id = "easyocr"
    display_name = "EasyOCR"
    description = "EasyOCR reader for local OCR inference."
    family_schema_version = "1"
    fields_schema = EasyOCRFields

    def validate_fields(self, fields: dict) -> dict:
        return EasyOCRFields.model_validate(fields).model_dump(
            mode="json", by_alias=True, exclude_none=True
        )


family = EasyOCRFamily()


__all__ = [
    "EasyOCRFields",
    "EasyOCRFamily",
    "family",
]
