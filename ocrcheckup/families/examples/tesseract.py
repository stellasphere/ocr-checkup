from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from ocrcheckup.families.base import ModelFamily


class TesseractFields(BaseModel):
    lang: str = Field(default="eng")
    psm: Optional[int] = None


class TesseractFamily:
    family_id = "tesseract"
    display_name = "Tesseract OCR"
    description = None
    family_schema_version = "1"
    fields_schema = TesseractFields

    def validate_fields(self, fields: dict) -> dict:
        return TesseractFields.model_validate(fields).model_dump(
            mode="json", by_alias=True, exclude_none=True
        )


family = TesseractFamily()
