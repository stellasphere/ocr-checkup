from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from ocrcheckup.families.base import ModelFamily


class MistralOCRFields(BaseModel):
    model: str = Field(
        default="mistral-ocr-2503",
        description="Mistral OCR model identifier.",
    )
    parse_markdown: bool = Field(
        default=True,
        description="If true, convert Markdown output into plain text.",
    )
    max_pages: Optional[int] = Field(
        default=None,
        ge=1,
        description="Optional limit on pages to aggregate when processing multi-page responses.",
    )


class MistralOCRFamily:
    family_id = "mistral-ocr"
    display_name = "Mistral OCR"
    description = "Hosted OCR offering from Mistral AI."
    family_schema_version = "1"
    fields_schema = MistralOCRFields

    def validate_fields(self, fields: dict) -> dict:
        return MistralOCRFields.model_validate(fields).model_dump(
            mode="json", by_alias=True, exclude_none=True
        )


family = MistralOCRFamily()


__all__ = [
    "MistralOCRFields",
    "MistralOCRFamily",
    "family",
]
