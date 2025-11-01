from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from ocrcheckup.families.base import ModelFamily


class TrOCRFields(BaseModel):
    model: str = Field(
        default="microsoft/trocr-base-printed",
        description="Hugging Face identifier for TrOCR VisionEncoderDecoder checkpoint.",
    )
    device: Optional[str] = Field(
        default=None,
        description="Optional torch device override.",
    )
    max_new_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        description="Optional limit for generated tokens during decoding.",
    )


class TrOCRFamily:
    family_id = "trocr"
    display_name = "TrOCR"
    description = "Microsoft TrOCR encoder-decoder models via Transformers."
    family_schema_version = "1"
    fields_schema = TrOCRFields

    def validate_fields(self, fields: dict) -> dict:
        return TrOCRFields.model_validate(fields).model_dump(
            mode="json", by_alias=True, exclude_none=True
        )


family = TrOCRFamily()


__all__ = [
    "TrOCRFields",
    "TrOCRFamily",
    "family",
]
