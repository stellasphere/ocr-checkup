from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from ocrcheckup.families.base import ModelFamily


class TrOCRFields(BaseModel):
    model: Literal["microsoft/trocr-base-printed"] = Field(
        default="microsoft/trocr-base-printed",
        description="Hugging Face identifier for TrOCR VisionEncoderDecoder checkpoint.",
    )
    max_new_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        description="Optional limit for generated tokens during decoding.",
    )


class TrOCRFamily(ModelFamily):
    family_id = "microsoft-trocr-base-printed"
    display_name = "TrOCR Base (Printed)"
    description = "Microsoft TrOCR encoder-decoder model for printed text."
    family_schema_version = "2"
    fields_schema = TrOCRFields

    def validate_fields(self, fields: dict) -> dict:
        return TrOCRFields.model_validate(fields).model_dump(
            mode="json", by_alias=True, exclude_none=True
        )


family = TrOCRFamily()


__all__ = ["TrOCRFields", "TrOCRFamily", "family"]
