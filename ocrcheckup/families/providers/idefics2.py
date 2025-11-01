from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ocrcheckup.families.base import ModelFamily
from ocrcheckup.utils.prompts import DEFAULT_OCR_INSTRUCTION


class Idefics2Fields(BaseModel):
    model: Literal["HuggingFaceM4/idefics2-8b"] = Field(
        default="HuggingFaceM4/idefics2-8b",
        description="Hugging Face identifier for Idefics2 checkpoint.",
    )
    prompt: str = Field(
        default=DEFAULT_OCR_INSTRUCTION,
        description="Instruction appended before generation.",
    )
    max_new_tokens: int = Field(
        default=128,
        ge=1,
        le=2048,
        description="Generation budget for decoder tokens.",
    )


class Idefics2Family(ModelFamily):
    family_id = "idefics2-8b"
    display_name = "Idefics2 8B"
    description = "Idefics2 multimodal checkpoint via Transformers."
    family_schema_version = "2"
    fields_schema = Idefics2Fields

    def validate_fields(self, fields: dict) -> dict:
        return Idefics2Fields.model_validate(fields).model_dump(
            mode="json", by_alias=True, exclude_none=True
        )


family = Idefics2Family()


__all__ = ["Idefics2Fields", "Idefics2Family", "family"]
