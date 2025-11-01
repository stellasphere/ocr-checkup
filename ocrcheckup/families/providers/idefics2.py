from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from ocrcheckup.families.base import ModelFamily
from ocrcheckup.utils.prompts import DEFAULT_OCR_INSTRUCTION


class Idefics2Fields(BaseModel):
    model: str = Field(
        default="HuggingFaceM4/idefics2-8b",
        description="Hugging Face identifier for Idefics2 checkpoint.",
    )
    torch_dtype: Optional[str] = Field(
        default="float16",
        description="Torch dtype hint when loading weights.",
    )
    device: Optional[str] = Field(
        default=None,
        description="Optional device override.",
    )
    prompt: str = Field(
        default=DEFAULT_OCR_INSTRUCTION,
        description="Instruction appended to chat template before generation.",
    )
    max_new_tokens: int = Field(
        default=128,
        ge=1,
        le=2048,
        description="Generation budget for decoder tokens.",
    )


class Idefics2Family:
    family_id = "idefics2"
    display_name = "Idefics2"
    description = "Idefics2 multimodal checkpoint via Transformers."
    family_schema_version = "1"
    fields_schema = Idefics2Fields

    def validate_fields(self, fields: dict) -> dict:
        return Idefics2Fields.model_validate(fields).model_dump(
            mode="json", by_alias=True, exclude_none=True
        )


family = Idefics2Family()


__all__ = [
    "Idefics2Fields",
    "Idefics2Family",
    "family",
]
