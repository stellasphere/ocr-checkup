from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ocrcheckup.families.base import ModelFamily
from ocrcheckup.utils.prompts import DEFAULT_OCR_INSTRUCTION


class QwenVisionFields(BaseModel):
    model: Literal["Qwen/Qwen2.5-VL-7B-Instruct"] = Field(
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        description="Hugging Face identifier for the Qwen2.5 vision-language checkpoint.",
    )
    prompt: str = Field(
        default=DEFAULT_OCR_INSTRUCTION,
        description="Prompt appended to the visual conversation.",
    )
    max_new_tokens: int = Field(
        default=512,
        ge=1,
        description="Generation budget for decoder output tokens.",
    )


class QwenVisionFamily(ModelFamily):
    family_id = "qwen2.5-vl-7b"
    display_name = "Qwen2.5-VL 7B"
    description = "Open Qwen2.5 7B vision-language checkpoint via Transformers."
    family_schema_version = "2"
    fields_schema = QwenVisionFields

    def validate_fields(self, fields: dict) -> dict:
        return QwenVisionFields.model_validate(fields).model_dump(
            mode="json", by_alias=True, exclude_none=True
        )


family = QwenVisionFamily()


__all__ = ["QwenVisionFields", "QwenVisionFamily", "family"]
