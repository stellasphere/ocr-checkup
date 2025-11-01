from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from ocrcheckup.families.base import ModelFamily
from ocrcheckup.utils.prompts import DEFAULT_OCR_INSTRUCTION


class QwenVisionFields(BaseModel):
    model: str = Field(
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        description="Hugging Face model identifier for Qwen vision-language checkpoint.",
    )
    prompt: str = Field(
        default=DEFAULT_OCR_INSTRUCTION,
        description="Prompt appended to the visual conversation.",
    )
    device: Optional[str] = Field(
        default=None,
        description="Force a specific device (e.g. cuda, cpu, mps). Auto-detected when omitted.",
    )
    max_new_tokens: int = Field(
        default=512,
        ge=1,
        description="Generation budget for decoder output tokens.",
    )
    dtype: Optional[Literal["float16", "bfloat16", "float32", "auto"]] = Field(
        default="auto",
        description="Torch dtype to load model parameters with.",
    )


class QwenVisionFamily:
    family_id = "qwen-vision"
    display_name = "Qwen Vision-Language"
    description = "Open Qwen2.5 vision-language checkpoints via Transformers."
    family_schema_version = "1"
    fields_schema = QwenVisionFields

    def validate_fields(self, fields: dict) -> dict:
        return QwenVisionFields.model_validate(fields).model_dump(
            mode="json", by_alias=True, exclude_none=True
        )


family = QwenVisionFamily()


__all__ = [
    "QwenVisionFields",
    "QwenVisionFamily",
    "family",
]
