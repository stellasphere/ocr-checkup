from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from ocrcheckup.families.base import ModelFamily
from ocrcheckup.utils.prompts import DEFAULT_OCR_INSTRUCTION


class OpenAIVisionFields(BaseModel):
    model: str = Field(
        ...,
        description="Fully-qualified OpenAI model identifier (e.g. gpt-4o-2024-05-13)",
    )
    prompt: str = Field(
        default=DEFAULT_OCR_INSTRUCTION,
        description="Instruction prompt provided as the first message to the model.",
    )
    detail: Optional[Literal["auto", "high", "low"]] = Field(
        default="high",
        description="Controls vision detail level passed to the image message.",
    )
    max_output_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        description="Optional upper bound for generated tokens returned by the model.",
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for the response.",
    )
    top_p: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter.",
    )


class OpenAIVisionFamily:
    family_id = "openai-vision"
    display_name = "OpenAI Vision Models"
    description = "Image-capable GPT and o-series models provided by OpenAI."
    family_schema_version = "1"
    fields_schema = OpenAIVisionFields

    def validate_fields(self, fields: dict) -> dict:
        return OpenAIVisionFields.model_validate(fields).model_dump(
            mode="json", by_alias=True, exclude_none=True
        )


family = OpenAIVisionFamily()


__all__ = [
    "OpenAIVisionFields",
    "OpenAIVisionFamily",
    "family",
]
