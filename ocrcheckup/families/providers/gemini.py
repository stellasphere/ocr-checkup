from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from ocrcheckup.families.base import ModelFamily
from ocrcheckup.utils.prompts import DEFAULT_OCR_INSTRUCTION


class GeminiVisionFields(BaseModel):
    model: str = Field(
        ...,
        description="Gemini vision model identifier (e.g. gemini-1.5-pro)",
    )
    prompt: str = Field(
        default=DEFAULT_OCR_INSTRUCTION,
        description="Prompt provided alongside the image.",
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for content generation.",
    )
    max_output_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        description="Optional cap for generated tokens.",
    )
    top_p: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling probability.",
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        description="Limits the number of highest-probability tokens considered.",
    )


class GeminiVisionFamily:
    family_id = "google-gemini"
    display_name = "Google Gemini Vision"
    description = "Gemini multimodal models served from the Google AI SDK."
    family_schema_version = "1"
    fields_schema = GeminiVisionFields

    def validate_fields(self, fields: dict) -> dict:
        return GeminiVisionFields.model_validate(fields).model_dump(
            mode="json", by_alias=True, exclude_none=True
        )


family = GeminiVisionFamily()


__all__ = [
    "GeminiVisionFields",
    "GeminiVisionFamily",
    "family",
]
