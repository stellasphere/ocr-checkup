from __future__ import annotations

from typing import Literal, Optional, Type

from pydantic import BaseModel, Field

from ocrcheckup.families.base import ModelFamily
from ocrcheckup.utils.prompts import DEFAULT_OCR_INSTRUCTION


class _GeminiVisionFields(BaseModel):
    prompt: str = Field(
        default=DEFAULT_OCR_INSTRUCTION,
        description="Prompt sent alongside the image input.",
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for decoding.",
    )
    max_output_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        description="Token cap for the generated response.",
    )


class Gemini15ProFields(_GeminiVisionFields):
    model: Literal["gemini-1.5-pro"] = Field(
        default="gemini-1.5-pro",
        description="Gemini model identifier.",
    )


class Gemini15FlashFields(_GeminiVisionFields):
    model: Literal["gemini-1.5-flash"] = Field(
        default="gemini-1.5-flash",
        description="Gemini model identifier.",
    )


class Gemini15Flash8BFields(_GeminiVisionFields):
    model: Literal["gemini-1.5-flash-8b"] = Field(
        default="gemini-1.5-flash-8b",
        description="Gemini model identifier.",
    )


class Gemini25ProPreviewFields(_GeminiVisionFields):
    model: Literal["gemini-2.5-pro-preview-03-25"] = Field(
        default="gemini-2.5-pro-preview-03-25",
        description="Gemini model identifier.",
    )


class Gemini20FlashFields(_GeminiVisionFields):
    model: Literal["gemini-2.0-flash"] = Field(
        default="gemini-2.0-flash",
        description="Gemini model identifier.",
    )


class Gemini20FlashLiteFields(_GeminiVisionFields):
    model: Literal["gemini-2.0-flash-lite"] = Field(
        default="gemini-2.0-flash-lite",
        description="Gemini model identifier.",
    )


class _GeminiVisionFamily(ModelFamily):
    family_schema_version = "2"
    fields_schema: Type[BaseModel]
    family_id: str
    display_name: str
    description: str | None

    def validate_fields(self, fields: dict) -> dict:
        return self.fields_schema.model_validate(fields).model_dump(
            mode="json", by_alias=True, exclude_none=True
        )


class Gemini15ProFamily(_GeminiVisionFamily):
    family_id = "google-gemini-1.5-pro"
    display_name = "Gemini 1.5 Pro"
    description = "Gemini 1.5 Pro multimodal model."
    fields_schema = Gemini15ProFields


class Gemini15FlashFamily(_GeminiVisionFamily):
    family_id = "google-gemini-1.5-flash"
    display_name = "Gemini 1.5 Flash"
    description = "Gemini 1.5 Flash multimodal model."
    fields_schema = Gemini15FlashFields


class Gemini15Flash8BFamily(_GeminiVisionFamily):
    family_id = "google-gemini-1.5-flash-8b"
    display_name = "Gemini 1.5 Flash-8B"
    description = "Gemini 1.5 Flash 8B multimodal model."
    fields_schema = Gemini15Flash8BFields


class Gemini25ProPreviewFamily(_GeminiVisionFamily):
    family_id = "google-gemini-2.5-pro-preview"
    display_name = "Gemini 2.5 Pro Preview"
    description = "Preview Gemini 2.5 Pro multimodal model."
    fields_schema = Gemini25ProPreviewFields


class Gemini20FlashFamily(_GeminiVisionFamily):
    family_id = "google-gemini-2.0-flash"
    display_name = "Gemini 2.0 Flash"
    description = "Gemini 2.0 Flash multimodal model."
    fields_schema = Gemini20FlashFields


class Gemini20FlashLiteFamily(_GeminiVisionFamily):
    family_id = "google-gemini-2.0-flash-lite"
    display_name = "Gemini 2.0 Flash Lite"
    description = "Gemini 2.0 Flash Lite multimodal model."
    fields_schema = Gemini20FlashLiteFields


gemini_15_pro_family = Gemini15ProFamily()
gemini_15_flash_family = Gemini15FlashFamily()
gemini_15_flash_8b_family = Gemini15Flash8BFamily()
gemini_25_pro_preview_family = Gemini25ProPreviewFamily()
gemini_20_flash_family = Gemini20FlashFamily()
gemini_20_flash_lite_family = Gemini20FlashLiteFamily()


__all__ = [
    "gemini_15_pro_family",
    "gemini_15_flash_family",
    "gemini_15_flash_8b_family",
    "gemini_25_pro_preview_family",
    "gemini_20_flash_family",
    "gemini_20_flash_lite_family",
    "Gemini15ProFamily",
    "Gemini15FlashFamily",
    "Gemini15Flash8BFamily",
    "Gemini25ProPreviewFamily",
    "Gemini20FlashFamily",
    "Gemini20FlashLiteFamily",
]
