from __future__ import annotations

from typing import Literal, Optional, Type

from pydantic import BaseModel, Field

from ocrcheckup.families.base import ModelFamily
from ocrcheckup.utils.prompts import DEFAULT_OCR_INSTRUCTION


DetailType = Literal["auto", "low", "high"]


class _OpenAIVisionFields(BaseModel):
    prompt: str = Field(
        default=DEFAULT_OCR_INSTRUCTION,
        description="Instruction prompt provided to the model.",
    )
    detail: DetailType = Field(
        default="high",
        description="Vision detail level requested from the API.",
    )
    max_output_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        description="Optional cap on generated tokens to guard against truncation.",
    )


class GPT4oFields(_OpenAIVisionFields):
    model: Literal["gpt-4o-2024-05-13"] = Field(
        default="gpt-4o-2024-05-13",
        description="Concrete OpenAI model identifier.",
    )


class GPT4oMiniFields(_OpenAIVisionFields):
    model: Literal["gpt-4o-mini-2024-07-18"] = Field(
        default="gpt-4o-mini-2024-07-18",
        description="Concrete OpenAI model identifier.",
    )


class GPT45PreviewFields(_OpenAIVisionFields):
    model: Literal["gpt-4.5-preview-2025-02-27"] = Field(
        default="gpt-4.5-preview-2025-02-27",
        description="Concrete OpenAI model identifier.",
    )


class O1Fields(_OpenAIVisionFields):
    model: Literal["o1-2024-12-17"] = Field(
        default="o1-2024-12-17",
        description="Concrete OpenAI model identifier.",
    )


class _OpenAIVisionFamily(ModelFamily):
    family_schema_version = "2"
    fields_schema: Type[BaseModel]
    family_id: str
    display_name: str
    description: str | None

    def validate_fields(self, fields: dict) -> dict:
        return self.fields_schema.model_validate(fields).model_dump(
            mode="json", by_alias=True, exclude_none=True
        )


class OpenAIGPT4oFamily(_OpenAIVisionFamily):
    family_id = "openai-gpt-4o"
    display_name = "OpenAI GPT-4o"
    description = "Multimodal GPT-4o model for vision-text understanding."
    fields_schema = GPT4oFields


class OpenAIGPT4oMiniFamily(_OpenAIVisionFamily):
    family_id = "openai-gpt-4o-mini"
    display_name = "OpenAI GPT-4o mini"
    description = "Lightweight GPT-4o variant optimized for cost."
    fields_schema = GPT4oMiniFields


class OpenAIGPT45PreviewFamily(_OpenAIVisionFamily):
    family_id = "openai-gpt-4.5-preview"
    display_name = "OpenAI GPT-4.5 Preview"
    description = "Preview GPT-4.5 multimodal model."
    fields_schema = GPT45PreviewFields


class OpenAIO1Family(_OpenAIVisionFamily):
    family_id = "openai-o1"
    display_name = "OpenAI o1"
    description = "Reasoning-focused OpenAI o1 multimodal model."
    fields_schema = O1Fields


gpt_4o_family = OpenAIGPT4oFamily()
gpt_4o_mini_family = OpenAIGPT4oMiniFamily()
gpt_45_preview_family = OpenAIGPT45PreviewFamily()
o1_family = OpenAIO1Family()


__all__ = [
    "gpt_4o_family",
    "gpt_4o_mini_family",
    "gpt_45_preview_family",
    "o1_family",
    "OpenAIGPT4oFamily",
    "OpenAIGPT4oMiniFamily",
    "OpenAIGPT45PreviewFamily",
    "OpenAIO1Family",
]
