from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from ocrcheckup.families.base import ModelFamily
from ocrcheckup.utils.prompts import DEFAULT_OCR_INSTRUCTION


class AnthropicClaudeFields(BaseModel):
    model: str = Field(
        ...,
        description="Anthropic Claude model identifier (e.g. claude-3-5-sonnet-20241022)",
    )
    prompt: str = Field(
        default=DEFAULT_OCR_INSTRUCTION,
        description="Instruction to prepend to the conversation.",
    )
    max_output_tokens: Optional[int] = Field(
        default=1024,
        ge=1,
        description="Maximum tokens to generate in the response.",
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Sampling temperature passed to Claude API.",
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        description="Limits highest-probability tokens considered during sampling.",
    )
    top_p: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter.",
    )


class AnthropicClaudeFamily:
    family_id = "anthropic-claude"
    display_name = "Anthropic Claude"
    description = "Vision-capable Claude models for OCR via Anthropic API."
    family_schema_version = "1"
    fields_schema = AnthropicClaudeFields

    def validate_fields(self, fields: dict) -> dict:
        return AnthropicClaudeFields.model_validate(fields).model_dump(
            mode="json", by_alias=True, exclude_none=True
        )


family = AnthropicClaudeFamily()


__all__ = [
    "AnthropicClaudeFields",
    "AnthropicClaudeFamily",
    "family",
]
