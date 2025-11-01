from __future__ import annotations

from typing import Literal, Optional, Type

from pydantic import BaseModel, Field

from ocrcheckup.families.base import ModelFamily
from ocrcheckup.utils.prompts import DEFAULT_OCR_INSTRUCTION


class _AnthropicClaudeFields(BaseModel):
    prompt: str = Field(
        default=DEFAULT_OCR_INSTRUCTION,
        description="Instruction appended to the request.",
    )
    max_output_tokens: Optional[int] = Field(
        default=1024,
        ge=1,
        description="Token budget for the generated response.",
    )


class Claude3OpusFields(_AnthropicClaudeFields):
    model: Literal["claude-3-opus-20240229"] = Field(
        default="claude-3-opus-20240229",
        description="Anthropic model identifier.",
    )


class Claude3SonnetFields(_AnthropicClaudeFields):
    model: Literal["claude-3-sonnet-20240229"] = Field(
        default="claude-3-sonnet-20240229",
        description="Anthropic model identifier.",
    )


class Claude3HaikuFields(_AnthropicClaudeFields):
    model: Literal["claude-3-haiku-20240307"] = Field(
        default="claude-3-haiku-20240307",
        description="Anthropic model identifier.",
    )


class Claude35SonnetFields(_AnthropicClaudeFields):
    model: Literal["claude-3-5-sonnet-20240620"] = Field(
        default="claude-3-5-sonnet-20240620",
        description="Anthropic model identifier.",
    )


class Claude35SonnetV2Fields(_AnthropicClaudeFields):
    model: Literal["claude-3-5-sonnet-20241022"] = Field(
        default="claude-3-5-sonnet-20241022",
        description="Anthropic model identifier.",
    )


class Claude35HaikuFields(_AnthropicClaudeFields):
    model: Literal["claude-3-5-haiku-20241022"] = Field(
        default="claude-3-5-haiku-20241022",
        description="Anthropic model identifier.",
    )


class Claude37SonnetFields(_AnthropicClaudeFields):
    model: Literal["claude-3-7-sonnet-20250219"] = Field(
        default="claude-3-7-sonnet-20250219",
        description="Anthropic model identifier.",
    )


class _AnthropicClaudeFamily(ModelFamily):
    family_schema_version = "2"
    fields_schema: Type[BaseModel]
    family_id: str
    display_name: str
    description: str | None

    def validate_fields(self, fields: dict) -> dict:
        return self.fields_schema.model_validate(fields).model_dump(
            mode="json", by_alias=True, exclude_none=True
        )


class Claude3OpusFamily(_AnthropicClaudeFamily):
    family_id = "anthropic-claude-3-opus"
    display_name = "Claude 3 Opus"
    description = "High-end Claude 3 Opus multimodal model."
    fields_schema = Claude3OpusFields


class Claude3SonnetFamily(_AnthropicClaudeFamily):
    family_id = "anthropic-claude-3-sonnet"
    display_name = "Claude 3 Sonnet"
    description = "Balanced Claude 3 Sonnet multimodal model."
    fields_schema = Claude3SonnetFields


class Claude3HaikuFamily(_AnthropicClaudeFamily):
    family_id = "anthropic-claude-3-haiku"
    display_name = "Claude 3 Haiku"
    description = "Fast Claude 3 Haiku multimodal model."
    fields_schema = Claude3HaikuFields


class Claude35SonnetFamily(_AnthropicClaudeFamily):
    family_id = "anthropic-claude-3.5-sonnet"
    display_name = "Claude 3.5 Sonnet"
    description = "Claude 3.5 Sonnet multimodal model."
    fields_schema = Claude35SonnetFields


class Claude35SonnetV2Family(_AnthropicClaudeFamily):
    family_id = "anthropic-claude-3.5-sonnet-v2"
    display_name = "Claude 3.5 Sonnet v2"
    description = "Latest Claude 3.5 Sonnet variant."
    fields_schema = Claude35SonnetV2Fields


class Claude35HaikuFamily(_AnthropicClaudeFamily):
    family_id = "anthropic-claude-3.5-haiku"
    display_name = "Claude 3.5 Haiku"
    description = "Claude 3.5 Haiku multimodal model."
    fields_schema = Claude35HaikuFields


class Claude37SonnetFamily(_AnthropicClaudeFamily):
    family_id = "anthropic-claude-3.7-sonnet"
    display_name = "Claude 3.7 Sonnet"
    description = "Claude 3.7 Sonnet multimodal model."
    fields_schema = Claude37SonnetFields


claude_3_opus_family = Claude3OpusFamily()
claude_3_sonnet_family = Claude3SonnetFamily()
claude_3_haiku_family = Claude3HaikuFamily()
claude_35_sonnet_family = Claude35SonnetFamily()
claude_35_sonnet_v2_family = Claude35SonnetV2Family()
claude_35_haiku_family = Claude35HaikuFamily()
claude_37_sonnet_family = Claude37SonnetFamily()


__all__ = [
    "claude_3_opus_family",
    "claude_3_sonnet_family",
    "claude_3_haiku_family",
    "claude_35_sonnet_family",
    "claude_35_sonnet_v2_family",
    "claude_35_haiku_family",
    "claude_37_sonnet_family",
    "Claude3OpusFamily",
    "Claude3SonnetFamily",
    "Claude3HaikuFamily",
    "Claude35SonnetFamily",
    "Claude35SonnetV2Family",
    "Claude35HaikuFamily",
    "Claude37SonnetFamily",
]
