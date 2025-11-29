from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from ocrcheckup.core.variant import AdapterRef, PricingRef, Variant
from ocrcheckup.families.base import BaseFamily
from ocrcheckup.prompts import PROMPTS

DEFAULT_PROMPT = PROMPTS["classic"]


class _AnthropicFieldsBase(BaseModel):
    version: str
    prompt: str | None = None


class Claude3OpusFields(_AnthropicFieldsBase):
    version: str = Field(default="claude-3-opus-20240229")


class Claude3SonnetFields(_AnthropicFieldsBase):
    version: str = Field(default="claude-3-sonnet-20240229")


class Claude3HaikuFields(_AnthropicFieldsBase):
    version: str = Field(default="claude-3-haiku-20240307")


class Claude3_5SonnetFields(_AnthropicFieldsBase):
    version: str = Field(default="claude-3-5-sonnet-20240620")


class Claude3_5HaikuFields(_AnthropicFieldsBase):
    version: str = Field(default="claude-3-5-haiku-20241022")


class Claude3_7SonnetFields(_AnthropicFieldsBase):
    version: str = Field(default="claude-3-7-sonnet-20250219")


class Claude3Opus(BaseFamily):
    family_id = "anthropic-claude-3-opus"
    display_name = "Claude 3 Opus"
    description = "Anthropic Claude 3 Opus vision model"
    family_schema_version = "1"
    fields_schema = Claude3OpusFields


class Claude3Sonnet(BaseFamily):
    family_id = "anthropic-claude-3-sonnet"
    display_name = "Claude 3 Sonnet"
    description = "Anthropic Claude 3 Sonnet vision model"
    family_schema_version = "1"
    fields_schema = Claude3SonnetFields


class Claude3Haiku(BaseFamily):
    family_id = "anthropic-claude-3-haiku"
    display_name = "Claude 3 Haiku"
    description = "Anthropic Claude 3 Haiku vision model"
    family_schema_version = "1"
    fields_schema = Claude3HaikuFields


class Claude3_5Sonnet(BaseFamily):
    family_id = "anthropic-claude-3.5-sonnet"
    display_name = "Claude 3.5 Sonnet"
    description = "Anthropic Claude 3.5 Sonnet vision model"
    family_schema_version = "1"
    fields_schema = Claude3_5SonnetFields


class Claude3_5Haiku(BaseFamily):
    family_id = "anthropic-claude-3.5-haiku"
    display_name = "Claude 3.5 Haiku"
    description = "Anthropic Claude 3.5 Haiku vision model"
    family_schema_version = "1"
    fields_schema = Claude3_5HaikuFields


class Claude3_7Sonnet(BaseFamily):
    family_id = "anthropic-claude-3.7-sonnet"
    display_name = "Claude 3.7 Sonnet"
    description = "Anthropic Claude 3.7 Sonnet vision model"
    family_schema_version = "1"
    fields_schema = Claude3_7SonnetFields


anthropic_claude3_opus = Claude3Opus()
anthropic_claude3_sonnet = Claude3Sonnet()
anthropic_claude3_haiku = Claude3Haiku()
anthropic_claude3_5_sonnet = Claude3_5Sonnet()
anthropic_claude3_5_haiku = Claude3_5Haiku()
anthropic_claude3_7_sonnet = Claude3_7Sonnet()

families = [
    anthropic_claude3_opus,
    anthropic_claude3_sonnet,
    anthropic_claude3_haiku,
    anthropic_claude3_5_sonnet,
    anthropic_claude3_5_haiku,
    anthropic_claude3_7_sonnet,
]

variants: List[Variant] = [
    Variant(
        name="claude-3-5-sonnet-classic",
        family_id=anthropic_claude3_5_sonnet.family_id,
        fields={
            "version": "claude-3-5-sonnet-20241022",
            "prompt": DEFAULT_PROMPT,
        },
        adapter=AdapterRef(id="anthropic"),
        pricing=PricingRef(id="anthropic"),
    ),
]

