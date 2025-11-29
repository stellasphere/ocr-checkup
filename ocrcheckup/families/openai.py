from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field

from ocrcheckup.core.variant import AdapterRef, PricingRef, Variant
from ocrcheckup.families.base import BaseFamily
from ocrcheckup.prompts import PROMPTS

DEFAULT_PROMPT = PROMPTS["classic"]


class _OpenAIFieldsBase(BaseModel):
    snapshot: str
    prompt: str | None = None
    image_detail: Literal["low", "high"] | None = Field(default="high")


class OpenAIGPT4oFields(_OpenAIFieldsBase):
    snapshot: str = Field(default="gpt-4o-2024-05-13")


class OpenAIGPT4oMiniFields(_OpenAIFieldsBase):
    snapshot: str = Field(default="gpt-4o-mini-2024-07-18")


class OpenAIGPT4_5Fields(_OpenAIFieldsBase):
    snapshot: str = Field(default="gpt-4.5-preview-2025-02-27")


class OpenAIGPT4o(BaseFamily):
    family_id = "openai-gpt-4o"
    display_name = "OpenAI GPT-4o"
    description = "OpenAI GPT-4o multimodal model"
    family_schema_version = "1"
    fields_schema = OpenAIGPT4oFields


class OpenAIGPT4oMini(BaseFamily):
    family_id = "openai-gpt-4o-mini"
    display_name = "OpenAI GPT-4o mini"
    description = "OpenAI GPT-4o mini multimodal model"
    family_schema_version = "1"
    fields_schema = OpenAIGPT4oMiniFields


class OpenAIGPT4_5(BaseFamily):
    family_id = "openai-gpt-4.5"
    display_name = "OpenAI GPT-4.5"
    description = "OpenAI GPT-4.5 snapshot model"
    family_schema_version = "1"
    fields_schema = OpenAIGPT4_5Fields


openai_gpt4o = OpenAIGPT4o()
openai_gpt4o_mini = OpenAIGPT4oMini()
openai_gpt4_5 = OpenAIGPT4_5()

families = [openai_gpt4o, openai_gpt4o_mini, openai_gpt4_5]

variants: List[Variant] = [
    Variant(
        name="openai-gpt4o-classic",
        family_id=openai_gpt4o.family_id,
        fields={
            "snapshot": "gpt-4o-2024-05-13",
            "prompt": DEFAULT_PROMPT,
            "image_detail": "high",
        },
        adapter=AdapterRef(id="openai"),
        pricing=PricingRef(id="openai"),
    ),
    Variant(
        name="openai-gpt4o-mini-classic",
        family_id=openai_gpt4o_mini.family_id,
        fields={
            "snapshot": "gpt-4o-mini-2024-07-18",
            "prompt": DEFAULT_PROMPT,
            "image_detail": "high",
        },
        adapter=AdapterRef(id="openai"),
        pricing=PricingRef(id="openai"),
    ),
    Variant(
        name="openai-gpt4_5-classic",
        family_id=openai_gpt4_5.family_id,
        fields={
            "snapshot": "gpt-4.5-preview-2025-02-27",
            "prompt": DEFAULT_PROMPT,
            "image_detail": "high",
        },
        adapter=AdapterRef(id="openai"),
        pricing=PricingRef(id="openai"),
    ),
]

