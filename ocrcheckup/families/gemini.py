from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from ocrcheckup.core.variant import AdapterRef, PricingRef, Variant
from ocrcheckup.families.base import BaseFamily
from ocrcheckup.prompts import PROMPTS

DEFAULT_PROMPT = PROMPTS["classic"]


class _GeminiFieldsBase(BaseModel):
    model_version: str
    prompt: str | None = None


class Gemini1_5ProFields(_GeminiFieldsBase):
    model_version: str = Field(default="gemini-1.5-pro")


class Gemini1_5FlashFields(_GeminiFieldsBase):
    model_version: str = Field(default="gemini-1.5-flash")


class Gemini1_5Flash8BFields(_GeminiFieldsBase):
    model_version: str = Field(default="gemini-1.5-flash-8b")


class Gemini2_5ProPreviewFields(_GeminiFieldsBase):
    model_version: str = Field(default="gemini-2.5-pro-preview-03-25")


class Gemini2_0FlashFields(_GeminiFieldsBase):
    model_version: str = Field(default="gemini-2.0-flash")


class Gemini2_0FlashLiteFields(_GeminiFieldsBase):
    model_version: str = Field(default="gemini-2.0-flash-lite")


class Gemini1_5Pro(BaseFamily):
    family_id = "gemini-1.5-pro"
    display_name = "Gemini 1.5 Pro"
    description = "Google Gemini 1.5 Pro multimodal model"
    family_schema_version = "1"
    fields_schema = Gemini1_5ProFields


class Gemini1_5Flash(BaseFamily):
    family_id = "gemini-1.5-flash"
    display_name = "Gemini 1.5 Flash"
    description = "Google Gemini 1.5 Flash multimodal model"
    family_schema_version = "1"
    fields_schema = Gemini1_5FlashFields


class Gemini1_5Flash8B(BaseFamily):
    family_id = "gemini-1.5-flash-8b"
    display_name = "Gemini 1.5 Flash 8B"
    description = "Google Gemini 1.5 Flash 8B multimodal model"
    family_schema_version = "1"
    fields_schema = Gemini1_5Flash8BFields


class Gemini2_5ProPreview(BaseFamily):
    family_id = "gemini-2.5-pro-preview"
    display_name = "Gemini 2.5 Pro Preview"
    description = "Google Gemini 2.5 Pro Preview multimodal model"
    family_schema_version = "1"
    fields_schema = Gemini2_5ProPreviewFields


class Gemini2_0Flash(BaseFamily):
    family_id = "gemini-2.0-flash"
    display_name = "Gemini 2.0 Flash"
    description = "Google Gemini 2.0 Flash multimodal model"
    family_schema_version = "1"
    fields_schema = Gemini2_0FlashFields


class Gemini2_0FlashLite(BaseFamily):
    family_id = "gemini-2.0-flash-lite"
    display_name = "Gemini 2.0 Flash Lite"
    description = "Google Gemini 2.0 Flash Lite multimodal model"
    family_schema_version = "1"
    fields_schema = Gemini2_0FlashLiteFields


gemini_1_5_pro = Gemini1_5Pro()
gemini_1_5_flash = Gemini1_5Flash()
gemini_1_5_flash_8b = Gemini1_5Flash8B()
gemini_2_5_pro_preview = Gemini2_5ProPreview()
gemini_2_0_flash = Gemini2_0Flash()
gemini_2_0_flash_lite = Gemini2_0FlashLite()

families = [
    gemini_1_5_pro,
    gemini_1_5_flash,
    gemini_1_5_flash_8b,
    gemini_2_5_pro_preview,
    gemini_2_0_flash,
    gemini_2_0_flash_lite,
]

variants: List[Variant] = [
    Variant(
        name="gemini-1.5-pro-classic",
        family_id=gemini_1_5_pro.family_id,
        fields={
            "model_version": "gemini-1.5-pro",
            "prompt": DEFAULT_PROMPT,
        },
        adapter=AdapterRef(id="gemini"),
        pricing=PricingRef(id="gemini"),
    ),
    Variant(
        name="gemini-1.5-flash-classic",
        family_id=gemini_1_5_flash.family_id,
        fields={
            "model_version": "gemini-1.5-flash",
            "prompt": DEFAULT_PROMPT,
        },
        adapter=AdapterRef(id="gemini"),
        pricing=PricingRef(id="gemini"),
    ),
]

