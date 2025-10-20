from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from ocrcheckup.families.base import ModelFamily


class OpenAIVLMFields(BaseModel):
    model_name: str = Field(default="gpt-4o-mini")
    model_version: Optional[str] = None
    prompt: Optional[str] = None
    temperature: Optional[float] = None


class OpenAIVLMFamily:
    family_id = "openai-vlm"
    display_name = "OpenAI VLM"
    description = None
    family_schema_version = "1"
    fields_schema = OpenAIVLMFields

    def validate_fields(self, fields: dict) -> dict:
        return OpenAIVLMFields.model_validate(fields).model_dump(
            mode="json", by_alias=True, exclude_none=True
        )


family = OpenAIVLMFamily()
