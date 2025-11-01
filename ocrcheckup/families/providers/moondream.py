from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from ocrcheckup.families.base import ModelFamily
from ocrcheckup.utils.prompts import DEFAULT_OCR_INSTRUCTION


class MoondreamFields(BaseModel):
    model: str = Field(
        default="vikhyatk/moondream2",
        description="Hugging Face repository id for Moondream checkpoint.",
    )
    revision: Optional[str] = Field(
        default=None,
        description="Optional revision/hash to pin when loading the model.",
    )
    prompt: str = Field(
        default=DEFAULT_OCR_INSTRUCTION,
        description="Question passed to the model after encoding the image.",
    )
    device: Optional[str] = Field(
        default=None,
        description="Optional torch device override (cpu, cuda, mps).",
    )


class MoondreamFamily:
    family_id = "moondream"
    display_name = "Moondream"
    description = "Moondream2 open-source vision-language checkpoint."
    family_schema_version = "1"
    fields_schema = MoondreamFields

    def validate_fields(self, fields: dict) -> dict:
        return MoondreamFields.model_validate(fields).model_dump(
            mode="json", by_alias=True, exclude_none=True
        )


family = MoondreamFamily()


__all__ = [
    "MoondreamFields",
    "MoondreamFamily",
    "family",
]
