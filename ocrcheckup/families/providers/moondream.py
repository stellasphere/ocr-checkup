from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from ocrcheckup.families.base import ModelFamily
from ocrcheckup.utils.prompts import DEFAULT_OCR_INSTRUCTION


class MoondreamFields(BaseModel):
    model: Literal["vikhyatk/moondream2"] = Field(
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


class MoondreamFamily(ModelFamily):
    family_id = "moondream2"
    display_name = "Moondream2"
    description = "Moondream2 open-source vision-language checkpoint."
    family_schema_version = "2"
    fields_schema = MoondreamFields

    def validate_fields(self, fields: dict) -> dict:
        return MoondreamFields.model_validate(fields).model_dump(
            mode="json", by_alias=True, exclude_none=True
        )


family = MoondreamFamily()


__all__ = ["MoondreamFields", "MoondreamFamily", "family"]
