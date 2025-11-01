from __future__ import annotations

from typing import Any, Dict, Literal, Type

from pydantic import BaseModel, Field

from ocrcheckup.families.base import ModelFamily


class _RoboflowWorkflowFields(BaseModel):
    workspace: Literal["leo-ueno"] = Field(
        default="leo-ueno",
        description="Roboflow workspace slug.",
    )
    workflow: Literal["ocr"] = Field(
        default="ocr",
        description="Workflow identifier to invoke.",
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional parameters forwarded to the workflow request.",
    )
    use_cache: bool = Field(
        default=False,
        description="If true, allow Roboflow caching layer to serve responses.",
    )


class Florence2LargeFields(_RoboflowWorkflowFields):
    model: Literal["florence-2-large-roboflow-hosted"] = Field(
        default="florence-2-large-roboflow-hosted",
        description="Workflow model parameter.",
    )


class Florence2BaseFields(_RoboflowWorkflowFields):
    model: Literal["florence-2-base-roboflow-hosted"] = Field(
        default="florence-2-base-roboflow-hosted",
        description="Workflow model parameter.",
    )


class _RoboflowWorkflowFamily(ModelFamily):
    family_schema_version = "2"
    fields_schema: Type[BaseModel]
    family_id: str
    display_name: str
    description: str | None

    def validate_fields(self, fields: dict) -> dict:
        return self.fields_schema.model_validate(fields).model_dump(
            mode="json", by_alias=True, exclude_none=True
        )


class Florence2LargeFamily(_RoboflowWorkflowFamily):
    family_id = "roboflow-florence-2-large"
    display_name = "Florence 2 Large (Roboflow)"
    description = "Florence 2 Large workflow hosted on Roboflow."
    fields_schema = Florence2LargeFields


class Florence2BaseFamily(_RoboflowWorkflowFamily):
    family_id = "roboflow-florence-2-base"
    display_name = "Florence 2 Base (Roboflow)"
    description = "Florence 2 Base workflow hosted on Roboflow."
    fields_schema = Florence2BaseFields


florence_2_large_family = Florence2LargeFamily()
florence_2_base_family = Florence2BaseFamily()


__all__ = [
    "florence_2_large_family",
    "florence_2_base_family",
    "Florence2LargeFamily",
    "Florence2BaseFamily",
]
