from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from ocrcheckup.families.base import ModelFamily


class RoboflowWorkflowFields(BaseModel):
    workspace: str = Field(
        ...,
        description="Roboflow workspace slug.",
    )
    workflow: str = Field(
        ...,
        description="Workflow identifier to invoke.",
    )
    model: str = Field(
        ...,
        description="Model parameter passed to the workflow.",
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional parameters forwarded to the workflow request.",
    )
    use_cache: bool = Field(
        default=False,
        description="If true, allow Roboflow caching layer to serve responses.",
    )


class RoboflowWorkflowFamily:
    family_id = "roboflow-workflow"
    display_name = "Roboflow Workflow"
    description = "Custom OCR workflows hosted on Roboflow Inference API."
    family_schema_version = "1"
    fields_schema = RoboflowWorkflowFields

    def validate_fields(self, fields: dict) -> dict:
        return RoboflowWorkflowFields.model_validate(fields).model_dump(
            mode="json", by_alias=True, exclude_none=True
        )


family = RoboflowWorkflowFamily()


__all__ = [
    "RoboflowWorkflowFields",
    "RoboflowWorkflowFamily",
    "family",
]
