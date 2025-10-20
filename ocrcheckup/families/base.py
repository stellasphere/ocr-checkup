from __future__ import annotations

from typing import Any, Protocol, Type

from pydantic import BaseModel


class ModelFamily(Protocol):
    family_id: str
    display_name: str
    description: str | None
    family_schema_version: str
    fields_schema: Type[BaseModel]

    def validate_fields(self, fields: dict) -> dict:
        model = self.fields_schema.model_validate(fields)
        return model.model_dump(mode="json", by_alias=True, exclude_none=True)


__all__ = ["ModelFamily"]
