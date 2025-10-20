from __future__ import annotations

from typing import Any, Dict, Optional, Protocol

from ocrcheckup.core.variant import Variant


class PricingModel(Protocol):
    id: str
    description: Optional[str]

    def quote(self, prediction_record: dict, variant: Variant, config: Optional[Dict[str, Any]]) -> float | None: ...

    def hash_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        return {}


__all__ = ["PricingModel"]
