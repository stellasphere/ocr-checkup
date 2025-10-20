from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

from ocrcheckup.core.types import Sample
from ocrcheckup.core.variant import Variant


@dataclass
class AdapterOutput:
    prediction: str
    metadata: Dict[str, Any]


class Adapter(Protocol):
    id: str
    description: Optional[str]

    def run(self, variant: Variant, sample: Sample) -> AdapterOutput: ...

    def hash_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        return {}


__all__ = ["Adapter", "AdapterOutput"]
