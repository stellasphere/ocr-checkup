from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class AdapterRef(BaseModel):
    id: str
    config: Optional[Dict[str, Any]] = None


class PricingRef(BaseModel):
    id: str
    config: Optional[Dict[str, Any]] = None


class Variant(BaseModel):
    name: str
    family_id: str
    fields: Dict[str, Any]
    adapter: AdapterRef
    pricing: PricingRef
    description: Optional[str] = None
    variant_id: Optional[str] = Field(default=None)

    def compute_variant_id(
        self,
        *,
        family_schema_version: str,
        canonical_fields: Dict[str, Any],
        adapter_hash_cfg: Dict[str, Any],
        pricing_hash_cfg: Dict[str, Any],
        truncate: int = 16,
    ) -> str:
        payload = {
            "family_id": self.family_id,
            "family_schema_version": family_schema_version,
            "fields": canonical_fields,
            "adapter": {"id": self.adapter.id, "config": adapter_hash_cfg},
            "pricing": {"id": self.pricing.id, "config": pricing_hash_cfg},
        }
        digest = hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")
        ).hexdigest()
        vid = digest[:truncate]
        self.variant_id = vid
        return vid


__all__ = [
    "Variant",
    "AdapterRef",
    "PricingRef",
]
