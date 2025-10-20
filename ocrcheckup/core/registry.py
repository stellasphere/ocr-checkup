from __future__ import annotations

from typing import Any, Dict, List, Optional


class ModelFamilyRegistry:
    def __init__(self) -> None:
        self._items: Dict[str, Any] = {}

    def register(self, family: Any) -> None:
        fid = getattr(family, "family_id", None)
        if not fid:
            raise ValueError("ModelFamily must have family_id")
        self._items[fid] = family

    def get(self, family_id: str) -> Any:
        if family_id not in self._items:
            raise KeyError(f"Unknown family_id: {family_id}")
        return self._items[family_id]

    def all(self) -> List[Any]:
        return list(self._items.values())


class AdapterRegistry:
    def __init__(self) -> None:
        self._items: Dict[str, Any] = {}

    def register(self, adapter: Any) -> None:
        aid = getattr(adapter, "id", None)
        if not aid:
            raise ValueError("Adapter must have id")
        self._items[aid] = adapter

    def get(self, adapter_id: str) -> Any:
        if adapter_id not in self._items:
            raise KeyError(f"Unknown adapter id: {adapter_id}")
        return self._items[adapter_id]

    def all(self) -> List[Any]:
        return list(self._items.values())


class PricingRegistry:
    def __init__(self) -> None:
        self._items: Dict[str, Any] = {}

    def register(self, pricing: Any) -> None:
        pid = getattr(pricing, "id", None)
        if not pid:
            raise ValueError("PricingModel must have id")
        self._items[pid] = pricing

    def get(self, pricing_id: str) -> Any:
        if pricing_id not in self._items:
            raise KeyError(f"Unknown pricing id: {pricing_id}")
        return self._items[pricing_id]

    def all(self) -> List[Any]:
        return list(self._items.values())


model_families = ModelFamilyRegistry()
adapters = AdapterRegistry()
pricing_models = PricingRegistry()


__all__ = [
    "ModelFamilyRegistry",
    "AdapterRegistry",
    "PricingRegistry",
    "model_families",
    "adapters",
    "pricing_models",
]
