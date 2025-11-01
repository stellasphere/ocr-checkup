from __future__ import annotations

import importlib
import pkgutil
from typing import Iterable, List

from ocrcheckup.core.variant import Variant


def _iter_variant_modules() -> Iterable[str]:
    package = __name__
    for module_info in pkgutil.iter_modules(__path__):  # type: ignore[name-defined]
        if module_info.name.startswith("_"):
            continue
        yield f"{package}.{module_info.name}"


def discover_variants() -> List[Variant]:
    variants: List[Variant] = []
    for module_name in _iter_variant_modules():
        module = importlib.import_module(module_name)
        module_variants = getattr(module, "VARIANTS", None)
        if not module_variants:
            continue
        variants.extend(module_variants)
    return variants


__all__ = ["discover_variants"]
