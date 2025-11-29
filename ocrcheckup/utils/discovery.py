from __future__ import annotations

import importlib
import pkgutil
import types
from typing import List

import ocrcheckup.families
from ocrcheckup.core.registry import model_families
from ocrcheckup.core.variant import Variant


def discover_families_and_variants() -> List[Variant]:
    """
    Walks through all modules in ocrcheckup.families package.
    Registers any 'family' or 'families' (list) found in model_families registry.
    Collects all 'variants' (list) found.
    """
    all_variants: List[Variant] = []

    # Walk packages in ocrcheckup.families
    package = ocrcheckup.families
    path = package.__path__
    prefix = package.__name__ + "."

    for _, name, _ in pkgutil.walk_packages(path, prefix):
        # Skip base module to avoid re-import or circular issues if any
        if name.endswith(".base"):
            continue

        try:
            module = importlib.import_module(name)
        except ImportError as e:
            print(f"Warning: Failed to import {name}: {e}")
            continue

        # 1. Register families
        # Look for single 'family' object
        single_family = getattr(module, "family", None)
        if single_family and hasattr(single_family, "family_id"):
            try:
                model_families.register(single_family)
            except Exception:
                pass  # Already registered or invalid

        # Look for 'families' list
        families_list = getattr(module, "families", None)
        if isinstance(families_list, list):
            for fam in families_list:
                if hasattr(fam, "family_id"):
                    try:
                        model_families.register(fam)
                    except Exception:
                        pass

        # 2. Collect variants
        module_variants = getattr(module, "variants", None)
        if isinstance(module_variants, list):
            for v in module_variants:
                if isinstance(v, Variant):
                    all_variants.append(v)

    return all_variants

