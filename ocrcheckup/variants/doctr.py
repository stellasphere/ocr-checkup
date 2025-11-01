from __future__ import annotations

from ocrcheckup.core.variant import AdapterRef, PricingRef, Variant


VARIANTS = [
    Variant(
        name="doctr-roboflow-hosted",
        family_id="doctr-roboflow-hosted",
        fields={},
        adapter=AdapterRef(id="doctr-roboflow-hosted", config={}),
        pricing=PricingRef(id="flat-per-image", config={"usd": 0.015, "round": 4}),
    ),
]


__all__ = ["VARIANTS"]
