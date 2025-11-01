from __future__ import annotations

from ocrcheckup.core.variant import AdapterRef, PricingRef, Variant


ELAPSED_PRICING = {
    "usd_per_second": 0.005,
    "round": 6,
}


VARIANTS = [
    Variant(
        name="roboflow-florence-2-large",
        family_id="roboflow-florence-2-large",
        fields={
            "parameters": {},
            "use_cache": False,
        },
        adapter=AdapterRef(
            id="roboflow-workflow",
            config={"rpm": 120, "api_url": "https://serverless.roboflow.com"},
        ),
        pricing=PricingRef(id="elapsed-seconds", config=ELAPSED_PRICING.copy()),
    ),
    Variant(
        name="roboflow-florence-2-base",
        family_id="roboflow-florence-2-base",
        fields={
            "parameters": {},
            "use_cache": False,
        },
        adapter=AdapterRef(
            id="roboflow-workflow",
            config={"rpm": 120, "api_url": "https://serverless.roboflow.com"},
        ),
        pricing=PricingRef(id="elapsed-seconds", config=ELAPSED_PRICING.copy()),
    ),
]


__all__ = ["VARIANTS"]
