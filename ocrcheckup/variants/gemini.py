from __future__ import annotations

from ocrcheckup.core.variant import AdapterRef, PricingRef, Variant
from ocrcheckup.utils.prompts import DEFAULT_OCR_INSTRUCTION


TOKEN_PRICING = {
    "input_per_million": 7.0,
    "output_per_million": 21.0,
    "per_request": 0.0,
    "round": 6,
}


def _variant(name: str, family_id: str, temperature: float | None = None) -> Variant:
    fields = {
        "prompt": DEFAULT_OCR_INSTRUCTION,
        "max_output_tokens": 896,
    }
    if temperature is not None:
        fields["temperature"] = temperature
    return Variant(
        name=name,
        family_id=family_id,
        fields=fields,
        adapter=AdapterRef(id="google-gemini-vision", config={}),
        pricing=PricingRef(id="token-usage", config=TOKEN_PRICING.copy()),
    )


VARIANTS = [
    _variant("google-gemini-1.5-pro", "google-gemini-1.5-pro", temperature=0.4),
    _variant("google-gemini-1.5-flash", "google-gemini-1.5-flash", temperature=0.2),
    _variant("google-gemini-1.5-flash-8b", "google-gemini-1.5-flash-8b", temperature=0.2),
    _variant("google-gemini-2.5-pro-preview", "google-gemini-2.5-pro-preview", temperature=0.4),
    _variant("google-gemini-2.0-flash", "google-gemini-2.0-flash", temperature=0.2),
    _variant("google-gemini-2.0-flash-lite", "google-gemini-2.0-flash-lite", temperature=0.2),
]


__all__ = ["VARIANTS"]
