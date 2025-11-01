from __future__ import annotations

from typing import Iterable

from ocrcheckup.adapters.providers import (
    anthropic_adapter,
    doctr_adapter,
    easyocr_adapter,
    gemini_adapter,
    idefics2_adapter,
    mistral_adapter,
    moondream_adapter,
    openai_adapter,
    qwen_adapter,
    roboflow_adapter,
    trocr_adapter,
)
from ocrcheckup.adapters.tesseract_pytesseract import adapter as tesseract_adapter
from ocrcheckup.core.registry import adapters, model_families, pricing_models
from ocrcheckup.families.examples.tesseract import family as tesseract_family
from ocrcheckup.families.providers import (
    anthropic_family,
    doctr_family,
    easyocr_family,
    gemini_family,
    idefics2_family,
    mistral_family,
    moondream_family,
    openai_family,
    qwen_family,
    roboflow_family,
    trocr_family,
)
from ocrcheckup.pricing.builtins.elapsed_seconds import pricing as elapsed_seconds_pricing
from ocrcheckup.pricing.builtins.flat_per_image import pricing as flat_per_image_pricing
from ocrcheckup.pricing.builtins.local import pricing as local_pricing
from ocrcheckup.pricing.builtins.per_page import pricing as per_page_pricing
from ocrcheckup.pricing.builtins.token_usage import pricing as token_usage_pricing


DEFAULT_FAMILIES = [
    openai_family,
    anthropic_family,
    gemini_family,
    mistral_family,
    qwen_family,
    moondream_family,
    idefics2_family,
    trocr_family,
    easyocr_family,
    doctr_family,
    roboflow_family,
    tesseract_family,
]

DEFAULT_ADAPTERS = [
    openai_adapter,
    anthropic_adapter,
    gemini_adapter,
    mistral_adapter,
    qwen_adapter,
    moondream_adapter,
    idefics2_adapter,
    trocr_adapter,
    easyocr_adapter,
    doctr_adapter,
    roboflow_adapter,
    tesseract_adapter,
]

DEFAULT_PRICING = [
    token_usage_pricing,
    per_page_pricing,
    elapsed_seconds_pricing,
    flat_per_image_pricing,
    local_pricing,
]


def _register_many(registry, items: Iterable[object]) -> None:
    for item in items:
        registry.register(item)


def register_default_components() -> None:
    """Register built-in families, adapters, and pricing models."""

    _register_many(model_families, DEFAULT_FAMILIES)
    _register_many(adapters, DEFAULT_ADAPTERS)
    _register_many(pricing_models, DEFAULT_PRICING)


__all__ = [
    "DEFAULT_FAMILIES",
    "DEFAULT_ADAPTERS",
    "DEFAULT_PRICING",
    "register_default_components",
]
