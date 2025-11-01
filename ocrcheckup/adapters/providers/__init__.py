"""Adapters targeting specific hosted or local providers."""

from .anthropic import adapter as anthropic_adapter
from .doctr import adapter as doctr_adapter
from .easyocr import adapter as easyocr_adapter
from .gemini import adapter as gemini_adapter
from .idefics2 import adapter as idefics2_adapter
from .mistral import adapter as mistral_adapter
from .moondream import adapter as moondream_adapter
from .openai import adapter as openai_adapter
from .qwen import adapter as qwen_adapter
from .roboflow import adapter as roboflow_adapter
from .trocr import adapter as trocr_adapter


__all__ = [
    "anthropic_adapter",
    "doctr_adapter",
    "easyocr_adapter",
    "gemini_adapter",
    "idefics2_adapter",
    "mistral_adapter",
    "moondream_adapter",
    "openai_adapter",
    "qwen_adapter",
    "roboflow_adapter",
    "trocr_adapter",
]
