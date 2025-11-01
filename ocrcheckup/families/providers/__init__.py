"""Model families for officially supported providers."""

from .anthropic import (
    claude_35_haiku_family,
    claude_35_sonnet_family,
    claude_35_sonnet_v2_family,
    claude_37_sonnet_family,
    claude_3_haiku_family,
    claude_3_opus_family,
    claude_3_sonnet_family,
)
from .doctr import family as doctr_family
from .easyocr import family as easyocr_family
from .gemini import (
    gemini_15_flash_8b_family,
    gemini_15_flash_family,
    gemini_15_pro_family,
    gemini_20_flash_family,
    gemini_20_flash_lite_family,
    gemini_25_pro_preview_family,
)
from .idefics2 import family as idefics2_family
from .mistral import family as mistral_family
from .moondream import family as moondream_family
from .openai import (
    gpt_45_preview_family,
    gpt_4o_family,
    gpt_4o_mini_family,
    o1_family,
)
from .qwen import family as qwen_family
from .roboflow import (
    florence_2_base_family,
    florence_2_large_family,
)
from .trocr import family as trocr_family


__all__ = [
    "claude_3_opus_family",
    "claude_3_sonnet_family",
    "claude_3_haiku_family",
    "claude_35_sonnet_family",
    "claude_35_sonnet_v2_family",
    "claude_35_haiku_family",
    "claude_37_sonnet_family",
    "doctr_family",
    "easyocr_family",
    "gemini_15_pro_family",
    "gemini_15_flash_family",
    "gemini_15_flash_8b_family",
    "gemini_25_pro_preview_family",
    "gemini_20_flash_family",
    "gemini_20_flash_lite_family",
    "idefics2_family",
    "mistral_family",
    "moondream_family",
    "gpt_4o_family",
    "gpt_4o_mini_family",
    "gpt_45_preview_family",
    "o1_family",
    "qwen_family",
    "florence_2_large_family",
    "florence_2_base_family",
    "trocr_family",
]
