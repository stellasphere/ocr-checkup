import time
import numpy as np

from .tesseract import Tesseract
from .doctr import DocTR_RoboflowHosted
from .moondream import Moondream2
from .trocr import TrOCR
from .easyocr import EasyOCR
from .idefics import Idefics2
from .gemini import (
    Gemini_1_5_Pro,
    Gemini_1_5_Flash,
    Gemini_1_5_Flash_8B,
    Gemini_2_5_Pro_Preview,
    Gemini_2_0_Flash,
    Gemini_2_0_Flash_Lite,
)
from .openai import (
    GPT_4o,
    O1,
    GPT_4_5_Preview,
    GPT_4o_Mini,
)
from .claude import (
    Claude_3_Opus,
    Claude_3_Sonnet,
    Claude_3_Haiku,
    Claude_3_5_Sonnet,
    Claude_3_5_Sonnet_V2,
    Claude_3_5_Haiku,
    Claude_3_7_Sonnet,
)
from .roboflow import (
    Florence2Large,
    Florence2Base,
)
from .mistral import MistralOCR
from .qwen import Qwen_2_5_VL_7B
__all__ = [
    "DocTR_RoboflowHosted",
    "EasyOCR",
    "GPT_4o",
    "O1",
    "GPT_4_5_Preview",
    "GPT_4o_Mini",
    "Moondream2",
    "TrOCR",
    "Idefics2",
    "Gemini_1_5_Pro",
    "Gemini_1_5_Flash",
    "Gemini_1_5_Flash_8B",
    "Gemini_2_5_Pro_Preview",
    "Gemini_2_0_Flash",
    "Gemini_2_0_Flash_Lite",
    "Claude_3_Opus",
    "Claude_3_Sonnet",
    "Claude_3_Haiku",
    "Claude_3_5_Sonnet",
    "Claude_3_5_Sonnet_V2",
    "Claude_3_5_Haiku",
    "Claude_3_7_Sonnet",
    "MistralOCR",
    "Florence2Large",
    "Florence2Base",
    "Qwen_2_5_VL_7B",
]