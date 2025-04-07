import time
import numpy as np

from .tesseract import Tesseract
from .doctr import DocTR_RoboflowHosted
from .test import TestModel
from .moondream import Moondream2
from .trocr import TrOCR
from .claude import Claude_3_Opus
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
    O3_mini,
    O1_mini,
    GPT_4_5_Preview,
    GPT_4o_Mini,
)
__all__ = [
    "DocTR_RoboflowHosted",
    "EasyOCR",
    "GPT_4o",
    "O1",
    "O3_mini",
    "O1_mini",
    "GPT_4_5_Preview",
    "GPT_4o_Mini",
    "Moondream2",
    "TrOCR",
    "Claude_3_Opus",
    "Idefics2",
    "Gemini_1_5_Pro",
    "Gemini_1_5_Flash",
    "Gemini_1_5_Flash_8B",
    "Gemini_2_5_Pro_Preview",
    "Gemini_2_0_Flash",
    "Gemini_2_0_Flash_Lite",
]