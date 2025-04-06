import time
import numpy as np

from ocrcheckup.models.tesseract import Tesseract
from ocrcheckup.models.doctr import DocTR_RoboflowHosted
from ocrcheckup.models.openai import OpenAI_GPT4o
from ocrcheckup.models.test import TestModel
from ocrcheckup.models.moondream import Moondream2
from ocrcheckup.models.trocr import TrOCR_Base_Printed