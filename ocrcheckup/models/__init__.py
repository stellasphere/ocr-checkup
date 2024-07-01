import time
import numpy as np

from ocrcheckup.models.tesseract import Tesseract
from ocrcheckup.models.doctr import DocTR_RFHosted
from ocrcheckup.models.openai import OpenAI_GPT4o
from ocrcheckup.models.test import TestModel