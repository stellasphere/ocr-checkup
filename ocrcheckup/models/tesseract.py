from ocrcheckup.benchmark.model import OCRBaseModel, OCRModelResponse, OCRModelInfo

import pytesseract
from PIL import Image

class Tesseract(OCRBaseModel):
  def info(self=None):
    return OCRModelInfo(
      name = "Tesseract",
      version = "PyTesseract (Default)",
      tags = ["local"]
    )

  def evaluate(self,image):
    text = pytesseract.image_to_string(Image.fromarray(image))
    return text