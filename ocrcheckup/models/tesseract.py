from ocrcheckup.models import OCRBaseModel

import pytesseract
from PIL import Image

class Tesseract(OCRBaseModel):
  name = "Tesseract"
  version = "PyTesseract (Default)"

  is_cloud = False
  is_lmm = False

  def evaluate(self,image):
    text = pytesseract.image_to_string(Image.fromarray(image))
    return text