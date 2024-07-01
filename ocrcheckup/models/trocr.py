from ocrcheckup.benchmark.model import OCRBaseModel, OCRModelResponse, OCRModelInfo

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

class TrOCR(OCRBaseModel):
  def info(self=None):
    raise NotImplementedError("This is a base TrOCR class, please use a specific version.")

  def __init__(self,api_key,version):
    super().__init__()

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    self.model_id = version

    self.model = VisionEncoderDecoderModel.from_pretrained(
        self.model_id,
        device_map=DEVICE
    ).eval()

    self.processor = TrOCRProcessor.from_pretrained(
        self.model_id,
        device_map=DEVICE
    )

  def evaluate(self,image):
    raise NotImplementedError("This is a base TrOCR class, please use a specific version.")
  

class TrOCR_
