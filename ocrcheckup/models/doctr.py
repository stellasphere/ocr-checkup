from ocrcheckup.models import OCRBaseModel

from inference_sdk import InferenceHTTPClient

class DocTR_RFHosted(OCRBaseModel):
  name = "DocTR"
  version = "Roboflow Hosted (Default)"

  is_cloud = True
  is_lmm = False

  def __init__(self,api_key):
    self.api_key = api_key
    super().__init__()

  def evaluate(self,image):
    CLIENT = InferenceHTTPClient(
      api_url="https://infer.roboflow.com",
      api_key=self.api_key
    )

    result = CLIENT.ocr_image(inference_input=image)
    text = result["result"]

    return {
      "result": text,
      "cost": 0.003
    }