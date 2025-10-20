from ocrcheckup.benchmark.model import OCRBaseModel, OCRModelResponse, OCRModelInfo
from ocrcheckup.cost import ModelCost, CostType
from inference_sdk import InferenceHTTPClient

class DocTR_RoboflowHosted(OCRBaseModel):
  def info(self=None):
    return OCRModelInfo(
      name = "DocTR",
      version = "roboflow-hosted",
      tags = ["cloud"],
      cost_type = "api"
    )

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

    return OCRModelResponse(
      prediction=text,
      cost_details=ModelCost(
        cost_type=CostType.EXTERNAL,
        info={
          "model_id": "roboflow-hosted",
          "inferences": 1
        }
      )
    )