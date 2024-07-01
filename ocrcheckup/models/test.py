from ocrcheckup.benchmark.model import OCRBaseModel, OCRModelResponse, OCRModelInfo

class TestModel(OCRBaseModel):
  def info(self=None):
    return OCRModelInfo(
      name = "Test Model",
      version = "v1",
      tags = [] # cloud, lmm, local
    )

  def __init__(self):
    super().__init__()

  def evaluate(self,image):
  

    return OCRModelResponse(
      prediction="text",
      cost=0.00
    )