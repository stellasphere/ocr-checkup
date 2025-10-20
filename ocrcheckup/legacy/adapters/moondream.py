from ocrcheckup.benchmark.model import OCRBaseModel, OCRModelResponse, OCRModelInfo

from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch
from .consts import OCR_VLM_PROMPT
class Moondream2(OCRBaseModel):
  def info(self) -> OCRModelInfo:
    return OCRModelInfo(
      name = "Moondream2",
      version = "2025-03-27",
      tags = ["local","lmm"],
      cost_type="compute"
    )

  def __init__(self):
    super().__init__()

    # Let PyTorch decide the device (CPU, CUDA, or MPS)
    if torch.cuda.is_available():
        DEVICE = "cuda:0"
    elif torch.backends.mps.is_available():
        DEVICE = "mps" # Use MPS if available
    else:
        DEVICE = "cpu"

    self.device = DEVICE # Store device
    print(f"Moondream2 initializing on device: {self.device}") # Keep print for confirmation

    model_id = "vikhyatk/moondream2"
    revision = "2025-03-27" # Use the specified revision

    # Load model to the specified device
    self.model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, revision=revision
    ).to(DEVICE)

    self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)


  def evaluate(self, image):
    # Ensure image is in PIL format
    image_pil = Image.fromarray(image)

    # Encode image and ask question
    enc_image = self.model.encode_image(image_pil)
    prediction = self.model.answer_question(
        enc_image,
        OCR_VLM_PROMPT,
        self.tokenizer
    )

    return OCRModelResponse(
      prediction=prediction
    ) 