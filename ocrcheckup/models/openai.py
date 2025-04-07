import abc
import os
from ocrcheckup.benchmark.model import OCRBaseModel, OCRModelResponse, OCRModelInfo

from openai import OpenAI
from io import BytesIO
from PIL import Image
import base64
from ..rate_limiter import RateLimiter

# Abstract Base Class for OpenAI models
class _OpenAIBase(OCRBaseModel, abc.ABC):
    """
    Abstract base class for OpenAI GPT models.
    Handles common initialization (API key, client, rate limiter) and evaluation logic.
    """
    # Accept rpm as argument
    def __init__(self, model_id: str, rpm: int, cost_per_second: float = None):
        # Create RateLimiter instance using passed rpm
        limiter = RateLimiter(rpm)
        # Pass limiter to superclass
        super().__init__(cost_per_second=cost_per_second, rate_limiter=limiter)
        self.model_id = model_id

        # Configure OpenAI client with API Key from environment variable
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        try:
            # Initialize the client
            self.client = OpenAI(api_key=api_key)
        except Exception as e:
            print(f"OpenAI Client initialization failed: {e}")
            raise

        # Common configurations (can be overridden if needed, though less common for OpenAI API structure)
        self.prompt = "Read the text in the image. Return only the text as it is visible in the image."


    @abc.abstractmethod
    def info(self) -> OCRModelInfo:
        """Return model information."""
        pass

    def evaluate(self, image) -> OCRModelResponse:
        """
        Evaluates the model on a single image using the OpenAI API.
        """
        buffered = BytesIO()
        # Convert numpy array to PIL Image if needed (assuming input is numpy array)
        if not isinstance(image, Image.Image):
             pil_image = Image.fromarray(image)
        else:
             pil_image = image
        pil_image.save(buffered, format="JPEG") # Consider PNG if JPEG compression is an issue
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        full_text = ""
        cost = 0.0
        prediction = ""

        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {
                    "role": "user",
                    "content": [
                        {
                        "type": "text",
                        "text": self.prompt
                        },
                        {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_str}",
                            "detail": "high" # Request high detail for better OCR
                        }
                        }
                    ]
                    }
                ]
            )

            if response.choices and response.choices[0].message and response.choices[0].message.content:
                 full_text = response.choices[0].message.content
            else:
                 # Handle cases where response might be empty or unexpected format
                 print(f"OpenAI ({self.model_id}): Received no text content or unexpected response format.")
                 print(f"Response: {response}")


            # Cost calculation - Using GPT-4o pricing as a default.
            # TODO: Verify and potentially customize cost calculation per specific OpenAI model.
            if response.usage:
                input_price = (response.usage.prompt_tokens / 1000000) * 5  # $5 per million input tokens
                output_price = (response.usage.completion_tokens / 1000000) * 15 # $15 per million output tokens
                cost = input_price + output_price
            else:
                print(f"OpenAI ({self.model_id}): Usage information not found in response.")


            prediction = full_text.strip()

        except Exception as e:
            print(f"OpenAI ({self.model_id}): Error during API call: {e}")
            # prediction remains ""
            # cost remains 0.0

        return OCRModelResponse(
            prediction=prediction,
            cost=cost
        )


# --- Specific OpenAI Model Implementations ---

class GPT_4o(_OpenAIBase):
  def __init__(self, cost_per_second: float = None):
    # Use specific model ID for GPT-4o
    super().__init__(model_id="gpt-4o-2024-05-13", rpm=500, cost_per_second=cost_per_second) # Updated RPM to 500

  def info(self) -> OCRModelInfo:
    return OCRModelInfo(
      name = "GPT-4o",
      version = self.model_id,
      tags = ["cloud", "lmm"]
    )

class O1(_OpenAIBase):
  def __init__(self, cost_per_second: float = None):
    super().__init__(model_id="o1-2024-12-17", rpm=500, cost_per_second=cost_per_second)

  def info(self) -> OCRModelInfo:
    return OCRModelInfo(
      name = "o1",
      version = self.model_id,
      tags = ["cloud", "lmm"]
    )

# -- API does not support images for o3 and o1 mini at this time
# class O3_mini(_OpenAIBase):
#   def __init__(self, cost_per_second: float = None):
#     super().__init__(model_id="o3-mini-2025-01-31", rpm=500, cost_per_second=cost_per_second)

#   def info(self) -> OCRModelInfo:
#     return OCRModelInfo(
#       name = "o3-mini",
#       version = self.model_id,
#       tags = ["cloud", "lmm"]
#     )

# class O1_mini(_OpenAIBase):
#   def __init__(self, cost_per_second: float = None):
#     super().__init__(model_id="o1-mini-2024-09-12", rpm=500, cost_per_second=cost_per_second)

#   def info(self) -> OCRModelInfo:
#     return OCRModelInfo(
#       name = "o1-mini",
#       version = self.model_id,
#       tags = ["cloud", "lmm"]
#     )

class GPT_4_5_Preview(_OpenAIBase):
  def __init__(self, cost_per_second: float = None):
    super().__init__(model_id="gpt-4.5-preview-2025-02-27", rpm=500, cost_per_second=cost_per_second)

  def info(self) -> OCRModelInfo:
    return OCRModelInfo(
      name = "GPT-4.5 Preview",
      version = self.model_id,
      tags = ["cloud", "lmm"]
    )

class GPT_4o_Mini(_OpenAIBase):
  def __init__(self, cost_per_second: float = None):
    super().__init__(model_id="gpt-4o-mini-2024-07-18", rpm=500, cost_per_second=cost_per_second)

  def info(self) -> OCRModelInfo:
    return OCRModelInfo(
      name = "GPT-4o mini",
      version = self.model_id,
      tags = ["cloud", "lmm"]
    )