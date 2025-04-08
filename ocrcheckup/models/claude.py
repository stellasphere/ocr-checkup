import abc
import os
from ocrcheckup.benchmark.model import OCRBaseModel, OCRModelResponse, OCRModelInfo
from ocrcheckup.cost import ModelCost, CostType
import anthropic
from io import BytesIO
from PIL import Image
import base64
from ..rate_limiter import RateLimiter
from .consts import OCR_VLM_PROMPT
# Abstract Base Class for Anthropic Claude models
class _ClaudeBase(OCRBaseModel, abc.ABC):
    """
    Abstract base class for Anthropic Claude models.
    Handles common initialization (API key, client, rate limiter) and evaluation logic.
    """
    def __init__(self, model_id: str, rpm: int, cost_per_second: float = None):
        # Create RateLimiter instance using passed rpm
        limiter = RateLimiter(rpm)
        # Pass limiter to superclass
        super().__init__(cost_per_second=cost_per_second, rate_limiter=limiter)
        self.model_id = model_id # Store model version for API call

        # Configure Anthropic client with API Key from environment variable
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set.")
        try:
            # Initialize the client
            self.client = anthropic.Anthropic(api_key=api_key)
        except Exception as e:
            print(f"Anthropic Client initialization failed: {e}")
            raise

    @abc.abstractmethod
    def info(self) -> OCRModelInfo:
        """Return model information."""
        pass

    def evaluate(self, image) -> OCRModelResponse:
        """
        Evaluates the model on a single image using the Anthropic API.
        """
        # Convert numpy array image to base64 encoded JPEG
        buffered = BytesIO()
        # Convert numpy array to PIL Image if needed
        if not isinstance(image, Image.Image):
             pil_image = Image.fromarray(image)
        else:
             pil_image = image
        pil_image.save(buffered, format="JPEG") # Use JPEG as specified in original code
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        prediction = ""
        cost_data = None
        error_message = None

        try:
            message = self.client.messages.create(
                model=self.model_id,
                max_tokens=1000,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": img_str
                                }
                            },
                            {
                                "type": "text",
                                "text": OCR_VLM_PROMPT
                            }
                        ]
                    }
                ]
            )

            # Extract text prediction
            if message.content and isinstance(message.content, list) and len(message.content) > 0:
                prediction = message.content[0].text.strip()
            else:
                print(f"Anthropic ({self.model_id}): Received no text content or unexpected response format.")
                print(f"Response: {message}")

            # Cost calculation 
            cost_data = ModelCost(
                cost_type=CostType.EXTERNAL,
                info={
                    "model_id": self.model_id,
                    "input_tokens": message.usage.input_tokens,
                    "output_tokens": message.usage.output_tokens
                }
            )
            
        except Exception as e:
            # Handle potential API errors gracefully
            print(f"Error during Anthropic API call ({self.model_id}): {e}")
            error_message = str(e)
            # prediction remains ""
            # cost remains 0.0

        return OCRModelResponse(
            prediction=prediction,
            cost_details=cost_data,
            error_message=error_message
        )

# --- Specific Claude Model Implementations ---

class Claude_3_Opus(_ClaudeBase):
    def __init__(self, cost_per_second: float = None):
        # Specific model ID and RPM for Opus
        super().__init__(model_id="claude-3-opus-20240229", rpm=50, cost_per_second=cost_per_second)

    def info(self):
        return OCRModelInfo(
            name="Claude 3 Opus",
            version=self.model_id,
            tags=["cloud", "lmm"]
        )

class Claude_3_Sonnet(_ClaudeBase):
    def __init__(self, cost_per_second: float = None):
        super().__init__(model_id="claude-3-sonnet-20240229", rpm=50, cost_per_second=cost_per_second)

    def info(self):
        return OCRModelInfo(
            name="Claude 3 Sonnet",
            version=self.model_id,
            tags=["cloud", "lmm"]
        )

class Claude_3_Haiku(_ClaudeBase):
    def __init__(self, cost_per_second: float = None):
        super().__init__(model_id="claude-3-haiku-20240307", rpm=50, cost_per_second=cost_per_second)

    def info(self):
        return OCRModelInfo(
            name="Claude 3 Haiku",
            version=self.model_id,
            tags=["cloud", "lmm"]
        )

class Claude_3_5_Sonnet(_ClaudeBase):
    def __init__(self, cost_per_second: float = None):
        # Refers to the original 3.5 Sonnet release
        super().__init__(model_id="claude-3-5-sonnet-20240620", rpm=50, cost_per_second=cost_per_second)

    def info(self):
        return OCRModelInfo(
            name="Claude 3.5 Sonnet",
            version=self.model_id,
            tags=["cloud", "lmm"]
        )

class Claude_3_5_Sonnet_V2(_ClaudeBase):
    def __init__(self, cost_per_second: float = None):
        # This corresponds to claude-3-5-sonnet-20241022 (latest)
        super().__init__(model_id="claude-3-5-sonnet-20241022", rpm=50, cost_per_second=cost_per_second)

    def info(self):
        return OCRModelInfo(
            name="Claude 3.5 Sonnet v2", # Distinguish name
            version=self.model_id,
            tags=["cloud", "lmm"]
        )

class Claude_3_5_Haiku(_ClaudeBase):
    def __init__(self, cost_per_second: float = None):
        # This corresponds to claude-3-5-haiku-20241022 (latest)
        super().__init__(model_id="claude-3-5-haiku-20241022", rpm=50, cost_per_second=cost_per_second)

    def info(self):
        return OCRModelInfo(
            name="Claude 3.5 Haiku",
            version=self.model_id,
            tags=["cloud", "lmm"]
        )

class Claude_3_7_Sonnet(_ClaudeBase):
    def __init__(self, cost_per_second: float = None):
        # This corresponds to claude-3-7-sonnet-20250219 (latest)
        super().__init__(model_id="claude-3-7-sonnet-20250219", rpm=50, cost_per_second=cost_per_second)

    def info(self):
        return OCRModelInfo(
            name="Claude 3.7 Sonnet",
            version=self.model_id,
            tags=["cloud", "lmm"]
        ) 