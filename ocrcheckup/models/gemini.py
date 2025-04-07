from ocrcheckup.benchmark.model import OCRBaseModel, OCRModelResponse, OCRModelInfo

import base64
from io import BytesIO
from PIL import Image
# Use genai Client and types
from google import genai
from google.genai import types
# Remove unused vertexai import
# from vertexai.preview.generative_models import Part
import os
import abc # Import abc for abstract base class
# Import RateLimiter
from ..rate_limiter import RateLimiter

# Base class for Gemini models
class _GeminiBase(OCRBaseModel, abc.ABC):
    """
    Abstract base class for Gemini models via Google AI SDK (using API Key and genai.Client).
    Handles common initialization and evaluation logic.
    """
    # Accept rpm as argument
    def __init__(self, model_id: str, rpm: int, cost_per_second: float = None):
        # Create RateLimiter instance using passed rpm
        limiter = RateLimiter(rpm)
        # Pass limiter to superclass
        super().__init__(cost_per_second=cost_per_second, rate_limiter=limiter)
        self.model_id = model_id

        # Configure genai client with API Key from environment variable
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        try:
            # Initialize the client
            self.client = genai.Client(api_key=api_key)
            # print(f"Google AI Client initialized for model {self.model_id}.")
        except Exception as e:
            print(f"Google AI Client initialization failed: {e}")
            raise # Re-raise exception if initialization fails

        # Remove genai.configure and genai.GenerativeModel initialization
        # self.model = genai.GenerativeModel(self.model_id) # Now using client

        # Common configurations (can be overridden by subclasses if needed)
        self.prompt = "Read the text in the image. Return only the text as it is visible in the image."
        # Adapt generation_config to types.GenerateContentConfig format
        self.generation_config = types.GenerateContentConfig(
            max_output_tokens=2048,
            temperature=0.4,
            top_p=1.0, # genai expects float
            top_k=32,
            safety_settings = [
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH),
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH),
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH),
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH),
            ]
        )

    @abc.abstractmethod
    def info(self) -> OCRModelInfo:
        """Return model information."""
        pass

    def evaluate(self, image) -> OCRModelResponse:
        """
        Evaluates the model on a single image using genai.Client.
        """
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(image)

        # Prepare contents list [prompt, image]
        contents = [self.prompt, pil_image]

        full_text = ""
        cost = 0.0 # Keep existing cost logic (may be inaccurate for genai API)

        try:
            # Use client.models.generate_content
            response = self.client.models.generate_content(
                model=self.model_id, # Pass model ID here
                contents=contents,
                config=self.generation_config,
            )

            # Extract text directly from response
            # Add safety check for potential blocks
            if response.candidates and response.candidates[0].content.parts:
                 full_text = response.text
            elif response.prompt_feedback.block_reason:
                 print(f"Gemini ({self.model_id}): Request blocked. Reason: {response.prompt_feedback.block_reason}")
                 print(f"Safety Ratings: {response.prompt_feedback.safety_ratings}")
            else:
                 # Handle cases where response might be empty or unexpected format
                 print(f"Gemini ({self.model_id}): Received no text content or unexpected response format.")
                 print(f"Response: {response}")


            # Cost calculation - reusing the previous logic.
            # TODO: Verify if this cost model applies accurately to all Gemini variants.
            # It might be better to fetch cost from usage metadata if available via the SDK.
            input_cost = (0.0025) + ((len(self.prompt) / 1000) * 0.000125)
            output_cost = (len(full_text) / 1000) * 0.000375
            cost = input_cost + output_cost

            prediction = full_text.strip()

        except Exception as e:
            print(f"Gemini ({self.model_id}): Error during API call: {e}")
            prediction = ""
            cost = 0.0 # Keep existing cost logic

        return OCRModelResponse(
            prediction=prediction,
            cost=cost
        )

# --- Specific Gemini Model Implementations ---

class Gemini_1_5_Pro(_GeminiBase):
    def __init__(self, cost_per_second: float = None):
        super().__init__(model_id="gemini-1.5-pro", rpm=1000, cost_per_second=cost_per_second)

    # Restore original info method
    def info(self) -> OCRModelInfo:
        return OCRModelInfo(
            name="Gemini 1.5 Pro",
            version=self.model_id, # Use the actual model_id
            tags=["cloud", "lmm"],
        )

class Gemini_1_5_Flash(_GeminiBase):
    def __init__(self, cost_per_second: float = None):
        # Pass specific RPM (2000) to base
        super().__init__(model_id="gemini-1.5-flash", rpm=2000, cost_per_second=cost_per_second)

    # Restore original info method
    def info(self) -> OCRModelInfo:
        return OCRModelInfo(
            name="Gemini 1.5 Flash",
            version=self.model_id,
            tags=["cloud", "lmm"],
        )

class Gemini_1_5_Flash_8B(_GeminiBase):
    def __init__(self, cost_per_second: float = None):
        # Pass specific RPM (4000) to base
        super().__init__(model_id="gemini-1.5-flash-8b", rpm=4000, cost_per_second=cost_per_second)

    # Restore original info method
    def info(self) -> OCRModelInfo:
        return OCRModelInfo(
            name="Gemini 1.5 Flash-8B",
            version=self.model_id,
            tags=["cloud", "lmm"],
        )

class Gemini_2_5_Pro_Preview(_GeminiBase): # Renamed based on user list
    def __init__(self, cost_per_second: float = None):
        # Pass specific RPM (150) to base
        super().__init__(model_id="gemini-2.5-pro-preview-03-25", rpm=150, cost_per_second=cost_per_second)

    # Restore original info method
    def info(self) -> OCRModelInfo:
        return OCRModelInfo(
            name="Gemini 2.5 Pro Preview",
            version=self.model_id,
            tags=["cloud", "lmm"],
        )

class Gemini_2_0_Flash(_GeminiBase): # Renamed based on user list
    def __init__(self, cost_per_second: float = None):
        # Pass specific RPM (2000) to base
        super().__init__(model_id="gemini-2.0-flash", rpm=2000, cost_per_second=cost_per_second)

    # Restore original info method
    def info(self) -> OCRModelInfo:
        return OCRModelInfo(
            name="Gemini 2.0 Flash",
            version=self.model_id,
            tags=["cloud", "lmm"],
        )

class Gemini_2_0_Flash_Lite(_GeminiBase): # Renamed based on user list
    def __init__(self, cost_per_second: float = None):
        # Pass specific RPM (4000) to base
        super().__init__(model_id="gemini-2.0-flash-lite", rpm=4000, cost_per_second=cost_per_second)

    # Restore original info method
    def info(self) -> OCRModelInfo:
        return OCRModelInfo(
            name="Gemini 2.0 Flash-Lite",
            version=self.model_id,
            tags=["cloud", "lmm"],
        )