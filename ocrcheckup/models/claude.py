from ocrcheckup.benchmark.model import OCRBaseModel, OCRModelResponse, OCRModelInfo

import anthropic
from io import BytesIO
from PIL import Image
import base64

class Claude_3_Opus(OCRBaseModel):
    def info(self):
        return OCRModelInfo(
            name="Claude 3 Opus",
            version="claude-3-opus-20240229",
            tags=["cloud", "lmm"]
        )

    def __init__(self, api_key: str):
        super().__init__()
        # Store the API key and initialize the client
        # It's generally recommended to initialize the client once in __init__
        # rather than in every evaluate call.
        self.api_key = api_key
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model_version = self.info().version # Store model version for API call

    def evaluate(self, image) -> OCRModelResponse:
        # Convert numpy array image to base64 encoded JPEG
        buffered = BytesIO()
        Image.fromarray(image).save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        try:
            message = self.client.messages.create(
                model=self.model_version,
                max_tokens=1000, # Consider making max_tokens configurable if needed
                temperature=0, # Using 0 temperature for deterministic output
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
                                "text": "Read the text in the image. Return only the text as it is visible in the image."
                            }
                        ]
                    }
                ]
            )

            # Extract text prediction
            prediction = message.content[0].text

            # Calculate cost based on usage (Opus pricing: $15/M input, $75/M output)
            input_cost = (message.usage.input_tokens / 1_000_000) * 15
            output_cost = (message.usage.output_tokens / 1_000_000) * 75
            cost = input_cost + output_cost

            return OCRModelResponse(
                prediction=prediction,
                cost=cost
            )

        except Exception as e:
            # Handle potential API errors gracefully
            print(f"Error during Claude API call: {e}")
            # Return an empty response or re-raise, depending on desired behavior
            return OCRModelResponse(
                prediction="",
                cost=0,
                error=str(e) # Optionally include error information
            ) 