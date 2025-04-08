import os
import base64
from io import BytesIO
from PIL import Image
import ocrcheckup.utils
from ocrcheckup.benchmark.model import OCRBaseModel, OCRModelResponse, OCRModelInfo
from mistralai import Mistral
from ..rate_limiter import RateLimiter
from ..cost import ModelCost, CostType
from markdown_it import MarkdownIt

class MistralOCR(OCRBaseModel):
    def info(self) -> OCRModelInfo:
        return OCRModelInfo(
            name="Mistral OCR",
            version="mistral-ocr-2503",
            tags=["cloud"],
            cost_type="api"
        )

    def __init__(self, cost_per_second: float = None):
        limiter = RateLimiter(requests_per_minute=360)
        super().__init__(cost_per_second=cost_per_second, rate_limiter=limiter)

        self.model_id = self.info().version

        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY environment variable not set.")
        try:
            self.client = Mistral(api_key=api_key)
        except Exception as e:
            print(f"Mistral client initialization failed: {e}")
            raise

    def evaluate(self, image) -> OCRModelResponse:
        if not isinstance(image, Image.Image):
             pil_image = Image.fromarray(image)
        else:
             pil_image = image

        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        prediction = ""
        cost_data = None

        try:
            ocr_response = self.client.ocr.process(
                model=self.model_id,
                document={
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{img_str}"
                }
            )
            
            cost_data = ModelCost(
                cost_type=CostType.EXTERNAL,
                info={
                    "model_id": self.model_id,
                    "pages": ocr_response.usage_info.pages_processed
                }
            )

            try:
                # Initialize markdown parser
                md = MarkdownIt()

                text_prediction = ""
                for page in ocr_response.pages:
                    # Parse markdown to tokens
                    tokens = md.parse(page.markdown)
                    # Extract text content from tokens
                    page_text = "".join([token.content for token in tokens])
                    text_prediction += page_text.strip() + "\n" # Add newline between pages if needed

                # Remove trailing newline if any
                prediction = text_prediction.strip()

            except Exception as e:
                 print(f"Mistral OCR ({self.model_id}): Error processing markdown response: {e}")
                 # Fallback: return raw markdown if parsing fails
                 prediction = "\n".join([page.markdown for page in ocr_response.pages])

        except Exception as e:
            print(f"Mistral OCR ({self.model_id}): Error during API call: {e}")
            raise

        return OCRModelResponse(
            prediction=prediction,
            cost_details=cost_data
        ) 