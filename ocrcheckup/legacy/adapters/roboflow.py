#qwen-2-5-vl,florence-2-large,florence-2-base,smolvlm-22b-instruct

import os
import abc
import base64
from io import BytesIO
from PIL import Image
from inference_sdk import InferenceHTTPClient
import time

from ocrcheckup.benchmark.model import OCRBaseModel, OCRModelResponse, OCRModelInfo
from ..rate_limiter import RateLimiter
from ..cost import ModelCost, CostType

# Abstract Base Class for Roboflow hosted models
class _RoboflowBase(OCRBaseModel, abc.ABC):
    """
    Abstract base class for Roboflow Hosted Workflow models.
    Handles common initialization (API key, client, rate limiter, workspace/workflow IDs)
    and evaluation logic using inference_sdk.
    """
    def __init__(self, model_id: str, rpm: int = 100):
        limiter = RateLimiter(rpm)
        super().__init__(rate_limiter=limiter)
        self.model_id = model_id # This is the model parameter for the workflow

        # --- Configuration from Environment Variables ---
        self.api_key = os.environ.get("ROBOFLOW_API_KEY")
        if not self.api_key:
            raise ValueError("ROBOFLOW_API_KEY environment variable not set.")

        # Use static values for workspace and workflow ID
        self.workspace_name = "leo-ueno"
        self.workflow_id = "ocr"

        # Default to hosted URL, allow override via environment variable
        self.api_url = os.environ.get("ROBOFLOW_API_URL", "https://serverless.roboflow.com")

        try:
            self.client = InferenceHTTPClient(
                api_url=self.api_url,
                api_key=self.api_key
            )
        except Exception as e:
            print(f"Roboflow Client initialization failed: {e}")
            raise

    @abc.abstractmethod
    def info(self) -> OCRModelInfo:
        """Return model information."""
        pass

    def evaluate(self, image) -> OCRModelResponse:
        """
        Evaluates the model on a single image using the Roboflow Workflow API.
        """
        if not isinstance(image, Image.Image):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG") # Use JPEG format
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        prediction = ""

        start_time = time.perf_counter()
        result = self.client.run_workflow(
            workspace_name=self.workspace_name,
            workflow_id=self.workflow_id,
            images={
                "image": img_str
            },
            parameters={
                # Pass the specific model ID as a parameter to the workflow
                "model": self.model_id.replace("-roboflow-hosted", "")
            },
            use_cache=False
        )
        base_model_id = self.model_id.replace("-roboflow-hosted", "")
        elapsed_time = time.perf_counter() - start_time

        cost_details = ModelCost(cost_type=CostType.EXTERNAL, info={"model_id": "roboflow-serverless", "elapsed_time": elapsed_time})

        if (
            isinstance(result, list) and len(result) > 0 and
            isinstance(result[0], dict) and
            base_model_id in result[0] and
            isinstance(result[0][base_model_id], dict) and
            "raw_output" in result[0][base_model_id]
        ):
            raw_pred = result[0][base_model_id]["raw_output"]
            # Remove potential surrounding quotes and strip whitespace
            if isinstance(raw_pred, str):
                prediction = raw_pred.strip().strip('"').strip()
            else:
                prediction = str(raw_pred).strip() # Fallback for non-string types
        else:
            print(f"Roboflow ({self.model_id}): Received unexpected or incomplete response format.")
            print(f"Response: {result}")


        return OCRModelResponse(
            prediction=prediction,
            cost_details=cost_details
        )


class Florence2Large(_RoboflowBase):
  def __init__(self):
    super().__init__(model_id="florence-2-large-roboflow-hosted")

  def info(self) -> OCRModelInfo:
    return OCRModelInfo(
      name = "Florence 2 Large",
      version = self.model_id,
      tags = ["cloud", "lmm"],
      cost_type="api"
    )

class Florence2Base(_RoboflowBase):
  def __init__(self):
    super().__init__(model_id="florence-2-base-roboflow-hosted")

  def info(self) -> OCRModelInfo:
    return OCRModelInfo(
      name = "Florence 2 Base",
      version = self.model_id,
      tags = ["cloud", "lmm"],
      cost_type="api"
    )
