from ocrcheckup.benchmark.model import OCRBaseModel, OCRModelResponse, OCRModelInfo

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
from PIL import Image # Import PIL

class TrOCR(OCRBaseModel):
    # Implement the standard info method directly
    def info(self) -> OCRModelInfo: # Changed to instance method for consistency
        return OCRModelInfo(
            name='TrOCR', # Use the requested name
            version="microsoft/trocr-base-printed", # Static version based on checkpoint
            tags=['local'], # TrOCR runs locally
            cost_type='compute' # Cost is based on compute time
        )

    def __init__(self):
        # Get model info to retrieve version/checkpoint name
        model_info = self.info()
        self.model_id = model_info.version # Set model_id from info

        super().__init__(model_id=self.model_id)

        # Determine device
        if torch.cuda.is_available():
            DEVICE = "cuda:0"
        elif torch.backends.mps.is_available():
            DEVICE = "mps"
        else:
            DEVICE = "cpu"

        self.device = DEVICE
        print(f"TrOCR initializing on device: {self.device} with checkpoint {self.model_id}") # Updated print

        # Load model and processor using self.model_id to the determined device
        self.model = VisionEncoderDecoderModel.from_pretrained(
            self.model_id
        ).to(self.device).eval() # Move model to device

        self.processor = TrOCRProcessor.from_pretrained(
            self.model_id
        )

    def evaluate(self, image) -> OCRModelResponse:
        # Convert numpy array (expected input) to PIL Image
        image_pil = Image.fromarray(image)

        # Process the image using the processor
        pixel_values = self.processor(image_pil, return_tensors="pt").pixel_values.to(self.device) # Move tensor to device

        # Generate text using the model
        with torch.inference_mode():
            generated_ids = self.model.generate(pixel_values)

        # Decode the generated IDs to text
        result = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return OCRModelResponse(
            prediction=result
        )
