import easyocr
from ocrcheckup.benchmark.model import OCRBaseModel, OCRModelResponse, OCRModelInfo
import importlib.metadata
import torch # For device check

# Note: easyocr might download language models on first run.

class EasyOCR(OCRBaseModel):
    def info(self) -> OCRModelInfo:
        """Provides information about the EasyOCR model."""
        try:
            # Attempt to get the installed package version
            version = importlib.metadata.version('easyocr')
        except importlib.metadata.PackageNotFoundError:
            version = "unknown" # Fallback if not found

        return OCRModelInfo(
            name="EasyOCR",
            version=version,
            tags=['local'],
            cost_type='compute'
        )

    def __init__(self, cost_per_second: float = None):
        super().__init__(cost_per_second=cost_per_second)

        use_gpu = False
        if torch.cuda.is_available():
            print("EasyOCR: CUDA detected, enabling GPU.")
            use_gpu = True
        else:
            print("EasyOCR: CUDA not available, using CPU.")
            use_gpu = False

        print("Initializing EasyOCR Reader...")
        
        self.reader = easyocr.Reader(['en'], gpu=use_gpu)
        print(f"EasyOCR Reader initialized (GPU={use_gpu}).")

    def evaluate(self, image) -> OCRModelResponse:
        # image is expected as a NumPy array from the benchmark framework
        # Use detail=0 to get only the text, as in the example
        result = self.reader.readtext(image, detail=0)

        # Join the list of text blocks into a single string
        text = " ".join(result)

        # Cost is handled by the base class's run_for_eval method
        return OCRModelResponse(
            prediction=text
        ) 