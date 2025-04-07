import torch
from PIL import Image
from transformers import AutoProcessor, Idefics2ForConditionalGeneration

from ocrcheckup.benchmark.model import OCRBaseModel, OCRModelInfo, OCRModelResponse


class Idefics2(OCRBaseModel):
    def info(self) -> OCRModelInfo:
        return OCRModelInfo(
            name="Idefics2",
            version="HuggingFaceM4/idefics2-8b",
            tags=["local", "lmm"],
            cost_type="compute",
        )

    def __init__(self, cost_per_second: float = None):
        model_info = self.info()
        self.model_id = model_info.version
        super().__init__(cost_per_second=cost_per_second, model_id=self.model_id)

        if torch.cuda.is_available():
            DEVICE = "cuda:0"
        elif torch.backends.mps.is_available():
            DEVICE = "mps"
        else:
            DEVICE = "cpu"

        self.device = DEVICE
        print(
            f"Idefics2 initializing on device: {self.device} with checkpoint {self.model_id}"
        )

        self.processor = AutoProcessor.from_pretrained(
            self.model_id, do_image_splitting=False
        )

        self.model = (
            Idefics2ForConditionalGeneration.from_pretrained(
                self.model_id, torch_dtype=torch.float16
            )
            .to(self.device)
            .eval()
        )

        self.prompt = "Read the text in the image. Return only the text as it is visible in the image."

    def evaluate(self, image) -> OCRModelResponse:
        image_pil = Image.fromarray(image)

        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": self.prompt}],
            }
        ]

        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            text=[text.strip()], images=[image_pil], return_tensors="pt", padding=True
        ).to(self.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=64)

        generated_texts = self.processor.batch_decode(
            generated_ids[:, inputs["input_ids"].size(1) :], skip_special_tokens=True
        )
        result = generated_texts[0]

        return OCRModelResponse(prediction=result) 