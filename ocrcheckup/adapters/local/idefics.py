from __future__ import annotations

from typing import Dict, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, Idefics2ForConditionalGeneration

from ocrcheckup.adapters.base import AdapterOutput, BaseAdapter
from ocrcheckup.core.types import Sample
from ocrcheckup.core.variant import Variant

DEFAULT_PROMPT = "Read the text in the image. Return only the text as it is visible in the image."


class IdeficsAdapter(BaseAdapter):
    id = "idefics"
    description = "Idefics2 vision-language adapter"

    def __init__(self) -> None:
        self._models: Dict[str, Tuple[Idefics2ForConditionalGeneration, AutoProcessor]] = {}

    def _get_model(self, model_version: str) -> Tuple[Idefics2ForConditionalGeneration, AutoProcessor]:
        if model_version not in self._models:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model = (
                Idefics2ForConditionalGeneration.from_pretrained(model_version, torch_dtype=torch.float16)
                .to(device)
                .eval()
            )
            processor = AutoProcessor.from_pretrained(model_version, do_image_splitting=False)
            self._models[model_version] = (model, processor)
        return self._models[model_version]

    def run(self, variant: Variant, sample: Sample) -> AdapterOutput:
        model_version = variant.fields.get("model_version") or "HuggingFaceM4/idefics2-8b"
        prompt = variant.fields.get("prompt") or DEFAULT_PROMPT
        model, processor = self._get_model(model_version)
        device = next(model.parameters()).device

        with Image.open(sample.image_uri) as image:
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(
                text=[text.strip()],
                images=[image],
                return_tensors="pt",
                padding=True,
            )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=128)

        offset = inputs["input_ids"].size(1)
        generated_texts = processor.batch_decode(
            generated_ids[:, offset:], skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        prediction = generated_texts[0].strip() if generated_texts else ""
        metadata = {"provider": "idefics", "model_version": model_version}
        return AdapterOutput(prediction=prediction, metadata=metadata)
