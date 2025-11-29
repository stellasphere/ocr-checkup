from __future__ import annotations

from typing import Dict, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from ocrcheckup.adapters.base import AdapterOutput, BaseAdapter
from ocrcheckup.core.types import Sample
from ocrcheckup.core.variant import Variant

DEFAULT_PROMPT = "Read the text in the image. Return only the text as it is visible in the image."


class QwenAdapter(BaseAdapter):
    id = "qwen"
    description = "Qwen2.5-VL adapter"

    def __init__(self) -> None:
        self._models: Dict[str, Tuple[Qwen2_5_VLForConditionalGeneration, AutoProcessor]] = {}

    def _get_model(self, model_version: str) -> Tuple[Qwen2_5_VLForConditionalGeneration, AutoProcessor]:
        if model_version not in self._models:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_version,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
            )
            model.eval()
            processor = AutoProcessor.from_pretrained(model_version, trust_remote_code=True)
            self._models[model_version] = (model, processor)
        return self._models[model_version]

    def run(self, variant: Variant, sample: Sample) -> AdapterOutput:
        model_version = variant.fields.get("model_version") or "Qwen/Qwen2.5-VL-7B-Instruct"
        prompt = variant.fields.get("prompt") or DEFAULT_PROMPT
        model, processor = self._get_model(model_version)
        device = next(model.parameters()).device

        image = Image.open(sample.image_uri)
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=512)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        outputs = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        prediction = outputs[0].strip() if outputs else ""

        metadata = {"provider": "qwen", "model_version": model_version}
        return AdapterOutput(prediction=prediction, metadata=metadata)
