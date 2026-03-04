from __future__ import annotations

from typing import Dict, Tuple

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from ocrcheckup.adapters.base import AdapterOutput, BaseAdapter
from ocrcheckup.core.types import Sample
from ocrcheckup.core.variant import Variant

DEFAULT_PROMPT = "Read the text in the image. Return only the text as it is visible in the image."


class MoondreamAdapter(BaseAdapter):
    id = "moondream"
    description = "Moondream vision-language adapter"

    def __init__(self) -> None:
        self._models: Dict[str, Tuple[AutoModelForCausalLM, AutoTokenizer]] = {}

    def _get_model(self, model_version: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        if model_version not in self._models:
            device = torch.device("cpu")
            model = AutoModelForCausalLM.from_pretrained(
                model_version, trust_remote_code=True, device_map={"": device}
            )
            tokenizer = AutoTokenizer.from_pretrained(model_version, trust_remote_code=True)
            self._models[model_version] = (model, tokenizer)
        return self._models[model_version]

    def run(self, variant: Variant, sample: Sample) -> AdapterOutput:
        model_version = variant.fields.get("model_version") or "vikhyatk/moondream2"
        prompt = variant.fields.get("prompt") or DEFAULT_PROMPT
        model, tokenizer = self._get_model(model_version)

        with Image.open(sample.image_uri) as image:
            enc_image = model.encode_image(image)
        prediction = model.answer_question(enc_image, prompt, tokenizer).strip()

        metadata = {"provider": "moondream", "model_version": model_version}
        return AdapterOutput(prediction=prediction, metadata=metadata)
