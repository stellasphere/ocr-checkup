from __future__ import annotations

from typing import Dict, Tuple

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from ocrcheckup.adapters.base import AdapterOutput, BaseAdapter
from ocrcheckup.core.types import Sample
from ocrcheckup.core.variant import Variant


class TrOCRAdapter(BaseAdapter):
    id = "trocr"
    description = "Microsoft TrOCR adapter"

    def __init__(self) -> None:
        self._models: Dict[str, Tuple[VisionEncoderDecoderModel, TrOCRProcessor]] = {}

    def _get_model(self, model_version: str) -> Tuple[VisionEncoderDecoderModel, TrOCRProcessor]:
        if model_version not in self._models:
            model = VisionEncoderDecoderModel.from_pretrained(model_version)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model = model.to(device).eval()
            processor = TrOCRProcessor.from_pretrained(model_version)
            self._models[model_version] = (model, processor)
        return self._models[model_version]

    def run(self, variant: Variant, sample: Sample) -> AdapterOutput:
        model_version = variant.fields.get("model_version") or "microsoft/trocr-base-printed"
        model, processor = self._get_model(model_version)
        device = next(model.parameters()).device

        with Image.open(sample.image_uri) as image:
            pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

        with torch.inference_mode():
            generated_ids = model.generate(pixel_values)

        prediction = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        metadata = {"provider": "trocr", "model_version": model_version}
        return AdapterOutput(prediction=prediction, metadata=metadata)
