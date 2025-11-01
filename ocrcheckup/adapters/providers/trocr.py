from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from ocrcheckup.adapters.base import AdapterOutput
from ocrcheckup.core.types import Sample
from ocrcheckup.core.variant import Variant


class TrocrTransformersAdapter:
    id = "trocr-transformers"
    description = "Microsoft TrOCR encoder-decoder adapter"

    def __init__(self) -> None:
        self._model_id: Optional[str] = None
        self._model = None
        self._processor = None
        self._device: Optional[torch.device] = None

    def hash_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        return {}

    def _resolve_device(self, device_hint: Optional[str]) -> torch.device:
        if device_hint:
            return torch.device(device_hint)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _ensure_loaded(self, variant: Variant) -> None:
        model_id = str(variant.fields.get("model", "microsoft/trocr-base-printed"))
        if self._model_id == model_id and self._model is not None:
            return

        device = self._resolve_device(variant.fields.get("device"))

        self._model = VisionEncoderDecoderModel.from_pretrained(model_id).to(device)
        self._model.eval()
        self._processor = TrOCRProcessor.from_pretrained(model_id)

        self._model_id = model_id
        self._device = device

    def run(self, variant: Variant, sample: Sample) -> AdapterOutput:
        self._ensure_loaded(variant)
        assert self._model is not None and self._processor is not None and self._device is not None

        max_new_tokens = variant.fields.get("max_new_tokens")

        image_path = Path(sample.image_uri)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at {image_path}")

        with Image.open(image_path) as img:
            image_pil = img.convert("RGB")

        inputs = self._processor(image_pil, return_tensors="pt").pixel_values.to(self._device)

        generate_kwargs: Dict[str, Any] = {}
        if max_new_tokens is not None:
            generate_kwargs["max_new_tokens"] = int(max_new_tokens)

        with torch.inference_mode():
            generated = self._model.generate(inputs, **generate_kwargs)

        prediction = self._processor.batch_decode(
            generated,
            skip_special_tokens=True,
        )[0].strip()

        metadata = {
            "provider": "trocr",
            "model": self._model_id,
        }

        return AdapterOutput(prediction=prediction, metadata=metadata)


adapter = TrocrTransformersAdapter()


__all__ = ["adapter", "TrocrTransformersAdapter"]
