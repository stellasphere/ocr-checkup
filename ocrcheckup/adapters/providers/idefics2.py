from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
from PIL import Image
from transformers import AutoProcessor, Idefics2ForConditionalGeneration

from ocrcheckup.adapters.base import AdapterOutput
from ocrcheckup.core.types import Sample
from ocrcheckup.core.variant import Variant


_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


class Idefics2Adapter:
    id = "idefics2-transformers"
    description = "Idefics2 multimodal adapter"

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
        model_id = str(variant.fields.get("model", "HuggingFaceM4/idefics2-8b"))
        if self._model_id == model_id and self._model is not None:
            return

        torch_dtype_key = variant.fields.get("torch_dtype", "float16")
        torch_dtype = _DTYPE_MAP.get(torch_dtype_key, torch.float16)
        device = self._resolve_device(variant.fields.get("device"))

        self._processor = AutoProcessor.from_pretrained(
            model_id,
            do_image_splitting=False,
        )
        self._model = Idefics2ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
        ).to(device)
        self._model.eval()

        self._model_id = model_id
        self._device = device

    def run(self, variant: Variant, sample: Sample) -> AdapterOutput:
        self._ensure_loaded(variant)
        assert self._model is not None and self._processor is not None and self._device is not None

        prompt = str(variant.fields.get("prompt"))
        max_new_tokens = int(variant.fields.get("max_new_tokens", 128))

        image_path = Path(sample.image_uri)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at {image_path}")

        with Image.open(image_path) as img:
            image_pil = img.convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
        inputs = self._processor(
            text=[text.strip()],
            images=[image_pil],
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
            )

        input_ids = inputs["input_ids"]
        trimmed = generated[:, input_ids.size(1) :]
        output_text = self._processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
        )
        prediction = output_text[0].strip() if output_text else ""

        metadata = {
            "provider": "idefics2",
            "model": self._model_id,
            "device": str(self._device),
            "max_new_tokens": max_new_tokens,
        }

        return AdapterOutput(prediction=prediction, metadata=metadata)


adapter = Idefics2Adapter()


__all__ = ["adapter", "Idefics2Adapter"]
