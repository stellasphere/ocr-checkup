from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from ocrcheckup.adapters.base import AdapterOutput
from ocrcheckup.core.types import Sample
from ocrcheckup.core.variant import Variant


class MoondreamTransformersAdapter:
    id = "moondream-transformers"
    description = "Moondream2 vision-language adapter"

    def __init__(self) -> None:
        self._model_id: Optional[str] = None
        self._revision: Optional[str] = None
        self._model = None
        self._tokenizer = None
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
        model_id = str(variant.fields.get("model", "vikhyatk/moondream2"))
        revision = variant.fields.get("revision")
        if (
            self._model is not None
            and self._model_id == model_id
            and self._revision == revision
        ):
            return

        adapter_cfg = variant.adapter.config or {}
        device = self._resolve_device(adapter_cfg.get("device"))

        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            revision=revision,
        ).to(device)
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
        )

        self._model_id = model_id
        self._revision = revision
        self._device = device

    def run(self, variant: Variant, sample: Sample) -> AdapterOutput:
        self._ensure_loaded(variant)
        assert self._model is not None and self._tokenizer is not None and self._device is not None

        prompt = str(variant.fields.get("prompt"))

        image_path = Path(sample.image_uri)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at {image_path}")

        with Image.open(image_path) as img:
            image_pil = img.convert("RGB")

        enc = self._model.encode_image(image_pil)
        prediction = self._model.answer_question(enc, prompt, self._tokenizer)

        metadata = {
            "provider": "moondream",
            "model": self._model_id,
        }

        return AdapterOutput(prediction=prediction.strip(), metadata=metadata)


adapter = MoondreamTransformersAdapter()


__all__ = ["adapter", "MoondreamTransformersAdapter"]
