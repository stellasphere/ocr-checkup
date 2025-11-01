from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from ocrcheckup.adapters.base import AdapterOutput
from ocrcheckup.core.types import Sample
from ocrcheckup.core.variant import Variant

try:
    from qwen_vl_utils import process_vision_info
except ImportError as exc:  # pragma: no cover - utility must be installed by caller
    raise ImportError("qwen_vl_utils is required for Qwen vision adapter") from exc


_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


class QwenVisionTransformersAdapter:
    id = "qwen-vision-transformers"
    description = "Qwen2.5 vision-language adapter backed by Transformers"

    def __init__(self) -> None:
        self._model_id: Optional[str] = None
        self._model: Optional[Qwen2_5_VLForConditionalGeneration] = None
        self._processor: Optional[AutoProcessor] = None
        self._device_override: Optional[torch.device] = None

    def hash_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        return {}

    def _ensure_loaded(self, variant: Variant) -> None:
        model_id = str(variant.fields.get("model"))
        if self._model_id == model_id and self._model is not None:
            return

        adapter_cfg = variant.adapter.config or {}
        dtype_key = adapter_cfg.get("torch_dtype", "auto")
        kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if dtype_key and dtype_key != "auto":
            if dtype_key not in _DTYPE_MAP:
                raise ValueError(f"Unsupported dtype '{dtype_key}' for Qwen adapter")
            kwargs["torch_dtype"] = _DTYPE_MAP[dtype_key]

        device_override = adapter_cfg.get("device")
        if device_override is not None:
            self._device_override = torch.device(str(device_override))
        else:
            self._device_override = None

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto" if self._device_override is None else None,
            **kwargs,
        )
        self._model.eval()

        if self._device_override is not None:
            self._model.to(self._device_override)

        self._processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
        )

        self._model_id = model_id

    def run(self, variant: Variant, sample: Sample) -> AdapterOutput:
        self._ensure_loaded(variant)
        assert self._model is not None and self._processor is not None

        adapter_cfg = variant.adapter.config or {}
        prompt = str(variant.fields.get("prompt"))
        max_new_tokens = int(variant.fields.get("max_new_tokens", 512))

        image_path = Path(sample.image_uri)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at {image_path}")

        with Image.open(image_path) as img:
            image_pil = img.convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_pil},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        )

        if self._device_override is not None:
            target_device = self._device_override
        else:
            target_device = getattr(self._model, "device", None)
            if target_device is None:
                target_device = next(self._model.parameters()).device

        inputs = {k: v.to(target_device) for k, v in inputs.items()}

        with torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
            )

        input_ids = inputs["input_ids"]
        trimmed = [out[len(inp) :] for inp, out in zip(input_ids, generated)]
        output_text = self._processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        prediction = output_text[0].strip() if output_text else ""

        metadata = {
            "provider": "qwen",
            "model": self._model_id,
        }

        return AdapterOutput(prediction=prediction, metadata=metadata)


adapter = QwenVisionTransformersAdapter()


__all__ = ["adapter", "QwenVisionTransformersAdapter"]
