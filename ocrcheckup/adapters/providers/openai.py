from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional

from openai import OpenAI
from PIL import Image

from ocrcheckup.adapters.base import AdapterOutput
from ocrcheckup.core.types import Sample
from ocrcheckup.core.variant import Variant
from ocrcheckup.adapters.utils import rate_limit


DEFAULT_RPM = 240


class OpenAIVisionChatAdapter:
    id = "openai-vision-chat"
    description = "OpenAI chat.completions vision adapter"

    def __init__(self) -> None:
        self._client: Optional[OpenAI] = None

    def _client_instance(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI()
        return self._client

    def hash_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not config:
            return {}
        hashed: Dict[str, Any] = {}
        if "rpm" in config and config["rpm"] is not None:
            hashed["rpm"] = int(config["rpm"])
        return hashed

    def run(self, variant: Variant, sample: Sample) -> AdapterOutput:
        model = str(variant.fields.get("model"))
        prompt = str(variant.fields.get("prompt"))
        detail = variant.fields.get("detail", "high") or "high"
        max_output_tokens = variant.fields.get("max_output_tokens")
        temperature = variant.fields.get("temperature")
        top_p = variant.fields.get("top_p")

        cfg = variant.adapter.config or {}
        rpm = int(cfg.get("rpm", DEFAULT_RPM))
        rate_limit(self.id, rpm)

        image_path = Path(sample.image_uri)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at {image_path}")

        with Image.open(image_path) as img:
            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

        message_content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image}",
                    "detail": detail,
                },
            },
        ]

        params: Dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": message_content,
                }
            ],
        }
        if max_output_tokens is not None:
            params["max_tokens"] = int(max_output_tokens)
        if temperature is not None:
            params["temperature"] = float(temperature)
        if top_p is not None:
            params["top_p"] = float(top_p)

        client = self._client_instance()
        response = client.chat.completions.create(**params)

        prediction = ""
        if response.choices:
            choice = response.choices[0]
            content = getattr(choice.message, "content", None)
            if isinstance(content, str):
                prediction = content.strip()
            elif isinstance(content, list):
                segments = [
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                ]
                prediction = "".join(segments).strip()

        usage = getattr(response, "usage", None)
        metadata: Dict[str, Any] = {
            "provider": "openai",
            "model": model,
        }
        if getattr(response, "id", None):
            metadata["response_id"] = response.id
        if usage is not None:
            metadata["usage"] = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }

        return AdapterOutput(prediction=prediction, metadata=metadata)


adapter = OpenAIVisionChatAdapter()


__all__ = ["adapter", "OpenAIVisionChatAdapter"]
