from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional

from anthropic import Anthropic
from PIL import Image

from ocrcheckup.adapters.base import AdapterOutput
from ocrcheckup.core.types import Sample
from ocrcheckup.core.variant import Variant
from ocrcheckup.rate_limiter import RateLimiter


DEFAULT_RPM = 50


class AnthropicClaudeAdapter:
    id = "anthropic-claude-messages"
    description = "Anthropic Claude messages API with vision support"

    def __init__(self) -> None:
        self._client: Optional[Anthropic] = None
        self._limiters: Dict[int, RateLimiter] = {}

    def _client_instance(self) -> Anthropic:
        if self._client is None:
            self._client = Anthropic()
        return self._client

    def _rate_limiter(self, rpm: int) -> RateLimiter:
        limiter = self._limiters.get(rpm)
        if limiter is None:
            limiter = RateLimiter(rpm)
            self._limiters[rpm] = limiter
        return limiter

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
        max_tokens = int(variant.fields.get("max_output_tokens", 1024))
        temperature = variant.fields.get("temperature")
        top_k = variant.fields.get("top_k")
        top_p = variant.fields.get("top_p")

        cfg = variant.adapter.config or {}
        rpm = int(cfg.get("rpm", DEFAULT_RPM))
        if rpm <= 0:
            raise ValueError("rpm must be positive")
        self._rate_limiter(rpm).wait_if_needed()

        image_path = Path(sample.image_uri)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at {image_path}")

        with Image.open(image_path) as img:
            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": image_b64,
                },
            },
            {"type": "text", "text": prompt},
        ]

        params: Dict[str, Any] = {
            "model": model,
            "max_output_tokens": max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
        }
        if temperature is not None:
            params["temperature"] = float(temperature)
        if top_k is not None:
            params["top_k"] = int(top_k)
        if top_p is not None:
            params["top_p"] = float(top_p)

        client = self._client_instance()
        response = client.messages.create(**params)

        text_segments = []
        for block in getattr(response, "content", []) or []:
            if getattr(block, "type", None) == "text":
                text_segments.append(getattr(block, "text", ""))
        prediction = "".join(text_segments).strip()

        usage = getattr(response, "usage", None)
        metadata: Dict[str, Any] = {
            "provider": "anthropic",
            "model": model,
        }
        if getattr(response, "id", None):
            metadata["response_id"] = response.id
        if usage is not None:
            metadata["usage"] = {
                "input_tokens": getattr(usage, "input_tokens", None),
                "output_tokens": getattr(usage, "output_tokens", None),
            }

        return AdapterOutput(prediction=prediction, metadata=metadata)


adapter = AnthropicClaudeAdapter()


__all__ = ["adapter", "AnthropicClaudeAdapter"]
