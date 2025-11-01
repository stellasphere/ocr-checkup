from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from google import genai
from google.genai import types as genai_types
from PIL import Image

from ocrcheckup.adapters.base import AdapterOutput
from ocrcheckup.core.types import Sample
from ocrcheckup.core.variant import Variant


class GeminiVisionAdapter:
    id = "google-gemini-vision"
    description = "Google Gemini vision adapter via genai SDK"

    def __init__(self) -> None:
        self._client: Optional[genai.Client] = None

    def _client_instance(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client()
        return self._client

    def hash_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        return {}

    def run(self, variant: Variant, sample: Sample) -> AdapterOutput:
        model = str(variant.fields.get("model"))
        prompt = str(variant.fields.get("prompt"))
        temperature = variant.fields.get("temperature")
        max_output_tokens = variant.fields.get("max_output_tokens")
        top_p = variant.fields.get("top_p")
        top_k = variant.fields.get("top_k")

        image_path = Path(sample.image_uri)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at {image_path}")

        with Image.open(image_path) as img:
            image_rgb = img.convert("RGB")

        config_kwargs: Dict[str, Any] = {}
        if temperature is not None:
            config_kwargs["temperature"] = float(temperature)
        if max_output_tokens is not None:
            config_kwargs["max_output_tokens"] = int(max_output_tokens)
        if top_p is not None:
            config_kwargs["top_p"] = float(top_p)
        if top_k is not None:
            config_kwargs["top_k"] = int(top_k)

        config = genai_types.GenerateContentConfig(**config_kwargs)

        client = self._client_instance()
        response = client.models.generate_content(
            model=model,
            contents=[prompt, image_rgb],
            config=config,
        )

        prediction = (response.text or "").strip()
        usage = getattr(response, "usage_metadata", None)
        metadata: Dict[str, Any] = {
            "provider": "google-gemini",
            "model": model,
        }
        if usage is not None:
            metadata["usage"] = {
                "prompt_tokens": getattr(usage, "prompt_token_count", None),
                "candidates_tokens": getattr(usage, "candidates_token_count", None),
                "total_tokens": getattr(usage, "total_token_count", None),
            }

        return AdapterOutput(prediction=prediction, metadata=metadata)


adapter = GeminiVisionAdapter()


__all__ = ["adapter", "GeminiVisionAdapter"]
