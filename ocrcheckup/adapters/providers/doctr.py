from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image
from inference_sdk import InferenceHTTPClient

from ocrcheckup.adapters.base import AdapterOutput
from ocrcheckup.core.types import Sample
from ocrcheckup.core.variant import Variant


class DocTRAdapter:
    id = "doctr-roboflow-hosted"
    description = "Roboflow-hosted DocTR API adapter"

    def __init__(self) -> None:
        self._client_key: Optional[Tuple[str, str]] = None
        self._client: Optional[InferenceHTTPClient] = None

    def hash_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not config:
            return {}
        hashed: Dict[str, Any] = {}
        if api_key := config.get("api_key"):
            hashed["api_key_last4"] = str(api_key)[-4:]
        return hashed

    def _ensure_client(self, api_url: str, api_key: str) -> None:
        signature = (api_url, api_key)
        if self._client is not None and self._client_key == signature:
            return
        self._client = InferenceHTTPClient(api_url=api_url, api_key=api_key)
        self._client_key = signature

    def run(self, variant: Variant, sample: Sample) -> AdapterOutput:
        adapter_cfg = variant.adapter.config or {}
        api_url = str(adapter_cfg.get("api_url", "https://infer.roboflow.com"))
        api_key = adapter_cfg.get("api_key") or os.getenv("ROBOFLOW_API_KEY")
        if not api_key:
            raise ValueError("DocTR adapter requires an API key via adapter.config['api_key'] or ROBOFLOW_API_KEY env var")

        self._ensure_client(api_url, api_key)
        assert self._client is not None

        image_path = Path(sample.image_uri)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at {image_path}")

        with Image.open(image_path) as img:
            image_array = np.array(img.convert("RGB"))

        result = self._client.ocr_image(image=image_array)
        prediction = str(result.get("result", "")).strip()

        metadata = {
            "provider": "roboflow",
            "model": "doctr",
            "api_url": api_url,
        }

        return AdapterOutput(prediction=prediction, metadata=metadata)


adapter = DocTRAdapter()


__all__ = ["adapter", "DocTRAdapter"]
