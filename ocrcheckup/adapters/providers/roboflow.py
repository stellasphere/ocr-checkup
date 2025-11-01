from __future__ import annotations

import base64
import os
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from PIL import Image
from inference_sdk import InferenceHTTPClient

from ocrcheckup.adapters.base import AdapterOutput
from ocrcheckup.adapters.utils import rate_limit
from ocrcheckup.core.types import Sample
from ocrcheckup.core.variant import Variant


DEFAULT_RPM = 100


class RoboflowWorkflowAdapter:
    id = "roboflow-workflow"
    description = "Roboflow workflow inference adapter"

    def __init__(self) -> None:
        self._client_key: Optional[Tuple[str, str]] = None
        self._client: Optional[InferenceHTTPClient] = None

    def hash_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not config:
            return {}
        hashed: Dict[str, Any] = {}
        if (api_key := config.get("api_key")) is not None:
            hashed["api_key_last4"] = str(api_key)[-4:]
        if (rpm := config.get("rpm")) is not None:
            hashed["rpm"] = int(rpm)
        if (api_url := config.get("api_url")) is not None:
            hashed["api_url"] = str(api_url)
        return hashed

    def _ensure_client(self, api_url: str, api_key: str) -> None:
        signature = (api_url, api_key)
        if self._client is not None and self._client_key == signature:
            return
        self._client = InferenceHTTPClient(api_url=api_url, api_key=api_key)
        self._client_key = signature

    def run(self, variant: Variant, sample: Sample) -> AdapterOutput:
        fields = variant.fields
        workspace = str(fields.get("workspace"))
        workflow = str(fields.get("workflow"))
        model = str(fields.get("model"))
        parameters = dict(fields.get("parameters", {}))
        use_cache = bool(fields.get("use_cache", False))

        cfg = variant.adapter.config or {}
        api_key = cfg.get("api_key") or os.getenv("ROBOFLOW_API_KEY")
        if not api_key:
            raise ValueError("Roboflow adapter requires api_key via adapter config or ROBOFLOW_API_KEY env var")
        api_url = str(cfg.get("api_url", "https://serverless.roboflow.com"))
        rpm = int(cfg.get("rpm", DEFAULT_RPM))

        rate_limit(self.id, rpm)
        self._ensure_client(api_url, api_key)
        assert self._client is not None

        image_path = Path(sample.image_uri)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at {image_path}")

        with Image.open(image_path) as img:
            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        payload_parameters = {
            **parameters,
            "model": model,
        }

        start = time.perf_counter()
        result = self._client.run_workflow(
            workspace_name=workspace,
            workflow_id=workflow,
            images={"image": image_b64},
            parameters=payload_parameters,
            use_cache=use_cache,
        )
        elapsed = time.perf_counter() - start

        prediction = ""
        if isinstance(result, list) and result:
            entry = result[0]
            if isinstance(entry, dict):
                model_key = model.replace("-roboflow-hosted", "")
                node = entry.get(model_key)
                if isinstance(node, dict):
                    raw_output = node.get("raw_output")
                    if isinstance(raw_output, str):
                        prediction = raw_output.strip().strip('"').strip()
                    elif raw_output is not None:
                        prediction = str(raw_output).strip()

        metadata = {
            "provider": "roboflow",
            "model": model,
            "elapsed_seconds": elapsed,
        }

        return AdapterOutput(prediction=prediction, metadata=metadata)


adapter = RoboflowWorkflowAdapter()


__all__ = ["adapter", "RoboflowWorkflowAdapter"]
