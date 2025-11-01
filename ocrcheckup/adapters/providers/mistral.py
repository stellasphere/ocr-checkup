from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional

from markdown_it import MarkdownIt
from mistralai import Mistral
from PIL import Image

from ocrcheckup.adapters.base import AdapterOutput
from ocrcheckup.adapters.utils import rate_limit
from ocrcheckup.core.types import Sample
from ocrcheckup.core.variant import Variant


DEFAULT_RPM = 360


class MistralOcrApiAdapter:
    id = "mistral-ocr-api"
    description = "Mistral OCR hosted API adapter"

    def __init__(self) -> None:
        self._client: Optional[Mistral] = None
        self._markdown = MarkdownIt()

    def _client_instance(self) -> Mistral:
        if self._client is None:
            self._client = Mistral()
        return self._client

    def hash_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not config:
            return {}
        hashed: Dict[str, Any] = {}
        if "rpm" in config and config["rpm"] is not None:
            hashed["rpm"] = int(config["rpm"])
        return hashed

    def run(self, variant: Variant, sample: Sample) -> AdapterOutput:
        model = str(variant.fields.get("model", "mistral-ocr-2503"))
        parse_markdown = bool(variant.fields.get("parse_markdown", True))
        max_pages = variant.fields.get("max_pages")

        cfg = variant.adapter.config or {}
        rpm = int(cfg.get("rpm", DEFAULT_RPM))
        rate_limit(self.id, rpm)

        image_path = Path(sample.image_uri)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at {image_path}")

        with Image.open(image_path) as img:
            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        client = self._client_instance()
        response = client.ocr.process(
            model=model,
            document={
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{image_b64}",
            },
        )

        pages = response.pages
        if max_pages is not None:
            pages = pages[: int(max_pages)]

        if parse_markdown:
            parts = []
            for page in pages:
                tokens = self._markdown.parse(page.markdown)
                parts.append("".join(token.content for token in tokens).strip())
            prediction = "\n".join(filter(None, parts)).strip()
        else:
            prediction = "\n".join(page.markdown for page in pages).strip()

        usage = getattr(response, "usage_info", None)
        metadata: Dict[str, Any] = {
            "provider": "mistral",
            "model": model,
        }
        if usage is not None:
            metadata["usage"] = {
                "pages_processed": getattr(usage, "pages_processed", None),
            }

        return AdapterOutput(prediction=prediction, metadata=metadata)


adapter = MistralOcrApiAdapter()


__all__ = ["adapter", "MistralOcrApiAdapter"]
