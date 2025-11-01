from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image

import easyocr

from ocrcheckup.adapters.base import AdapterOutput
from ocrcheckup.core.types import Sample
from ocrcheckup.core.variant import Variant


class EasyOcrReaderAdapter:
    id = "easyocr-reader"
    description = "EasyOCR adapter"

    def __init__(self) -> None:
        self._reader: Optional[easyocr.Reader] = None
        self._reader_signature: Optional[tuple] = None

    def hash_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        return {}

    def _ensure_reader(self, variant: Variant) -> None:
        languages = tuple(variant.fields.get("languages", ["en"]))
        adapter_cfg = variant.adapter.config or {}
        gpu_override = adapter_cfg.get("gpu")

        if gpu_override is None:
            use_gpu = torch.cuda.is_available()
        else:
            use_gpu = bool(gpu_override)

        signature = (languages, use_gpu)
        if self._reader is not None and self._reader_signature == signature:
            return

        self._reader = easyocr.Reader(list(languages), gpu=use_gpu)
        self._reader_signature = signature

    def run(self, variant: Variant, sample: Sample) -> AdapterOutput:
        self._ensure_reader(variant)
        assert self._reader is not None

        detail = 0
        paragraph = bool(variant.fields.get("paragraph", False))

        image_path = Path(sample.image_uri)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at {image_path}")

        with Image.open(image_path) as img:
            array = np.array(img.convert("RGB"))

        result = self._reader.readtext(array, detail=detail, paragraph=paragraph)

        if detail == 0:
            texts = [segment for segment in result if isinstance(segment, str)]
        else:
            texts = [segment[1] for segment in result]

        prediction = "\n".join(text.strip() for text in texts if text).strip()

        metadata = {
            "provider": "easyocr",
        }

        return AdapterOutput(prediction=prediction, metadata=metadata)


adapter = EasyOcrReaderAdapter()


__all__ = ["adapter", "EasyOcrReaderAdapter"]
