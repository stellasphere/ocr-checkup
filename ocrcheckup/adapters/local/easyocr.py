from __future__ import annotations

from typing import Dict, List, Tuple

import easyocr
import torch

from ocrcheckup.adapters.base import AdapterOutput, BaseAdapter
from ocrcheckup.core.types import Sample
from ocrcheckup.core.variant import Variant


class EasyOCRAdapter(BaseAdapter):
    id = "easyocr"
    description = "EasyOCR adapter"

    def __init__(self) -> None:
        self._reader_cache: Dict[Tuple[Tuple[str, ...], bool], easyocr.Reader] = {}

    def _get_reader(self, langs: List[str], use_gpu: bool) -> easyocr.Reader:
        key = (tuple(langs), use_gpu)
        if key not in self._reader_cache:
            self._reader_cache[key] = easyocr.Reader(langs, gpu=use_gpu)
        return self._reader_cache[key]

    def run(self, variant: Variant, sample: Sample) -> AdapterOutput:
        langs = variant.fields.get("langs") or ["en"]
        if not isinstance(langs, list):
            raise ValueError("EasyOCRAdapter expects 'langs' field to be a list of language codes.")
        use_gpu_field = variant.fields.get("gpu")
        use_gpu = bool(use_gpu_field) if use_gpu_field is not None else torch.cuda.is_available()

        reader = self._get_reader(langs, use_gpu)
        result = reader.readtext(str(sample.image_uri), detail=0)
        prediction = " ".join(result).strip()

        metadata = {"provider": "easyocr", "langs": langs}
        metadata["gpu"] = use_gpu

        return AdapterOutput(prediction=prediction, metadata=metadata)
