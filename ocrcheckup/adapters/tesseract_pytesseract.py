from __future__ import annotations

from typing import Any, Dict

import pytesseract
from PIL import Image

from ocrcheckup.adapters.base import AdapterOutput, BaseAdapter
from ocrcheckup.core.types import Sample
from ocrcheckup.core.variant import Variant


class TesseractPyAdapter(BaseAdapter):
    id = "tesseract-pytesseract"
    description = "Tesseract OCR via pytesseract"

    def run(self, variant: Variant, sample: Sample) -> AdapterOutput:
        lang = str(variant.fields.get("lang", "eng"))
        psm = variant.fields.get("psm")

        img = Image.open(sample.image_uri)
        config = f"--psm {int(psm)}" if psm is not None else ""
        text = pytesseract.image_to_string(img, lang=lang, config=config)
        meta: Dict[str, Any] = {"engine": "pytesseract", "lang": lang}
        if psm is not None:
            meta["psm"] = int(psm)
        return AdapterOutput(prediction=text, metadata=meta)


adapter = TesseractPyAdapter()
