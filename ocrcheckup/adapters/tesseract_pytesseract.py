from __future__ import annotations

from typing import Any, Dict, Optional

from PIL import Image
import pytesseract

from ocrcheckup.adapters.base import Adapter, AdapterOutput
from ocrcheckup.core.types import Sample
from ocrcheckup.core.variant import Variant


class TesseractPyAdapter:
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

    def hash_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        # No adapter-specific config; behavior driven by Variant.fields.
        return {}


adapter = TesseractPyAdapter()


