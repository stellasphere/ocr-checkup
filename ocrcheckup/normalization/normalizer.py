from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Literal, Optional

from pydantic import BaseModel


class NormalizationSpec(BaseModel):
    unicode_form: Literal["NFC", "NFD", "NFKC", "NFKD"]
    lowercase: bool
    collapse_whitespace: bool
    remove_punctuation: bool
    spec_version: str
    description: Optional[str] = None


class Normalizer:
    def __init__(self, spec: NormalizationSpec) -> None:
        self.spec = spec

    def id(self) -> str:
        payload = self.spec.model_dump(mode="json", by_alias=True, exclude_none=True)
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        return digest[:16]

    def normalize(self, text: str) -> str:
        t = unicodedata.normalize(self.spec.unicode_form, text)
        if self.spec.lowercase:
            t = t.lower()
        if self.spec.remove_punctuation:
            t = "".join(c for c in t if not unicodedata.category(c).startswith("P"))
        if self.spec.collapse_whitespace:
            t = re.sub(r"\s+", " ", t).strip()
        return t


__all__ = ["NormalizationSpec", "Normalizer"]
