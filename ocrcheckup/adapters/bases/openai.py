from __future__ import annotations

import base64
import os
from io import BytesIO
from pathlib import Path
from typing import Optional

from PIL import Image
from openai import OpenAI


class OpenAIAdapterBase:
    """Shared OpenAI adapter utilities: client setup and image encoding."""

    def __init__(self) -> None:
        self._client: Optional[OpenAI] = None

    def setup(self) -> None:
        if self._client is not None:
            return
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        self._client = OpenAI(api_key=api_key)

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            raise RuntimeError("OpenAI client is not initialized. Call setup() first.")
        return self._client

    def encode_image_to_base64(self, image_uri: str | Path) -> str:
        with Image.open(image_uri) as img:
            buffer = BytesIO()
            img.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


