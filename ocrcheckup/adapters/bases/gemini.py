from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from PIL import Image
from google import genai


class GeminiAdapterBase:
    """Shared Gemini adapter utilities: client setup and image loading."""

    def __init__(self) -> None:
        self._client: Optional[genai.Client] = None

    def setup(self) -> None:
        if self._client is not None:
            return
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        self._client = genai.Client(api_key=api_key)

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            raise RuntimeError("Gemini client is not initialized. Call setup() first.")
        return self._client

    def load_image(self, image_uri: str | Path) -> Image.Image:
        return Image.open(image_uri)


