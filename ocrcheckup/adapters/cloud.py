from __future__ import annotations

import base64
from io import BytesIO
from typing import Dict, List

from markdown_it import MarkdownIt
from PIL import Image

from ocrcheckup.adapters.base import AdapterOutput, BaseAdapter
from ocrcheckup.adapters.bases import (
    AnthropicAdapterBase,
    GeminiAdapterBase,
    MistralAdapterBase,
    OpenAIAdapterBase,
    RoboflowAdapterBase,
)
from ocrcheckup.core.types import Sample
from ocrcheckup.core.variant import Variant

DEFAULT_PROMPT = "Read the text in the image. Return only the text as it is visible in the image."


class OpenAIAdapter(BaseAdapter, OpenAIAdapterBase):
    id = "openai"
    description = "OpenAI multimodal chat completion adapter"

    def setup(self) -> None:
        OpenAIAdapterBase.setup(self)

    def run(self, variant: Variant, sample: Sample) -> AdapterOutput:
        snapshot = str(variant.fields.get("snapshot"))
        if not snapshot:
            raise ValueError("Variant.fields.snapshot must be provided for OpenAIAdapter.")
        prompt = variant.fields.get("prompt") or DEFAULT_PROMPT
        image_detail = variant.fields.get("image_detail") or "high"
        if image_detail not in {"low", "high"}:
            raise ValueError("image_detail must be 'low' or 'high' when provided.")

        image_b64 = self.encode_image_to_base64(sample.image_uri)
        response = self.client.chat.completions.create(
            model=snapshot,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}", "detail": image_detail},
                        },
                    ],
                }
            ],
        )

        prediction = ""
        if response.choices and response.choices[0].message:
            prediction = (response.choices[0].message.content or "").strip()

        # pricing-only metadata
        metadata: Dict[str, object] = {}
        usage = getattr(response, "usage", None)
        if usage is not None:
            if getattr(usage, "prompt_tokens", None) is not None:
                metadata["input_tokens"] = usage.prompt_tokens
            if getattr(usage, "completion_tokens", None) is not None:
                metadata["output_tokens"] = usage.completion_tokens

        return AdapterOutput(prediction=prediction, metadata=metadata)


class AnthropicAdapter(BaseAdapter, AnthropicAdapterBase):
    id = "anthropic"
    description = "Anthropic Claude vision adapter"

    def setup(self) -> None:
        AnthropicAdapterBase.setup(self)

    def run(self, variant: Variant, sample: Sample) -> AdapterOutput:
        version = str(variant.fields.get("version"))
        if not version:
            raise ValueError("Variant.fields.version must be provided for AnthropicAdapter.")
        prompt = variant.fields.get("prompt") or DEFAULT_PROMPT

        image_b64 = self.encode_image_to_base64(sample.image_uri)
        message = self.client.messages.create(
            model=version,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )

        prediction = ""
        if message.content and isinstance(message.content, list) and message.content[0].text:
            prediction = message.content[0].text.strip()

        # pricing-only metadata
        metadata: Dict[str, object] = {}
        usage = getattr(message, "usage", None)
        if usage is not None:
            if getattr(usage, "input_tokens", None) is not None:
                metadata["input_tokens"] = usage.input_tokens
            if getattr(usage, "output_tokens", None) is not None:
                metadata["output_tokens"] = usage.output_tokens

        return AdapterOutput(prediction=prediction, metadata=metadata)


class GeminiAdapter(BaseAdapter, GeminiAdapterBase):
    id = "gemini"
    description = "Google Gemini multimodal adapter"

    def setup(self) -> None:
        GeminiAdapterBase.setup(self)

    def run(self, variant: Variant, sample: Sample) -> AdapterOutput:
        model_version = str(variant.fields.get("model_version"))
        if not model_version:
            raise ValueError("Variant.fields.model_version must be provided for GeminiAdapter.")
        prompt = variant.fields.get("prompt") or DEFAULT_PROMPT

        image = self.load_image(sample.image_uri)
        response = self.client.models.generate_content(
            model=model_version,
            contents=[prompt, image],
        )

        prediction = getattr(response, "text", "") or ""
        prediction = prediction.strip()

        # pricing-only metadata
        metadata: Dict[str, object] = {}
        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            if getattr(usage, "prompt_token_count", None) is not None:
                metadata["input_tokens"] = usage.prompt_token_count
            if getattr(usage, "candidates_token_count", None) is not None:
                metadata["output_tokens"] = usage.candidates_token_count

        return AdapterOutput(prediction=prediction, metadata=metadata)


class MistralAdapter(BaseAdapter, MistralAdapterBase):
    id = "mistral"
    description = "Mistral OCR adapter"

    def setup(self) -> None:
        MistralAdapterBase.setup(self)

    def run(self, variant: Variant, sample: Sample) -> AdapterOutput:
        version = str(variant.fields.get("version"))
        if not version:
            raise ValueError("Variant.fields.version must be provided for MistralAdapter.")

        image_b64 = self.encode_image_to_base64(sample.image_uri)
        ocr_response = self.client.ocr.process(
            model=version,
            document={"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_b64}"},
        )

        md = MarkdownIt()
        texts: List[str] = []
        for page in getattr(ocr_response, "pages", []):
            markdown = getattr(page, "markdown", "")
            tokens = md.parse(markdown)
            texts.append("".join(token.content for token in tokens).strip())
        prediction = "\n".join(filter(None, texts)).strip()

        # pricing-only metadata
        metadata: Dict[str, object] = {}
        usage_info = getattr(ocr_response, "usage_info", None)
        if usage_info is not None and getattr(usage_info, "pages_processed", None) is not None:
            metadata["pages_processed"] = usage_info.pages_processed

        return AdapterOutput(prediction=prediction, metadata=metadata)


class RoboflowAdapter(BaseAdapter, RoboflowAdapterBase):
    id = "roboflow"
    description = "Roboflow workflow adapter"

    def setup(self) -> None:
        RoboflowAdapterBase.setup(self)

    def run(self, variant: Variant, sample: Sample) -> AdapterOutput:
        version = str(variant.fields.get("version"))
        if not version:
            raise ValueError("Variant.fields.version must be provided for RoboflowAdapter.")

        with Image.open(sample.image_uri) as img:
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
        image_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        result = self.client.run_workflow(
            workspace_name=self.workspace,
            workflow_id=self.workflow,
            images={"image": image_b64},
            parameters={"model": version},
            use_cache=False,
        )

        prediction = ""
        if isinstance(result, list) and result:
            first = result[0]
            if isinstance(first, dict):
                model_block = first.get(version)
                if isinstance(model_block, dict):
                    raw_output = model_block.get("raw_output")
                    if isinstance(raw_output, str):
                        prediction = raw_output.strip().strip('"').strip()

        # pricing-only metadata (empty, as per plan)
        metadata: Dict[str, object] = {}
        return AdapterOutput(prediction=prediction, metadata=metadata)
