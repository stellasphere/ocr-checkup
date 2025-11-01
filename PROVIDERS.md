# Provider Catalog

The retrofit keeps all previously supported adapters but exposes them through the new `ModelFamily → Variant → Adapter` ontology. The table below summarizes the built-in catalog that ships with `ocrcheckup.catalog.register_default_components()`.

| Provider | Family ID | Adapter ID | Default Pricing Model | Key Fields | Notes |
| --- | --- | --- | --- | --- | --- |
| OpenAI Vision | `openai-vision` | `openai-vision-chat` | `token-usage` (configure Azure style rates) | `model`, `prompt`, `detail`, `max_output_tokens`, `temperature`, `top_p` | Uses new OpenAI Python client (`chat.completions`). Reads `OPENAI_API_KEY` (or configured client env). Metadata includes token usage for billing. |
| Anthropic Claude | `anthropic-claude` | `anthropic-claude-messages` | `token-usage` | `model`, `prompt`, `max_output_tokens`, `temperature`, `top_k`, `top_p` | Requires `ANTHROPIC_API_KEY`. Uploads base64 JPEG. Metadata includes Claude usage counters. |
| Google Gemini | `google-gemini` | `google-gemini-vision` | `token-usage` | `model`, `prompt`, `temperature`, `top_p`, `top_k`, `max_output_tokens` | Requires `GOOGLE_API_KEY` via `google-genai` client (or `GEMINI_API_KEY`). Response metadata carries prompt/output tokens. |
| Mistral OCR | `mistral-ocr` | `mistral-ocr-api` | `per-page` | `model`, `parse_markdown`, `max_pages` | Requires `MISTRAL_API_KEY`. By default strips Markdown to text. Metadata tracks `pages_processed`. |
| Roboflow Workflow | `roboflow-workflow` | `roboflow-workflow` | `elapsed-seconds` or `flat-per-image` | `workspace`, `workflow`, `model`, `parameters`, `use_cache` | Requires `ROBOFLOW_API_KEY`. Metadata records elapsed seconds. |
| DocTR Hosted | `doctr-roboflow` | `doctr-roboflow-hosted` | `flat-per-image` | `api_url` | Also depends on `ROBOFLOW_API_KEY`; returns the raw transcription string. |
| Qwen 2.5 VL | `qwen-vision` | `qwen-vision-transformers` | `local` (or `flat-per-image` if desired) | `model`, `prompt`, `max_new_tokens`, `device`, `dtype` | Loads HF checkpoint with optional device override. Requires `qwen-vl-utils`. |
| Moondream | `moondream` | `moondream-transformers` | `local` | `model`, `revision`, `prompt`, `device` | Auto-selects CUDA/MPS/CPU when not specified. |
| Idefics2 | `idefics2` | `idefics2-transformers` | `local` | `model`, `prompt`, `torch_dtype`, `device`, `max_new_tokens` | Loads HF checkpoint with configurable dtype. |
| TrOCR | `trocr` | `trocr-transformers` | `local` | `model`, `device`, `max_new_tokens` | Works with printed TrOCR base by default. |
| EasyOCR | `easyocr` | `easyocr-reader` | `local` | `languages`, `gpu`, `detail`, `paragraph` | Downloads EasyOCR language packs on first use. GPU usage auto-detected unless forced. |
| Tesseract | `tesseract` | `tesseract-pytesseract` | `local` | `lang`, `psm` | Registered by catalog for completeness. |

## Usage

```python
from ocrcheckup import Variant, AdapterRef, PricingRef, register_default_components
from ocrcheckup.catalog import DEFAULT_FAMILIES, DEFAULT_ADAPTERS

register_default_components()

variant = Variant(
    name="gpt-4o-vision",
    family_id="openai-vision",
    fields={
        "model": "gpt-4o-2024-05-13",
        "prompt": "Read the text in the image. Return only the text exactly as it appears.",
        "detail": "high",
    },
    adapter=AdapterRef(id="openai-vision-chat", config={"rpm": 300}),
    pricing=PricingRef(
        id="token-usage",
        config={
            "input_per_million": 5.0,
            "output_per_million": 15.0,
            "per_request": 0.0,
            "round": 6,
        },
    ),
)
```

All adapters return metadata tailored for their pricing model (token counts, page counts, elapsed time, etc.). You can swap in `flat-per-image` or custom pricing models as desired.

> **Secrets:** never store API keys in `Variant.adapter.config`. The catalog’s adapters accept optional config overrides (e.g., rate limits), but credentials should come from environment variables or your own secure loader before calling `register_default_components()`.
