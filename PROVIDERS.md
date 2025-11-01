# Provider Catalog

Each provider/model combination now has its own `ModelFamily`, keeping benchmark rows aligned with the base model that users recognize (e.g. GPT-4o vs GPT-4o mini). Families expose only the variant fields that meaningfully affect focused scene text recognition—mainly prompt wording and a handful of decoding knobs. All families/adapters/pricing models below are registered by `ocrcheckup.catalog.register_default_components()`.

| Provider | Family ID | Adapter ID | Default Pricing | Key Variant Fields | Notes |
| --- | --- | --- | --- | --- | --- |
| OpenAI | `openai-gpt-4o` | `openai-vision-chat` | `token-usage` | `prompt`, `detail`, `max_output_tokens` | GPT-4o flagship multimodal model. `detail` controls image resolution (`auto`/`low`/`high`). |
| OpenAI | `openai-gpt-4o-mini` | `openai-vision-chat` | `token-usage` | `prompt`, `detail`, `max_output_tokens` | Cost-optimised GPT-4o mini. Same adapter, distinct family row. |
| OpenAI | `openai-gpt-4.5-preview` | `openai-vision-chat` | `token-usage` | `prompt`, `detail`, `max_output_tokens` | Preview GPT-4.5 multimodal model. |
| OpenAI | `openai-o1` | `openai-vision-chat` | `token-usage` | `prompt`, `detail`, `max_output_tokens` | Reasoning-focused o1 multimodal model. |
| Anthropic | `anthropic-claude-3-opus` | `anthropic-claude-messages` | `token-usage` | `prompt`, `max_output_tokens` | Claude 3 Opus high-end tier. |
| Anthropic | `anthropic-claude-3-sonnet` | `anthropic-claude-messages` | `token-usage` | `prompt`, `max_output_tokens` | Claude 3 Sonnet balanced tier. |
| Anthropic | `anthropic-claude-3-haiku` | `anthropic-claude-messages` | `token-usage` | `prompt`, `max_output_tokens` | Claude 3 Haiku fast tier. |
| Anthropic | `anthropic-claude-3.5-sonnet` | `anthropic-claude-messages` | `token-usage` | `prompt`, `max_output_tokens` | Claude 3.5 Sonnet (June 2024). |
| Anthropic | `anthropic-claude-3.5-sonnet-v2` | `anthropic-claude-messages` | `token-usage` | `prompt`, `max_output_tokens` | Latest Claude 3.5 Sonnet (Oct 2024). |
| Anthropic | `anthropic-claude-3.5-haiku` | `anthropic-claude-messages` | `token-usage` | `prompt`, `max_output_tokens` | Claude 3.5 Haiku fast tier. |
| Anthropic | `anthropic-claude-3.7-sonnet` | `anthropic-claude-messages` | `token-usage` | `prompt`, `max_output_tokens` | Claude 3.7 Sonnet (Feb 2025). |
| Google | `google-gemini-1.5-pro` | `google-gemini-vision` | `token-usage` | `prompt`, `temperature`, `max_output_tokens` | Gemini 1.5 Pro vision model. |
| Google | `google-gemini-1.5-flash` | `google-gemini-vision` | `token-usage` | `prompt`, `temperature`, `max_output_tokens` | Gemini 1.5 Flash general tier. |
| Google | `google-gemini-1.5-flash-8b` | `google-gemini-vision` | `token-usage` | `prompt`, `temperature`, `max_output_tokens` | Gemini 1.5 Flash 8B checkpoint. |
| Google | `google-gemini-2.5-pro-preview` | `google-gemini-vision` | `token-usage` | `prompt`, `temperature`, `max_output_tokens` | Gemini 2.5 Pro preview model. |
| Google | `google-gemini-2.0-flash` | `google-gemini-vision` | `token-usage` | `prompt`, `temperature`, `max_output_tokens` | Gemini 2.0 Flash model. |
| Google | `google-gemini-2.0-flash-lite` | `google-gemini-vision` | `token-usage` | `prompt`, `temperature`, `max_output_tokens` | Gemini 2.0 Flash Lite model. |
| Mistral | `mistral-ocr-2503` | `mistral-ocr-api` | `per-page` | `parse_markdown`, `max_pages` | Hosted OCR model parsing Markdown into text. |
| Alibaba | `qwen2.5-vl-7b` | `qwen-vision-transformers` | `local` | `prompt`, `max_new_tokens` | Qwen2.5-VL 7B local checkpoint. Device/dtype configured via adapter config. |
| Moondream | `moondream2` | `moondream-transformers` | `local` | `prompt`, `revision` | Open-source Moondream2 checkpoint. Device override via adapter config. |
| Hugging Face | `idefics2-8b` | `idefics2-transformers` | `local` | `prompt`, `max_new_tokens` | Idefics2 8B open-source checkpoint. dtype/device via adapter config. |
| Microsoft | `microsoft-trocr-base-printed` | `trocr-transformers` | `local` | `max_new_tokens` | Printed TrOCR encoder-decoder model. |
| EasyOCR | `easyocr` | `easyocr-reader` | `local` | `languages`, `paragraph` | EasyOCR reader; GPU toggle via adapter config. |
| Roboflow | `doctr-roboflow-hosted` | `doctr-roboflow-hosted` | `flat-per-image` | *(no variant fields)* | Hosted DocTR API; API key supplied via adapter config. |
| Roboflow | `roboflow-florence-2-large` | `roboflow-workflow` | `elapsed-seconds` | `parameters`, `use_cache` | Florence 2 Large workflow (`leo-ueno/ocr`). |
| Roboflow | `roboflow-florence-2-base` | `roboflow-workflow` | `elapsed-seconds` | `parameters`, `use_cache` | Florence 2 Base workflow (`leo-ueno/ocr`). |
| MoT | `tesseract` | `tesseract-pytesseract` | `local` | `lang`, `psm` | PyTesseract adapter retained for local baselines. |

## Usage

```python
from ocrcheckup import Variant, AdapterRef, PricingRef, register_default_components

register_default_components()

variant = Variant(
    name="gpt-4o-vision",
    family_id="openai-gpt-4o",
    fields={
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

> **Secrets:** never store API keys in `Variant.adapter.config`. The catalog’s adapters accept optional config overrides (e.g., rate limits, device selection), but credentials should come from environment variables or your own secure loader before calling `register_default_components()`.
