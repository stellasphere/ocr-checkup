# OCRCheckup (overhaul)

This package provides a simple, trustworthy, and extensible framework for evaluating scene text recognition (STR) models using a two-phase pipeline: Predict → Evaluate.

## Quickstart

1. Install dependencies (requires Python 3.10+):

```bash
pip install -r requirements.txt
```

2. Prepare dataset manifest:
- Place a dataset JSON at `datasets/10x10.json` with fields: `dataset_id`, `name`, `domains[]`, each domain with `domain_id`, `name`, and `samples[]` of `{sample_id, image_uri, ground_truth_raw}`.

3. Run the demo script:

```bash
python -m examples.demo_predict_and_eval
```

This will:
- Register example components (family, adapter, pricing)
- Construct a `Variant`
- Run predictions across the dataset into `runs/<run_id>.jsonl`
- Evaluate metrics into `evals/<evaluation_id>.jsonl`

## Concepts

- Families define fields schema and canonicalization for variants.
- Adapters execute predictions and return `{prediction, metadata}`.
- PricingModels compute `cost_usd` during evaluation from the prediction record and variant.
- Variants fully specify family, fields, adapter, and pricing; their identity is a hash of canonical fields and sanitized configs.
- Normalizer applies deterministic text normalization.
- Evaluators compute metrics (built-ins: `accuracy`, `correctness`, `cost_usd`).

## Outputs

- Predictions: JSONL with fields `run_id`, `variant_id`, `sample_id`, `prediction_raw`, `started_at`, `ended_at`, `metadata`, `error`.
- Evaluations: JSONL with fields `evaluation_id`, `run_id`, `variant_id`, `sample_id`, `gt_norm`, `pred_norm`, `metrics`.

## Legacy

Existing model implementations have been moved to `ocrcheckup/legacy/adapters/` for later adaptation to the new interfaces.
