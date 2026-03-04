from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import List

from pydantic import BaseModel, ConfigDict

from ocrcheckup.core.ids import new_run_id
from ocrcheckup.core.jsonl import jsonl_writer
from ocrcheckup.core.time import utc_now_iso
from ocrcheckup.core.types import Dataset, Sample
from ocrcheckup.core.variant import Variant
from ocrcheckup.core.registry import adapters, pricing_models, model_families
from ocrcheckup.rate_limiter import RateLimiter


class PredictionRun(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_id: str
    dataset_id: str
    variant: Variant
    seed: int
    shuffled: bool
    created_at: str


def _run_with_retry(adapter, variant, sample, *, max_retries: int, retry_backoff_s: float):
    """Run adapter with retry and exponential backoff for transient failures."""
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return adapter.run(variant, sample)
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                wait = retry_backoff_s * (2 ** attempt)
                time.sleep(wait)
    raise last_error


def run_prediction(
    dataset: Dataset,
    variant: Variant,
    *,
    seed: int,
    out_dir: Path | str | None = Path("results") / "runs",
    out_path: Path | str | None = None,
    max_retries: int = 2,
    retry_backoff_s: float = 1.0,
    requests_per_minute: int | None = None,
) -> str:
    run_id = new_run_id()
    pr = PredictionRun(
        run_id=run_id,
        dataset_id=dataset.dataset_id,
        variant=variant,
        seed=seed,
        shuffled=True,
        created_at=utc_now_iso(),
    )

    # Validate and canonicalize fields; compute variant_id
    family = model_families.get(variant.family_id)
    canonical_fields = family.validate_fields(variant.fields)
    adapter = adapters.get(variant.adapter.id)
    pricing = pricing_models.get(variant.pricing.id)
    adapter_hash_cfg = adapter.hash_config(variant.adapter.config)
    pricing_hash_cfg = pricing.hash_config(variant.pricing.config)
    variant.compute_variant_id(
        family_schema_version=family.family_schema_version,
        canonical_fields=canonical_fields,
        adapter_hash_cfg=adapter_hash_cfg,
        pricing_hash_cfg=pricing_hash_cfg,
    )

    # Build sample list and shuffle deterministically
    all_samples: List[Sample] = []
    for d in dataset.domains:
        all_samples.extend(d.samples)
    rng = random.Random(seed)
    rng.shuffle(all_samples)

    # Prepare output file at results/runs/<run_id>.jsonl (or custom paths)
    if out_path is not None:
        out_file = Path(out_path)
    else:
        base_dir = Path(out_dir) if out_dir is not None else (Path("results") / "runs")
        out_file = base_dir / f"{run_id}.jsonl"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Write run metadata for re-evaluation support
    meta_file = out_file.with_suffix(".meta.json")
    meta_payload = {
        "run_id": pr.run_id,
        "dataset_id": pr.dataset_id,
        "variant": variant.model_dump(mode="json"),
        "variant_id": variant.variant_id,
        "seed": pr.seed,
        "shuffled": pr.shuffled,
        "created_at": pr.created_at,
    }
    meta_file.write_text(
        json.dumps(meta_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Optional rate limiter for cloud APIs
    limiter = RateLimiter(requests_per_minute) if requests_per_minute else None

    setup = getattr(adapter, "setup", None)
    if callable(setup):
        setup()

    with jsonl_writer(out_file) as write:
        for sample in all_samples:
            if limiter:
                limiter.wait_if_needed()

            started_at = utc_now_iso()
            error = None
            prediction_raw = ""
            metadata = {}
            try:
                result = _run_with_retry(
                    adapter, variant, sample,
                    max_retries=max_retries,
                    retry_backoff_s=retry_backoff_s,
                )
                prediction_raw = result.prediction
                metadata = result.metadata
            except Exception as e:
                error = str(e)
            ended_at = utc_now_iso()

            record = {
                "run_id": pr.run_id,
                "variant_id": variant.variant_id,
                "sample_id": sample.sample_id,
                "prediction_raw": prediction_raw,
                "started_at": started_at,
                "ended_at": ended_at,
                "metadata": metadata,
                "error": error,
            }
            write(record)

    return pr.run_id


def load_run_metadata(
    run_id: str, runs_dir: Path | str = Path("results") / "runs"
) -> dict:
    """Load the metadata JSON for a prediction run."""
    meta_path = Path(runs_dir) / f"{run_id}.meta.json"
    return json.loads(meta_path.read_text(encoding="utf-8"))


def find_run_by_variant_id(
    variant_id: str, runs_dir: Path | str = Path("results") / "runs"
) -> str | None:
    """Find an existing run_id for a given variant_id. Returns None if not found."""
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return None

    # Check meta files first (fast)
    for meta_file in sorted(runs_dir.glob("*.meta.json"), reverse=True):
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            if meta.get("variant_id") == variant_id:
                return meta["run_id"]
        except (json.JSONDecodeError, KeyError):
            continue

    # Fallback: scan JSONL first lines (for runs without meta files)
    from ocrcheckup.core.jsonl import iterate_jsonl

    for jsonl_file in sorted(runs_dir.glob("*.jsonl"), reverse=True):
        try:
            for rec in iterate_jsonl(jsonl_file):
                if rec.get("variant_id") == variant_id:
                    return rec["run_id"]
                break  # only check first record
        except Exception:
            continue

    return None
