from __future__ import annotations

import random
from pathlib import Path
from typing import List

from pydantic import BaseModel, ConfigDict

from ocrcheckup.core.ids import new_run_id
from ocrcheckup.core.jsonl import jsonl_writer
from ocrcheckup.core.time import utc_now_iso
from ocrcheckup.core.types import Dataset, Sample
from ocrcheckup.core.variant import Variant
from ocrcheckup.core.registry import adapters, pricing_models, model_families


class PredictionRun(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_id: str
    dataset_id: str
    variant: Variant
    seed: int
    shuffled: bool
    created_at: str


def run_prediction(
    dataset: Dataset,
    variant: Variant,
    *,
    seed: int,
    out_dir: Path | str | None = Path("runs"),
    out_path: Path | str | None = None,
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

    # Prepare output file at runs/<run_id>.jsonl (or custom paths)
    if out_path is not None:
        out_file = Path(out_path)
    else:
        base_dir = Path(out_dir) if out_dir is not None else Path("runs")
        out_file = base_dir / f"{run_id}.jsonl"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with jsonl_writer(out_file) as write:
        for sample in all_samples:
            started_at = utc_now_iso()
            error = None
            prediction_raw = ""
            metadata = {}
            try:
                result = adapter.run(variant, sample)
                prediction_raw = result.prediction
                metadata = result.metadata
            except Exception as e:  # errors recorded into record per TDD
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
