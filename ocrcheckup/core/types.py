from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

from pydantic import BaseModel


class Sample(BaseModel):
    sample_id: str
    image_uri: str
    ground_truth_raw: str


class Domain(BaseModel):
    domain_id: str
    name: str
    description: Optional[str] = None
    samples: List[Sample]


class Dataset(BaseModel):
    dataset_id: str
    name: str
    description: Optional[str] = None
    domains: List[Domain]


def build_sample_index(dataset: Dataset) -> Dict[str, Tuple[str, Sample]]:
    sample_index: Dict[str, Tuple[str, Sample]] = {}
    for domain in dataset.domains:
        for sample in domain.samples:
            sample_index[sample.sample_id] = (domain.domain_id, sample)
    return sample_index


def load_dataset(path: str | Path) -> Dataset:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    return Dataset.model_validate(data)


__all__ = [
    "Sample",
    "Domain",
    "Dataset",
    "build_sample_index",
    "load_dataset",
]
