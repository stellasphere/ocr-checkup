from __future__ import annotations

from typing import Optional, Dict, Any

from ocrcheckup.adapters.base import Adapter, AdapterOutput
from ocrcheckup.core.types import Sample
from ocrcheckup.core.variant import Variant


class MockAdapter:
    id = "mock"
    description = "Returns ground truth as prediction (for smoke tests)"

    def run(self, variant: Variant, sample: Sample) -> AdapterOutput:
        return AdapterOutput(
            prediction=sample.ground_truth_raw,
            metadata={"note": "echo_gt"},
        )

    def hash_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        return {}


adapter = MockAdapter()
