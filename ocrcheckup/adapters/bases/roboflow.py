from __future__ import annotations

import os
from typing import Optional

from inference_sdk import InferenceHTTPClient


class RoboflowAdapterBase:
    """Shared Roboflow adapter utilities: client setup."""

    def __init__(self) -> None:
        self._client: Optional[InferenceHTTPClient] = None
        self._workspace_name: Optional[str] = None
        self._workflow_id: Optional[str] = None

    def setup(self) -> None:
        if self._client is not None:
            return
        api_key = os.environ.get("ROBOFLOW_API_KEY")
        if not api_key:
            raise ValueError("ROBOFLOW_API_KEY environment variable not set.")
        api_url = os.environ.get("ROBOFLOW_API_URL", "https://serverless.roboflow.com")
        workspace = os.environ.get("ROBOFLOW_WORKSPACE")
        workflow = os.environ.get("ROBOFLOW_WORKFLOW")
        if not workspace or not workflow:
            raise ValueError("ROBOFLOW_WORKSPACE and ROBOFLOW_WORKFLOW environment variables must be set.")
        self._client = InferenceHTTPClient(api_url=api_url, api_key=api_key)
        self._workspace_name = workspace
        self._workflow_id = workflow

    @property
    def client(self) -> InferenceHTTPClient:
        if self._client is None:
            raise RuntimeError("Roboflow client is not initialized. Call setup() first.")
        return self._client

    @property
    def workspace(self) -> str:
        if self._workspace_name is None:
            raise RuntimeError("Roboflow workspace is not initialized. Call setup() first.")
        return self._workspace_name

    @property
    def workflow(self) -> str:
        if self._workflow_id is None:
            raise RuntimeError("Roboflow workflow is not initialized. Call setup() first.")
        return self._workflow_id

