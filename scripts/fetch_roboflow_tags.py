"""
Fetch image tags from a Roboflow project and save a domain mapping.

Uses the Roboflow Python SDK to query images by tag,
producing a JSON file mapping sample_id -> domain (tag name).

Usage:
    python scripts/fetch_roboflow_tags.py \
        --workspace YOUR_WORKSPACE \
        --project YOUR_PROJECT \
        --tags "signage,handwriting,receipts,screens,documents" \
        --out datasets/domain_map.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch image tags from Roboflow.")
    parser.add_argument("--workspace", required=True, help="Roboflow workspace name.")
    parser.add_argument("--project", required=True, help="Roboflow project name.")
    parser.add_argument(
        "--tags",
        required=True,
        help="Comma-separated list of tags to fetch (these become domain IDs).",
    )
    parser.add_argument(
        "--out",
        default="datasets/domain_map.json",
        help="Output path for domain map JSON.",
    )
    args = parser.parse_args()

    from roboflow import Roboflow

    rf = Roboflow()  # uses ROBOFLOW_API_KEY from env
    project = rf.workspace(args.workspace).project(args.project)

    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    domain_map: dict[str, str] = {}

    for tag in tags:
        print(f"Fetching images with tag: {tag}")
        for batch in project.search_all(tag=tag, fields=["id", "name"]):
            for image_data in batch:
                # Use filename stem as sample_id (matches manifest builder)
                name = image_data.get("name", "")
                sample_id = Path(name).stem
                if sample_id:
                    domain_map[sample_id] = tag

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(domain_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Domain map written to {out_path} ({len(domain_map)} images mapped)")


if __name__ == "__main__":
    main()
