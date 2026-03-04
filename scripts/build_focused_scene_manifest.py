from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    test_dir = repo_root / "datasets" / "raw" / "Focused Scene" / "test"
    ann_path = test_dir / "annotations.jsonl"

    # Load domain map if available (sample_id -> domain tag)
    domain_map_path = repo_root / "datasets" / "domain_map.json"
    domain_map: dict[str, str] = {}
    if domain_map_path.exists():
        domain_map = json.loads(domain_map_path.read_text(encoding="utf-8"))
        print(f"Loaded domain map with {len(domain_map)} entries")

    # Parse annotations into samples grouped by domain
    samples_by_domain: dict[str, list[dict]] = defaultdict(list)

    for line in ann_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        image_name = obj["image"]
        # prefer direct file in test/; fallback to images/ if exists
        img_path = test_dir / image_name
        if not img_path.exists():
            alt = test_dir / "images" / image_name
            if alt.exists():
                img_path = alt
            else:
                raise FileNotFoundError(f"Image not found for {image_name}")
        sample_id = Path(image_name).stem
        gt = obj.get("suffix", "")

        sample = {
            "sample_id": sample_id,
            "image_uri": str(img_path.resolve()),
            "ground_truth_raw": gt,
        }

        # Assign to domain from map, or fallback to a single default domain
        domain_id = domain_map.get(sample_id, "focused-scene-test")
        samples_by_domain[domain_id].append(sample)

    # Build domain objects
    domains = []
    for domain_id in sorted(samples_by_domain.keys()):
        domains.append(
            {
                "domain_id": domain_id,
                "name": domain_id.replace("-", " ").title(),
                "samples": samples_by_domain[domain_id],
            }
        )

    dataset = {
        "dataset_id": "focused-scene",
        "name": "Focused Scene",
        "domains": domains,
    }

    out_path = repo_root / "datasets" / "focused_scene.json"
    out_path.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    total_samples = sum(len(d["samples"]) for d in domains)
    print(f"Written {out_path} ({len(domains)} domains, {total_samples} samples)")


if __name__ == "__main__":
    main()
