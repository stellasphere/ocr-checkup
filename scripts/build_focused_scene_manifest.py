from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    test_dir = repo_root / "datasets" / "raw" / "Focused Scene" / "test"
    ann_path = test_dir / "annotations.jsonl"

    samples = []
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
        samples.append(
            {
                "sample_id": sample_id,
                "image_uri": str(img_path.resolve()),
                "ground_truth_raw": gt,
            }
        )

    dataset = {
        "dataset_id": "focused-scene",
        "name": "Focused Scene",
        "domains": [
            {
                "domain_id": "focused-scene-test",
                "name": "Focused Scene Test",
                "samples": samples,
            }
        ],
    }

    out_path = repo_root / "datasets" / "focused_scene.json"
    out_path.write_text(
        json.dumps(dataset, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    print(str(out_path))


if __name__ == "__main__":
    main()


