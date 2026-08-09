"""Validate every manifest cache row before fine-tuning."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from image_synthesis.data.sketchinpainter_dataset import load_jsonl, load_vq_tokens


def validate_manifest_caches(path: Path) -> dict[str, Any]:
    rows = load_jsonl(path)
    errors: list[str] = []
    datasets: dict[str, int] = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        datasets[str(row["dataset"])] = datasets.get(str(row["dataset"]), 0) + 1
        edge_path = Path(row["edge_path"])
        token_path = Path(row["token_path"])
        edge = cv2.imread(str(edge_path), cv2.IMREAD_GRAYSCALE) if edge_path.is_file() else None
        if edge is None or edge.size == 0:
            errors.append(f"{sample_id}: invalid edge cache: {edge_path}")
        try:
            load_vq_tokens(token_path)
        except (OSError, ValueError) as error:
            errors.append(f"{sample_id}: {error}")
    return {
        "manifest": str(path.resolve()),
        "rows": len(rows),
        "datasets": datasets,
        "errors": errors,
        "valid": not errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    report = validate_manifest_caches(args.manifest)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
