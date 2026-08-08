"""Precompute MuGE edges. Default mode is a read-only plan; --execute runs MuGE."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def rows(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sketchinpainter-root", type=Path, required=True)
    parser.add_argument("--muge-source-root", type=Path, required=True)
    parser.add_argument("--muge-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--execute", action="store_true", help="actually load MuGE and write edges")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    manifest_rows = list(rows(args.manifest))
    missing = [row for row in manifest_rows if not Path(row["edge_path"]).is_file()]
    print(json.dumps({"conditions": len(manifest_rows), "missing_edges": len(missing), "execute": args.execute}, indent=2))
    if not args.execute:
        return
    if not args.muge_checkpoint.is_file():
        raise FileNotFoundError(args.muge_checkpoint)
    sys.path.insert(0, str(args.sketchinpainter_root.resolve()))
    from dataset.makeedge.muge import get_muge_extractor

    extractor = get_muge_extractor(
        source_root=str(args.muge_source_root), checkpoint_path=str(args.muge_checkpoint), device=args.device
    )
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    pending = [row for row in manifest_rows if not (args.resume and Path(row["edge_path"]).is_file())]
    completed = len(manifest_rows) - len(pending)
    for start in range(0, len(pending), args.batch_size):
        chunk = pending[start:start + args.batch_size]
        grouped: dict[tuple[int, ...], list[tuple[dict, np.ndarray]]] = {}
        for row in chunk:
            image = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Failed to read {row['image_path']}")
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            grouped.setdefault(tuple(rgb.shape), []).append((row, rgb))
        for values in grouped.values():
            batch = np.stack([rgb for _, rgb in values], axis=0)
            edges = extractor.extract_batch(
                batch, alpha=1.0, inference_seed=42, line_polarity="black_on_white"
            )
            for (row, _), edge in zip(values, edges):
                output = Path(row["edge_path"])
                output.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(output), edge):
                    raise OSError(f"Failed to write {output}")
        completed += len(chunk)
        if completed % 100 < len(chunk) or completed == len(manifest_rows):
            print(f"{completed}/{len(manifest_rows)}", flush=True)


if __name__ == "__main__":
    main()
