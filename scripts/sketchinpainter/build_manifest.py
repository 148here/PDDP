"""Build a collision-safe ArtBench/Mural1 manifest for PDDP adaptation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def stable_split(sample_id: str, validation_percent: float) -> str:
    value = int.from_bytes(hashlib.sha256(sample_id.encode("utf-8")).digest()[:8], "big") / 2**64
    return "validation" if value < validation_percent / 100.0 else "train"


def iter_images(dataset: str, root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and path.parent.name == "images":
            rel_key = path.relative_to(root).as_posix()
            yield dataset, rel_key, path.resolve()


def build_rows(dataset: str, root: Path, output_root: Path, validation_percent: float):
    for name, rel_key, image_path in iter_images(dataset, root):
        sample_id = f"{name}:{rel_key}"
        safe_key = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
        yield {
            "schema_version": 1,
            "sample_id": sample_id,
            "dataset": name,
            "rel_key": rel_key,
            "split": stable_split(sample_id, validation_percent),
            "image_path": str(image_path),
            "edge_path": str((output_root / "muge_edges" / name / f"{safe_key}.png").resolve()),
            "token_path": str((output_root / "vq_tokens" / name / f"{safe_key}.npy").resolve()),
        }


def training_root(root: Path) -> Path:
    candidate = root / "train"
    return candidate if candidate.is_dir() else root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artbench-root", type=Path, required=True)
    parser.add_argument("--mural1-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validation-percent", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 0 < args.validation_percent < 50:
        raise ValueError("validation-percent must be in (0, 50)")
    roots = {"artbench": args.artbench_root.resolve(), "mural1": args.mural1_root.resolve()}
    for name, root in roots.items():
        if not root.is_dir():
            raise FileNotFoundError(f"Missing {name} root: {root}")
    scan_roots = {name: training_root(root) for name, root in roots.items()}
    rows = [
        *build_rows("artbench", scan_roots["artbench"], args.output_root, args.validation_percent),
        *build_rows("mural1", scan_roots["mural1"], args.output_root, args.validation_percent),
    ]
    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sample_id values detected")
    summary = {
        "total": len(rows),
        "datasets": {name: sum(row["dataset"] == name for row in rows) for name in roots},
        "scan_roots": {name: str(path) for name, path in scan_roots.items()},
        "splits": {split: sum(row["split"] == split for row in rows) for split in ("train", "validation")},
        "manifest": str(args.manifest.resolve()),
        "written": not args.dry_run,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.dry_run:
        return
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
