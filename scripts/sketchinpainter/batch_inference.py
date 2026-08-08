"""Canonical 702-condition PDDP inference with resume and audit metadata.

Without --execute this command only validates and reports the canonical plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from image_synthesis.data.sketchinpainter_dataset import crop_and_pad_sketch, sha256_file


def load_conditions(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    ids = [str(row["condition_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate canonical condition_id values")
    return rows


def asset_path(row: dict[str, Any], key: str) -> Path:
    asset = row.get("assets", {}).get(key)
    if not asset:
        raise KeyError(f"{row['condition_id']} has no asset {key!r}")
    path = Path(asset.get("snapshot_path") or asset.get("source_path"))
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = asset.get("sha256")
    if expected and sha256_file(path) != expected:
        raise ValueError(f"SHA-256 mismatch for {path}")
    return path


def safe_id(condition_id: str) -> str:
    return hashlib.sha256(condition_id.encode("utf-8")).hexdigest()[:24]


def validate(rows: list[dict[str, Any]], output_root: Path, verify_hashes: bool) -> dict[str, Any]:
    datasets: dict[str, int] = {}
    rounds: dict[str, int] = {}
    missing: list[str] = []
    for row in rows:
        datasets[row["dataset_name"]] = datasets.get(row["dataset_name"], 0) + 1
        round_key = f"round_{int(row['round_index']):03d}"
        rounds[round_key] = rounds.get(round_key, 0) + 1
        try:
            for key in ("gt_rgb", "hole_mask", "base_sketch"):
                if verify_hashes:
                    asset_path(row, key)
                else:
                    asset = row.get("assets", {}).get(key, {})
                    path = Path(asset.get("snapshot_path") or asset.get("source_path") or "")
                    if not path.is_file():
                        raise FileNotFoundError(path)
        except (KeyError, FileNotFoundError, ValueError) as error:
            missing.append(f"{row['condition_id']}: {error}")
    return {
        "conditions": len(rows),
        "unique_sources": len({row["source_id"] for row in rows}),
        "datasets": datasets,
        "rounds": rounds,
        "missing_or_invalid": len(missing),
        "first_errors": missing[:20],
        "output_root": str(output_root.resolve()),
    }


def load_model(config_path: Path, checkpoint: Path, device: str):
    import torch
    from image_synthesis.modeling.build import build_model
    from image_synthesis.utils.io import load_yaml_config

    config = load_yaml_config(str(config_path))
    model = build_model(config)
    state = torch.load(checkpoint, map_location="cpu")
    model_state = state.get("model", state)
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    print(json.dumps({"missing_keys": missing, "unexpected_keys": unexpected}, indent=2))
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, torch


def run_one(model, torch, row: dict[str, Any], output_root: Path, device: str, truncation: float) -> None:
    condition_id = str(row["condition_id"])
    item_id = safe_id(condition_id)
    item_dir = output_root / row["dataset_name"] / f"round_{int(row['round_index']):03d}" / item_id
    generated_path = item_dir / "generated.png"
    metadata_path = item_dir / "metadata.json"
    if generated_path.is_file() and metadata_path.is_file():
        return
    gt_path, mask_path, sketch_path = (asset_path(row, key) for key in ("gt_rgb", "hole_mask", "base_sketch"))
    gt_image = Image.open(gt_path).convert("RGB")
    gt = np.asarray(gt_image, dtype=np.uint8)
    mask_native = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8) >= 128
    mask_256 = cv2.resize(mask_native.astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST)
    sketch = cv2.imread(str(sketch_path), cv2.IMREAD_GRAYSCALE)
    if sketch is None:
        raise ValueError(f"Failed to read {sketch_path}")
    sketch = cv2.resize(sketch, (256, 256), interpolation=cv2.INTER_LINEAR)
    pddp_sketch = crop_and_pad_sketch(sketch, mask_256, output_size=224, bbox_scale=1.2)
    image_256 = np.asarray(gt_image.resize((256, 256), Image.Resampling.BICUBIC), dtype=np.float32).copy()
    image_tensor = torch.from_numpy(image_256).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        tokens = model.content_codec.vq.get_tokens(image_tensor)["token"].reshape(1, -1)
    batch = {
        "quantized_image": tokens,
        "obj_mask": torch.from_numpy(mask_256.astype(np.float32)).unsqueeze(0).to(device),
        "sketch": torch.from_numpy(pddp_sketch.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(device),
    }
    seed = int(row.get("seed", 0)) % (2**31 - 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    with torch.no_grad():
        result = model.generate_content(batch=batch, filter_ratio=0, replicate=1, content_ratio=1,
                                        return_att_weight=False, sample_type=f"top{truncation}r")
    generated_256 = result["content"][0].permute(1, 2, 0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
    generated_native = np.asarray(Image.fromarray(generated_256).resize(gt_image.size, Image.Resampling.BICUBIC))
    composite = gt.copy()
    composite[mask_native] = generated_native[mask_native]
    item_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(composite).save(generated_path)
    Image.fromarray(pddp_sketch).save(item_dir / "pddp_sketch.png")
    Image.fromarray((mask_256 * 255).astype(np.uint8)).save(item_dir / "pddp_hole_mask.png")
    metadata = {
        "schema_version": 1,
        "condition_id": condition_id,
        "source_id": row["source_id"],
        "dataset_name": row["dataset_name"],
        "round_index": row["round_index"],
        "seed": seed,
        "prompt_used": False,
        "prompt_note": "PDDP has no text-prompt input; canonical prompt is intentionally ignored.",
        "canonical_shared_condition_hash": row.get("shared_condition_hash"),
        "inputs": {key: {"path": str(path), "sha256": sha256_file(path)} for key, path in
                   (("gt_rgb", gt_path), ("hole_mask", mask_path), ("base_sketch", sketch_path))},
        "generated_path": str(generated_path),
        "generated_sha256": sha256_file(generated_path),
        "outside_hole_exact_gt": True,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-conditions", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/sketchinpainter_finetune.yaml"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--truncation", type=float, default=0.85)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--verify-hashes", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    rows = load_conditions(args.canonical_conditions)
    report = validate(rows, args.output_root, args.verify_hashes)
    report["execute"] = args.execute
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["missing_or_invalid"]:
        raise SystemExit(2)
    if not args.execute:
        return
    if args.checkpoint is None or not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    model, torch = load_model(args.config, args.checkpoint, args.device)
    selected = rows[:args.limit] if args.limit else rows
    for index, row in enumerate(selected, 1):
        run_one(model, torch, row, args.output_root, args.device, args.truncation)
        if index % 10 == 0:
            print(f"{index}/{len(selected)}")


if __name__ == "__main__":
    main()
