"""Canonical 702-condition PDDP inference with resume and audit metadata.

Without --execute this command only validates and reports the canonical plan.
"""

from __future__ import annotations

import argparse
import gc
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

from image_synthesis.data.sketchinpainter_dataset import prepare_pddp_sketch, sha256_file


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


def read_condition_ids(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate condition IDs in {path}")
    return values


def core_key_mismatches(keys: list[str]) -> list[str]:
    """Ignore only VQ parameters supplied by the separately audited VQ checkpoint."""
    return [key for key in keys if not key.startswith("content_codec.vq.")]


def raw_image_stats(value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value, dtype=np.float32)
    finite = bool(np.isfinite(array).all())
    safe = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
    clipped = np.clip(safe, 0, 255).astype(np.uint8)
    near_black = float((clipped <= 5).mean())
    near_white = float((clipped >= 250).mean())
    std = float(safe.std())
    return {
        "finite": finite,
        "min": float(safe.min()),
        "max": float(safe.max()),
        "mean": float(safe.mean()),
        "std": std,
        "near_black_ratio": near_black,
        "near_white_ratio": near_white,
        "near_constant": std < 2.0,
        "near_all_black": near_black > 0.98,
        "near_all_white": near_white > 0.98,
    }


def json_scalar(value: Any) -> Any:
    """Convert scalar checkpoint metadata to a JSON-safe value."""
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (RuntimeError, ValueError):
            pass
    return value


def apply_checkpoint_weights(model: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Load base modules then overwrite the transformer with its EMA weights."""
    model_state = state.get("model", state)
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    critical_missing = core_key_mismatches(list(missing))
    critical_unexpected = core_key_mismatches(list(unexpected))
    ema_missing: list[str] = []
    ema_unexpected: list[str] = []
    ema_applied = False
    if "ema" in state:
        ema_missing, ema_unexpected = model.get_ema_model().load_state_dict(state["ema"], strict=False)
        ema_applied = True
    result = {
        "model_missing_keys": list(missing),
        "model_unexpected_keys": list(unexpected),
        "critical_model_missing_keys": critical_missing,
        "critical_model_unexpected_keys": critical_unexpected,
        "ema_present": "ema" in state,
        "ema_applied": ema_applied,
        "ema_missing_keys": list(ema_missing),
        "ema_unexpected_keys": list(ema_unexpected),
    }
    if critical_missing or critical_unexpected:
        raise RuntimeError("PDDP checkpoint has critical model key mismatches")
    if not ema_applied or ema_missing or ema_unexpected:
        raise RuntimeError("PDDP EMA weights are absent or incompatible")
    return result


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
    vq_checkpoint = Path(config["model"]["params"]["content_codec_config"]["params"]["ckpt_path"])
    if not vq_checkpoint.is_file():
        raise FileNotFoundError(vq_checkpoint)
    state = torch.load(checkpoint, map_location="cpu")
    key_audit = apply_checkpoint_weights(model, state)
    audit = {
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_top_level_keys": sorted(str(key) for key in state.keys()),
        "last_epoch": json_scalar(state.get("last_epoch", state.get("epoch"))),
        "last_iter": json_scalar(state.get("last_iter")),
        "vq_checkpoint_path": str(vq_checkpoint.resolve()),
        "vq_checkpoint_sha256": sha256_file(vq_checkpoint),
        **key_audit,
    }
    print(json.dumps(audit, indent=2))
    del state
    gc.collect()
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, torch, audit


def run_one(
    model,
    torch,
    row: dict[str, Any],
    output_root: Path,
    device: str,
    truncation: float,
    *,
    sketch_scope: str = "bbox_crop",
    bbox_scale: float = 1.2,
) -> None:
    condition_id = str(row["condition_id"])
    item_id = safe_id(condition_id)
    item_dir = output_root / row["dataset_name"] / f"round_{int(row['round_index']):03d}" / item_id
    generated_path = item_dir / "generated.png"
    raw_path = item_dir / "raw_256.png"
    metadata_path = item_dir / "metadata.json"
    if generated_path.is_file() and raw_path.is_file() and metadata_path.is_file():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        preprocessing = existing.get("sketch_preprocessing", {})
        if preprocessing.get("scope") != sketch_scope or float(preprocessing.get("bbox_scale", -1)) != float(bbox_scale):
            raise ValueError(
                f"Existing result uses incompatible sketch preprocessing for {condition_id}: {preprocessing}"
            )
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
    pddp_sketch, sketch_preprocessing = prepare_pddp_sketch(
        sketch,
        mask_256,
        scope=sketch_scope,
        output_size=224,
        bbox_scale=bbox_scale,
    )
    image_256 = np.asarray(gt_image.resize((256, 256), Image.Resampling.BICUBIC), dtype=np.float32).copy()
    image_tensor = torch.from_numpy(image_256).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        tokens = model.content_codec.vq.get_tokens(image_tensor)["token"].reshape(1, -1)
    batch = {
        "quantized_image": tokens,
        "obj_mask": torch.from_numpy(mask_256.astype(np.float32)).unsqueeze(0).to(device),
        "sketch": torch.from_numpy(pddp_sketch.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(device),
    }
    comparison_seed = int(row.get("comparison_seed", row.get("seed", 0)))
    effective_seed = comparison_seed % (2**31 - 1)
    random.seed(effective_seed)
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective_seed)
    with torch.no_grad():
        result = model.generate_content(batch=batch, filter_ratio=0, replicate=1, content_ratio=1,
                                        return_att_weight=False, sample_type=f"top{truncation}r")
    generated_float = result["content"][0].permute(1, 2, 0).detach().float().cpu().numpy()
    stats = raw_image_stats(generated_float)
    if not stats["finite"]:
        raise ValueError(f"Non-finite pretrained output for {condition_id}")
    generated_256 = generated_float.clip(0, 255).astype(np.uint8)
    generated_native = np.asarray(Image.fromarray(generated_256).resize(gt_image.size, Image.Resampling.BICUBIC))
    composite = gt.copy()
    composite[mask_native] = generated_native[mask_native]
    if not np.array_equal(composite[~mask_native], gt[~mask_native]):
        raise AssertionError(f"Outside-hole pixels changed for {condition_id}")
    item_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(composite).save(generated_path)
    Image.fromarray(generated_256).save(raw_path)
    sketch_output_path = item_dir / "pddp_sketch.png"
    mask_output_path = item_dir / "pddp_hole_mask.png"
    Image.fromarray(pddp_sketch).save(sketch_output_path)
    Image.fromarray((mask_256 * 255).astype(np.uint8)).save(mask_output_path)
    metadata = {
        "schema_version": 1,
        "condition_id": condition_id,
        "source_id": row["source_id"],
        "dataset_name": row["dataset_name"],
        "round_index": row["round_index"],
        "seed": effective_seed,
        "comparison_seed": comparison_seed,
        "effective_seed": effective_seed,
        "seed_note": "PDDP/NumPy require a 31-bit seed; effective_seed = comparison_seed mod (2^31-1).",
        "prompt_used": False,
        "prompt_note": "PDDP has no text-prompt input; canonical prompt is intentionally ignored.",
        "sketch_preprocessing": sketch_preprocessing,
        "canonical_shared_condition_hash": row.get("shared_condition_hash"),
        "inputs": {key: {"path": str(path), "sha256": sha256_file(path)} for key, path in
                   (("gt_rgb", gt_path), ("hole_mask", mask_path), ("base_sketch", sketch_path))},
        "generated_path": str(generated_path),
        "generated_sha256": sha256_file(generated_path),
        "outside_hole_exact_gt": True,
        "raw_stats": stats,
        "outputs": {
            "raw_256": {"path": str(raw_path), "sha256": sha256_file(raw_path)},
            "composite": {"path": str(generated_path), "sha256": sha256_file(generated_path)},
            "pddp_sketch": {"path": str(sketch_output_path), "sha256": sha256_file(sketch_output_path)},
            "pddp_hole_mask": {"path": str(mask_output_path), "sha256": sha256_file(mask_output_path)},
        },
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
    parser.add_argument("--sketch-scope", choices=("bbox_crop", "full", "hole_crop"), default="bbox_crop")
    parser.add_argument("--bbox-scale", type=float, default=1.2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--condition-id-file", type=Path)
    parser.add_argument("--verify-hashes", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.bbox_scale <= 0:
        raise ValueError("--bbox-scale must be positive")
    rows = load_conditions(args.canonical_conditions)
    if args.condition_id_file:
        requested = read_condition_ids(args.condition_id_file)
        by_id = {str(row["condition_id"]): row for row in rows}
        missing_ids = [condition_id for condition_id in requested if condition_id not in by_id]
        if missing_ids:
            raise KeyError(f"Unknown canonical condition IDs: {missing_ids}")
        rows = [by_id[condition_id] for condition_id in requested]
    report = validate(rows, args.output_root, args.verify_hashes)
    report["execute"] = args.execute
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["missing_or_invalid"]:
        raise SystemExit(2)
    if not args.execute:
        return
    if args.checkpoint is None or not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    args.output_root.mkdir(parents=True, exist_ok=True)
    model, torch, checkpoint_audit = load_model(args.config, args.checkpoint, args.device)
    (args.output_root / "checkpoint_audit.json").write_text(
        json.dumps(checkpoint_audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    selected = rows[:args.limit] if args.limit else rows
    for index, row in enumerate(selected, 1):
        run_one(
            model,
            torch,
            row,
            args.output_root,
            args.device,
            args.truncation,
            sketch_scope=args.sketch_scope,
            bbox_scale=args.bbox_scale,
        )
        if index % 10 == 0:
            print(f"{index}/{len(selected)}")


if __name__ == "__main__":
    main()
