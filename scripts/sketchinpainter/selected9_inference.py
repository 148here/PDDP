"""Prepare and audit exact selected-nine, three-round PDDP inference.

The source manifest is owned by SketchInpainter.  This adapter never scans a
dataset or regenerates masks/sketches: it verifies and reuses the exact saved
assets, including the full comparison seed used by OminiControl.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw

MAX_COMPARISON_SEED = 2**63 - 1
MAX_PDDP_SEED = 2**31 - 1
REQUIRED_ASSETS = ("rgb", "edge", "mask", "sketch", "prompt", "source_metadata")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_id(condition_id: str) -> str:
    return hashlib.sha256(condition_id.encode("utf-8")).hexdigest()[:24]


def comparison_seed(global_seed: int, round_index: int, dataset_name: str, rel_key: str) -> int:
    if round_index < 1:
        raise ValueError("round_index must be >= 1")
    payload = f"{int(global_seed)}::{round_index - 1}::{dataset_name}::{rel_key}"
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big") % MAX_COMPARISON_SEED


def _resolve(path: str, manifest_dir: Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (manifest_dir / value).resolve()


@dataclass(frozen=True)
class SelectedSample:
    sample_id: str
    dataset_name: str
    rel_key: str
    paths: dict[str, Path]
    hashes: dict[str, str]


def load_manifest(path: Path, *, verify_files: bool = True) -> tuple[dict[str, Any], list[SelectedSample]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("mask_convention") != "white_is_hole":
        raise ValueError("Only white_is_hole masks are supported")
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list):
        raise ValueError("samples must be a list")
    expected = int(payload.get("expected_sample_count", len(raw_samples)))
    if len(raw_samples) != expected or expected != 9:
        raise ValueError(f"Selected-nine manifest must contain exactly 9 samples, found {len(raw_samples)}")
    samples: list[SelectedSample] = []
    identities: set[tuple[str, str]] = set()
    sample_ids: set[str] = set()
    for raw in raw_samples:
        sample_id = str(raw["sample_id"])
        dataset_name = str(raw["dataset_name"])
        rel_key = str(raw["rel_key"]).replace("\\", "/")
        if sample_id in sample_ids or (dataset_name, rel_key) in identities:
            raise ValueError(f"Duplicate selected-nine identity: {sample_id}")
        sample_ids.add(sample_id)
        identities.add((dataset_name, rel_key))
        raw_paths = raw.get("paths", {})
        raw_hashes = raw.get("sha256", {})
        missing = [name for name in REQUIRED_ASSETS if name not in raw_paths or name not in raw_hashes]
        if missing:
            raise ValueError(f"{sample_id} missing assets/hashes: {missing}")
        paths = {name: _resolve(str(raw_paths[name]), path.parent) for name in REQUIRED_ASSETS}
        hashes = {name: str(raw_hashes[name]).lower() for name in REQUIRED_ASSETS}
        if verify_files:
            for name in REQUIRED_ASSETS:
                if not paths[name].is_file():
                    raise FileNotFoundError(paths[name])
                actual = sha256_file(paths[name])
                if actual != hashes[name]:
                    raise ValueError(f"{sample_id} {name} hash mismatch: {actual} != {hashes[name]}")
            with Image.open(paths["rgb"]) as rgb, Image.open(paths["mask"]) as mask:
                if rgb.size != (512, 512) or mask.size != rgb.size:
                    raise ValueError(f"{sample_id} requires 512x512 RGB/mask, got {rgb.size}/{mask.size}")
                mask_values = set(int(value) for value in np.unique(np.asarray(mask.convert("L"))))
                if mask_values != {0, 255}:
                    raise ValueError(f"{sample_id} mask is not binary white-is-hole: {sorted(mask_values)}")
            if not paths["prompt"].read_text(encoding="utf-8").strip():
                raise ValueError(f"{sample_id} has an empty prompt")
            if not isinstance(json.loads(paths["source_metadata"].read_text(encoding="utf-8")), dict):
                raise ValueError(f"{sample_id} source metadata is not an object")
        samples.append(SelectedSample(sample_id, dataset_name, rel_key, paths, hashes))
    return payload, samples


def build_rows(samples: Iterable[SelectedSample], *, global_seed: int, num_rounds: int) -> list[dict[str, Any]]:
    if num_rounds != 3:
        raise ValueError("The selected-nine comparison protocol requires exactly 3 rounds")
    rows: list[dict[str, Any]] = []
    for sample in samples:
        shared_payload = json.dumps(
            {"sample_id": sample.sample_id, "dataset_name": sample.dataset_name,
             "rel_key": sample.rel_key, "sha256": sample.hashes},
            sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        )
        shared_hash = hashlib.sha256(shared_payload.encode("utf-8")).hexdigest()
        for round_index in range(1, num_rounds + 1):
            full_seed = comparison_seed(global_seed, round_index, sample.dataset_name, sample.rel_key)
            condition_id = f"selected9::{sample.sample_id}::round_{round_index:03d}"
            rows.append({
                "condition_id": condition_id,
                "source_id": sample.sample_id,
                "dataset_name": sample.dataset_name,
                "rel_key": sample.rel_key,
                "round_index": round_index,
                "seed": full_seed,
                "comparison_seed": full_seed,
                "effective_seed": full_seed % MAX_PDDP_SEED,
                "shared_condition_hash": shared_hash,
                "assets": {
                    "gt_rgb": {"source_path": str(sample.paths["rgb"]), "sha256": sample.hashes["rgb"]},
                    "hole_mask": {"source_path": str(sample.paths["mask"]), "sha256": sample.hashes["mask"]},
                    "base_sketch": {"source_path": str(sample.paths["sketch"]), "sha256": sample.hashes["sketch"]},
                    "edge": {"source_path": str(sample.paths["edge"]), "sha256": sample.hashes["edge"]},
                    "prompt": {"source_path": str(sample.paths["prompt"]), "sha256": sample.hashes["prompt"]},
                    "source_metadata": {"source_path": str(sample.paths["source_metadata"]),
                                        "sha256": sample.hashes["source_metadata"]},
                },
            })
    if len(rows) != 27 or len({row["condition_id"] for row in rows}) != 27:
        raise AssertionError("Expected exactly 27 unique selected-nine conditions")
    return rows


def write_plan(plan_root: Path, manifest_path: Path, manifest: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    plan_root.mkdir(parents=True, exist_ok=True)
    conditions_path = plan_root / "selected9_conditions.jsonl"
    conditions_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    seed_rows: dict[str, dict[str, int]] = {}
    for row in rows:
        seed_rows.setdefault(row["source_id"], {})[f"round_{row['round_index']:03d}"] = row["comparison_seed"]
    (plan_root / "seed_manifest.json").write_text(json.dumps({
        "version": 1,
        "policy": "sha256(global_seed::zero_based_round::dataset_name::rel_key)[:8] mod (2^63-1)",
        "global_seed": int(manifest.get("global_seed", 20260425)),
        "num_rounds": 3,
        "samples": seed_rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (plan_root / "input_audit.json").write_text(json.dumps({
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "sample_count": len({row["source_id"] for row in rows}),
        "condition_count": len(rows),
        "mask_convention": manifest["mask_convention"],
        "all_source_hashes_verified": True,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    input_names = {
        "gt_rgb": "rgb.png", "hole_mask": "mask.png", "base_sketch": "sketch.png",
        "edge": "edge.png", "prompt": "prompt.txt", "source_metadata": "source_metadata.json",
    }
    input_index: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    for row in rows:
        if row["source_id"] in seen_sources:
            continue
        seen_sources.add(row["source_id"])
        sample_root = plan_root / "inputs" / row["source_id"]
        sample_root.mkdir(parents=True, exist_ok=True)
        copied: dict[str, dict[str, str]] = {}
        for asset_key, filename in input_names.items():
            source = Path(row["assets"][asset_key]["source_path"])
            expected = row["assets"][asset_key]["sha256"]
            target = sample_root / filename
            if target.is_file() and sha256_file(target) != expected:
                raise ValueError(f"Existing input snapshot hash mismatch: {target}")
            if not target.is_file():
                shutil.copy2(source, target)
            if sha256_file(target) != expected:
                raise ValueError(f"Copied input snapshot hash mismatch: {target}")
            copied[asset_key] = {"path": str(target), "sha256": expected}
        input_index.append({"sample_id": row["source_id"], "dataset_name": row["dataset_name"],
                            "rel_key": row["rel_key"], "assets": copied})
    (plan_root / "input_index.json").write_text(json.dumps({
        "version": 1, "sample_count": len(input_index), "samples": input_index,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_rgb(path: Path, size: int = 256) -> Image.Image:
    return Image.open(path).convert("RGB").resize((size, size), Image.Resampling.LANCZOS)


def _tile(image: Image.Image, label: str) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + 24), "white")
    canvas.paste(image, (0, 24))
    ImageDraw.Draw(canvas).text((6, 6), label, fill="black")
    return canvas


def audit_results(rows: list[dict[str, Any]], output_root: Path, *, make_contacts: bool = True) -> dict[str, Any]:
    errors: list[str] = []
    complete: list[dict[str, Any]] = []
    by_source: dict[str, list[tuple[dict[str, Any], Path]]] = {}
    for row in rows:
        item_dir = output_root / row["dataset_name"] / f"round_{row['round_index']:03d}" / safe_id(row["condition_id"])
        generated = item_dir / "generated.png"
        metadata_path = item_dir / "metadata.json"
        if not generated.is_file() or not metadata_path.is_file():
            errors.append(f"missing result: {row['condition_id']}")
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("comparison_seed") != row["comparison_seed"]:
            errors.append(f"comparison seed mismatch: {row['condition_id']}")
        if metadata.get("effective_seed") != row["effective_seed"]:
            errors.append(f"effective seed mismatch: {row['condition_id']}")
        if metadata.get("canonical_shared_condition_hash") != row["shared_condition_hash"]:
            errors.append(f"input identity mismatch: {row['condition_id']}")
        if not metadata.get("outside_hole_exact_gt"):
            errors.append(f"outside-hole mismatch: {row['condition_id']}")
        if sha256_file(generated) != metadata.get("generated_sha256"):
            errors.append(f"generated hash mismatch: {row['condition_id']}")
        complete.append({"condition_id": row["condition_id"], "path": str(generated),
                         "sha256": sha256_file(generated)})
        by_source.setdefault(row["source_id"], []).append((row, generated))
    contact_paths: list[str] = []
    if make_contacts and not errors and len(complete) == 27:
        contact_root = output_root / "contact_sheets"
        contact_root.mkdir(parents=True, exist_ok=True)
        for source_id, values in by_source.items():
            values.sort(key=lambda value: value[0]["round_index"])
            row0 = values[0][0]
            rgb = Path(row0["assets"]["gt_rgb"]["source_path"])
            mask = Path(row0["assets"]["hole_mask"]["source_path"])
            sketch = Path(row0["assets"]["base_sketch"]["source_path"])
            tiles = [_tile(_load_rgb(rgb), "Input / GT"), _tile(_load_rgb(mask), "Mask (white=hole)"),
                     _tile(_load_rgb(sketch), "Sketch")]
            tiles += [_tile(_load_rgb(path), f"PDDP round {row['round_index']}") for row, path in values]
            sheet = Image.new("RGB", (256 * len(tiles), 280), (230, 230, 230))
            for index, tile in enumerate(tiles):
                sheet.paste(tile, (index * 256, 0))
            contact_path = contact_root / f"{source_id}.png"
            sheet.save(contact_path)
            contact_paths.append(str(contact_path))
    report = {
        "expected_results": 27,
        "successful_results": len(complete),
        "unique_samples": len(by_source),
        "rounds_per_sample": {key: len(value) for key, value in sorted(by_source.items())},
        "errors": errors,
        "contact_sheets": contact_paths,
        "results": complete,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "result_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-rounds", type=int, default=3)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--audit-results", action="store_true")
    parser.add_argument("--no-verify-files", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    manifest, samples = load_manifest(args.manifest.resolve(), verify_files=not args.no_verify_files)
    rows = build_rows(samples, global_seed=int(manifest.get("global_seed", 20260425)), num_rounds=args.num_rounds)
    report = {"samples": len(samples), "conditions": len(rows), "datasets": {}, "unique_seeds": len({r["comparison_seed"] for r in rows})}
    for sample in samples:
        report["datasets"][sample.dataset_name] = report["datasets"].get(sample.dataset_name, 0) + 1
    if args.prepare:
        write_plan(args.output_root, args.manifest, manifest, rows)
    if args.audit_results:
        report["result_audit"] = audit_results(rows, args.output_root)
        if report["result_audit"]["errors"] or report["result_audit"]["successful_results"] != 27:
            print(json.dumps(report, ensure_ascii=False, indent=2))
            raise SystemExit(2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
