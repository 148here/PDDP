"""SketchInpainter-compatible data adapter for PDDP fine-tuning.

The adapter deliberately consumes precomputed MuGE edges and VQ tokens.  It
does not load either heavyweight model in DataLoader workers.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def stable_seed(*parts: object, modulo: int = 2**31 - 1) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulo


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row["sample_id"])
            if sample_id in seen:
                raise ValueError(f"Duplicate sample_id at line {line_number}: {sample_id}")
            seen.add(sample_id)
            rows.append(row)
    return rows


def load_vq_tokens(path: str | Path, expected_count: int = 1024) -> np.ndarray:
    try:
        tokens = np.load(path, allow_pickle=False)
    except Exception as error:
        raise ValueError(f"Invalid VQ token cache: {path}") from error
    tokens = np.asarray(tokens, dtype=np.int64).reshape(-1)
    if tokens.size != expected_count:
        raise ValueError(f"Expected {expected_count} VQ tokens in {path}, got {tokens.size}")
    if tokens.min(initial=0) < 0:
        raise ValueError(f"VQ token cache contains negative indices: {path}")
    return tokens


def _read_gray(path: str | Path) -> np.ndarray:
    value = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if value is None:
        raise ValueError(f"Failed to read grayscale image: {path}")
    return value


def load_hole_mask(
    paths: Iterable[str | Path],
    *,
    size: tuple[int, int] = (256, 256),
    rotations: Iterable[int] | None = None,
) -> np.ndarray:
    selected = list(paths)
    rotation_values = list(rotations or [0] * len(selected))
    if len(selected) != len(rotation_values):
        raise ValueError("mask paths and rotations must have equal lengths")
    combined = np.zeros(size, dtype=np.uint8)
    for path, degrees in zip(selected, rotation_values):
        mask = _read_gray(path)
        turns = (int(degrees) // 90) % 4
        if turns:
            mask = np.rot90(mask, k=turns).copy()
        mask = cv2.resize(mask, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
        combined = np.maximum(combined, (mask >= 128).astype(np.uint8))
    if not combined.any():
        raise ValueError(f"Selected masks produce an empty hole: {selected}")
    return combined


def crop_and_pad_sketch(
    sketch: np.ndarray,
    hole_mask: np.ndarray,
    *,
    output_size: int = 224,
    bbox_scale: float = 1.2,
) -> np.ndarray:
    """Keep lines inside the hole, crop an expanded bbox, and pad white."""
    if sketch.ndim == 3:
        sketch = cv2.cvtColor(sketch, cv2.COLOR_BGR2GRAY)
    if sketch.shape != hole_mask.shape:
        sketch = cv2.resize(sketch, (hole_mask.shape[1], hole_mask.shape[0]), interpolation=cv2.INTER_LINEAR)
    ys, xs = np.where(hole_mask > 0)
    if not len(xs):
        raise ValueError("hole_mask is empty")
    local = np.full_like(sketch, 255)
    local[hole_mask > 0] = sketch[hole_mask > 0]
    cx, cy = (xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0
    width = max(1, int(round((xs.max() - xs.min() + 1) * bbox_scale)))
    height = max(1, int(round((ys.max() - ys.min() + 1) * bbox_scale)))
    x0 = max(0, int(round(cx - width / 2)))
    y0 = max(0, int(round(cy - height / 2)))
    x1 = min(local.shape[1], x0 + width)
    y1 = min(local.shape[0], y0 + height)
    crop = local[y0:y1, x0:x1]
    scale = min(output_size / crop.shape[1], output_size / crop.shape[0])
    new_w = max(1, int(round(crop.shape[1] * scale)))
    new_h = max(1, int(round(crop.shape[0] * scale)))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.full((output_size, output_size), 255, dtype=np.uint8)
    xoff, yoff = (output_size - new_w) // 2, (output_size - new_h) // 2
    canvas[yoff:yoff + new_h, xoff:xoff + new_w] = resized
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)


def expanded_mask_bbox(hole_mask: np.ndarray, *, bbox_scale: float = 1.2) -> tuple[int, int, int, int]:
    """Return a clipped, exclusive-end bbox around all nonzero mask pixels."""
    if bbox_scale <= 0:
        raise ValueError("bbox_scale must be positive")
    if hole_mask.ndim != 2:
        raise ValueError(f"hole_mask must be 2D, got shape {hole_mask.shape}")
    ys, xs = np.where(hole_mask > 0)
    if xs.size == 0:
        raise ValueError("hole_mask is empty")
    source_x0, source_x1 = int(xs.min()), int(xs.max()) + 1
    source_y0, source_y1 = int(ys.min()), int(ys.max()) + 1
    width = max(1, int(round((source_x1 - source_x0) * bbox_scale)))
    height = max(1, int(round((source_y1 - source_y0) * bbox_scale)))
    center_x = (source_x0 + source_x1) / 2.0
    center_y = (source_y0 + source_y1) / 2.0
    x0 = max(0, int(round(center_x - width / 2.0)))
    y0 = max(0, int(round(center_y - height / 2.0)))
    x1 = min(hole_mask.shape[1], int(round(center_x + width / 2.0)))
    y1 = min(hole_mask.shape[0], int(round(center_y + height / 2.0)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"expanded mask bbox is empty: {(x0, y0, x1, y1)}")
    return x0, y0, x1, y1


def _pad_sketch_crop(crop: np.ndarray, *, output_size: int) -> np.ndarray:
    if output_size <= 0:
        raise ValueError("output_size must be positive")
    if crop.size == 0:
        raise ValueError("sketch crop is empty")
    scale = min(output_size / crop.shape[1], output_size / crop.shape[0])
    new_w = max(1, int(round(crop.shape[1] * scale)))
    new_h = max(1, int(round(crop.shape[0] * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(crop, (new_w, new_h), interpolation=interpolation)
    canvas = np.full((output_size, output_size), 255, dtype=np.uint8)
    xoff, yoff = (output_size - new_w) // 2, (output_size - new_h) // 2
    canvas[yoff:yoff + new_h, xoff:xoff + new_w] = resized
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)


def bbox_crop_and_pad_sketch(
    sketch: np.ndarray,
    hole_mask: np.ndarray,
    *,
    output_size: int = 224,
    bbox_scale: float = 1.2,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop every sketch line in the expanded mask bbox, then aspect-pad white.

    The free-form mask is used only to locate the crop.  Lines inside the crop
    are deliberately retained even when they lie outside the mask itself.
    """
    if sketch.ndim == 3:
        sketch = cv2.cvtColor(sketch, cv2.COLOR_BGR2GRAY)
    if sketch.shape != hole_mask.shape:
        sketch = cv2.resize(sketch, (hole_mask.shape[1], hole_mask.shape[0]), interpolation=cv2.INTER_LINEAR)
    bbox = expanded_mask_bbox(hole_mask, bbox_scale=bbox_scale)
    x0, y0, x1, y1 = bbox
    return _pad_sketch_crop(sketch[y0:y1, x0:x1], output_size=output_size), bbox


def resize_full_sketch(sketch: np.ndarray, *, output_size: int = 224) -> np.ndarray:
    """Preserve the complete canonical sketch and only satisfy PDDP's input size."""
    if sketch.ndim == 3:
        sketch = cv2.cvtColor(sketch, cv2.COLOR_BGR2GRAY)
    interpolation = cv2.INTER_AREA if max(sketch.shape) > output_size else cv2.INTER_LINEAR
    resized = cv2.resize(sketch, (output_size, output_size), interpolation=interpolation)
    return cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)


def prepare_pddp_sketch(
    sketch: np.ndarray,
    hole_mask: np.ndarray,
    *,
    scope: str,
    output_size: int = 224,
    bbox_scale: float = 1.2,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Single training/inference entry point for all supported sketch scopes."""
    normalized_scope = str(scope).strip().lower()
    if normalized_scope == "full":
        value = resize_full_sketch(sketch, output_size=output_size)
        bbox = None
    elif normalized_scope == "hole_crop":
        value = crop_and_pad_sketch(sketch, hole_mask, output_size=output_size, bbox_scale=bbox_scale)
        bbox = expanded_mask_bbox(hole_mask, bbox_scale=bbox_scale)
    elif normalized_scope == "bbox_crop":
        value, bbox = bbox_crop_and_pad_sketch(
            sketch, hole_mask, output_size=output_size, bbox_scale=bbox_scale
        )
    else:
        raise ValueError("sketch scope must be one of: bbox_crop, full, hole_crop")
    return value, {
        "scope": normalized_scope,
        "bbox_scale": float(bbox_scale),
        "crop_bbox_xyxy": list(bbox) if bbox is not None else None,
        "source_size_hw": [int(hole_mask.shape[0]), int(hole_mask.shape[1])],
        "output_size_hw": [int(output_size), int(output_size)],
    }


class SketchInpainterPDDPDataset(Dataset):
    """Dataset backed by a collision-safe JSONL manifest and per-image tokens."""

    def __init__(
        self,
        data_root: str = "",
        manifest_path: str = "",
        phase: str = "train",
        mask_dirs: list[str] | None = None,
        sketchinpainter_root: str = "",
        global_seed: int = 20260425,
        dataset_weights: dict[str, float] | None = None,
        mask_count_probs: dict[int, float] | None = None,
        random_mask_rotate_90: bool = True,
        image_size: list[int] | tuple[int, int] = (256, 256),
        sketch_size: list[int] | tuple[int, int] = (224, 224),
        bbox_scale: float = 1.2,
        sketch_scope: str = "bbox_crop",
        sketch_overrides: dict[str, Any] | None = None,
        validate_files: bool = True,
        **_: Any,
    ) -> None:
        del data_root
        self.phase = str(phase)
        self.rows = [row for row in load_jsonl(manifest_path) if row.get("split") == self.phase]
        if not self.rows:
            raise ValueError(f"No manifest rows for split={self.phase!r}: {manifest_path}")
        self.mask_paths = sorted(
            path for root in (mask_dirs or []) for path in Path(root).rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.mask_paths:
            raise FileNotFoundError(f"No mask images found under: {mask_dirs}")
        self.sketchinpainter_root = Path(sketchinpainter_root).resolve()
        self.global_seed = int(global_seed)
        # Shared tensor stays visible in already-running persistent workers.
        self._epoch = torch.zeros((), dtype=torch.int64).share_memory_()
        self.image_size = tuple(int(value) for value in image_size)
        self.sketch_size = tuple(int(value) for value in sketch_size)
        self.bbox_scale = float(bbox_scale)
        self.sketch_scope = str(sketch_scope).strip().lower()
        if self.sketch_scope not in {"bbox_crop", "full", "hole_crop"}:
            raise ValueError("sketch_scope must be one of: bbox_crop, full, hole_crop")
        self.weights = {"artbench": 1.0, "coco": 0.43369, "mural1": 0.1, **(dataset_weights or {})}
        self.mask_count_probs = {int(key): float(value) for key, value in (mask_count_probs or {1: 0.5, 2: 0.5}).items()}
        self.random_mask_rotate_90 = bool(random_mask_rotate_90)
        self.sketch_overrides = dict(sketch_overrides or {})
        self.by_dataset: dict[str, list[dict[str, Any]]] = {}
        for row in self.rows:
            self.by_dataset.setdefault(str(row["dataset"]), []).append(row)
        self.effective_length = len(self.rows) if self.phase != "train" else max(
            1, int(round(sum(len(rows) * self.weights.get(name, 1.0) for name, rows in self.by_dataset.items())))
        )
        if validate_files:
            for row in self.rows:
                for key in ("image_path", "edge_path", "token_path"):
                    if not Path(row[key]).is_file():
                        raise FileNotFoundError(f"Missing {key} for {row['sample_id']}: {row[key]}")

    def set_epoch(self, epoch: int) -> None:
        self._epoch.fill_(int(epoch))

    @property
    def epoch(self) -> int:
        return int(self._epoch.item())

    def __len__(self) -> int:
        return self.effective_length

    def _row_for_index(self, index: int) -> dict[str, Any]:
        if self.phase != "train":
            return self.rows[index]
        rng = random.Random(stable_seed(self.global_seed, self.epoch, index, "dataset"))
        names = sorted(self.by_dataset)
        totals = [len(self.by_dataset[name]) * self.weights.get(name, 1.0) for name in names]
        name = rng.choices(names, weights=totals, k=1)[0]
        return self.by_dataset[name][rng.randrange(len(self.by_dataset[name]))]

    def _condition(self, row: dict[str, Any], index: int) -> tuple[np.ndarray, np.ndarray, list[str], list[int], int]:
        seed = stable_seed(self.global_seed, self.epoch, row["sample_id"], index)
        rng = random.Random(seed)
        counts, weights = zip(*sorted(self.mask_count_probs.items()))
        count = rng.choices(counts, weights=weights, k=1)[0]
        selected = rng.sample(self.mask_paths, k=min(count, len(self.mask_paths)))
        rotations = [rng.randrange(4) * 90 if self.random_mask_rotate_90 else 0 for _ in selected]
        mask = load_hole_mask(selected, size=self.image_size, rotations=rotations)
        return mask, np.asarray(selected, dtype=object), [str(path) for path in selected], rotations, seed

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self._row_for_index(index)
        image = Image.open(row["image_path"]).convert("RGB").resize(
            (self.image_size[1], self.image_size[0]), Image.Resampling.BICUBIC
        )
        mask, _, mask_paths, rotations, condition_seed = self._condition(row, index)
        edge = _read_gray(row["edge_path"])
        edge = cv2.resize(edge, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_LINEAR)
        root = str(self.sketchinpainter_root)
        if root not in sys.path:
            sys.path.insert(0, root)
        from dataset.makesketch import make_sketch_from_edge

        sketch = make_sketch_from_edge(
            edge,
            seed=condition_seed,
            mask=(mask * 255).astype(np.uint8),
            mask_mode="mask_region",
            boundary_pin_px=12.0,
            **self.sketch_overrides,
        )
        pddp_sketch, sketch_preprocessing = prepare_pddp_sketch(
            sketch,
            mask,
            scope=self.sketch_scope,
            output_size=self.sketch_size[0],
            bbox_scale=self.bbox_scale,
        )
        tokens = load_vq_tokens(row["token_path"])
        return {
            "image": np.asarray(image, dtype=np.float32).transpose(2, 0, 1),
            "obj_mask": mask.astype(np.float32),
            "sketch": pddp_sketch.astype(np.float32).transpose(2, 0, 1),
            "quantized_image": tokens,
            "path": str(row["image_path"]),
            "sample_id": str(row["sample_id"]),
            "sketch_file": str(row["sample_id"]),
            "condition_seed": condition_seed,
            "mask_paths": "|".join(mask_paths),
            # Strings keep default DataLoader collation valid when mask_count is 1 or 2.
            "mask_rotations": "|".join(str(value) for value in rotations),
            "sketch_preprocessing": json.dumps(sketch_preprocessing, sort_keys=True),
        }
