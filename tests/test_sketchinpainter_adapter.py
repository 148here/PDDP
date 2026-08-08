import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest
from torch.utils.data import DataLoader

from image_synthesis.data.sketchinpainter_dataset import (
    crop_and_pad_sketch,
    resize_full_sketch,
    load_hole_mask,
    load_jsonl,
    load_vq_tokens,
    SketchInpainterPDDPDataset,
    stable_seed,
)
from scripts.sketchinpainter.build_manifest import stable_split, training_root


def test_stable_seed_is_repeatable_and_sensitive():
    assert stable_seed(7, 2, "sample") == stable_seed(7, 2, "sample")
    assert stable_seed(7, 3, "sample") != stable_seed(7, 2, "sample")


def test_mask_composition_rotation_and_token_pool_shape(tmp_path: Path):
    first = np.zeros((8, 8), dtype=np.uint8)
    first[1:3, 2:6] = 255
    second = np.zeros((8, 8), dtype=np.uint8)
    second[5:7, 1:3] = 255
    first_path, second_path = tmp_path / "a.png", tmp_path / "b.png"
    cv2.imwrite(str(first_path), first)
    cv2.imwrite(str(second_path), second)
    mask = load_hole_mask([first_path, second_path], size=(256, 256), rotations=[0, 90])
    assert mask.shape == (256, 256)
    assert mask.dtype == np.uint8
    assert 0 < mask.sum() < mask.size
    pooled = cv2.resize(mask, (32, 32), interpolation=cv2.INTER_AREA) > 0
    assert pooled.shape == (32, 32)


def test_crop_and_pad_sketch_keeps_only_hole_lines():
    sketch = np.full((256, 256), 255, dtype=np.uint8)
    sketch[10:245, 10] = 0  # outside distractor
    sketch[100:150, 120] = 0
    hole = np.zeros((256, 256), dtype=np.uint8)
    hole[80:180, 80:180] = 1
    result = crop_and_pad_sketch(sketch, hole, output_size=224, bbox_scale=1.2)
    assert result.shape == (224, 224, 3)
    assert result.dtype == np.uint8
    # Bilinear upsampling may antialias a one-pixel black line.
    assert result.min() <= 64
    assert np.all(result[:, 0] == 255)


def test_resize_full_sketch_preserves_lines_outside_hole_semantics():
    sketch = np.full((256, 256), 255, dtype=np.uint8)
    sketch[10:245, 10] = 0
    sketch[100:150, 120] = 0
    result = resize_full_sketch(sketch, output_size=224)
    assert result.shape == (224, 224, 3)
    assert result.dtype == np.uint8
    assert result[:, :12].min() < 128
    assert result[:, 90:120].min() < 128


def test_duplicate_manifest_id_is_rejected(tmp_path: Path):
    manifest = tmp_path / "manifest.jsonl"
    row = {"sample_id": "artbench:x"}
    manifest.write_text("\n".join(json.dumps(row) for _ in range(2)), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate sample_id"):
        load_jsonl(manifest)


def test_validation_split_is_deterministic():
    values = [stable_split(f"sample:{index}", 1.0) for index in range(10000)]
    assert values == [stable_split(f"sample:{index}", 1.0) for index in range(10000)]
    assert 50 <= values.count("validation") <= 150


def test_training_root_excludes_sibling_test_split(tmp_path: Path):
    (tmp_path / "train").mkdir()
    (tmp_path / "test").mkdir()
    assert training_root(tmp_path) == tmp_path / "train"


def test_vq_token_cache_schema_and_corruption(tmp_path: Path):
    valid = tmp_path / "valid.npy"
    invalid = tmp_path / "invalid.npy"
    corrupt = tmp_path / "corrupt.npy"
    np.save(valid, np.arange(1024, dtype=np.int64), allow_pickle=False)
    np.save(invalid, np.arange(10, dtype=np.int64), allow_pickle=False)
    corrupt.write_bytes(b"not a numpy file")
    assert load_vq_tokens(valid).shape == (1024,)
    with pytest.raises(ValueError, match="Expected 1024"):
        load_vq_tokens(invalid)
    with pytest.raises(ValueError, match="Invalid VQ token cache"):
        load_vq_tokens(corrupt)


def test_real_sketchinpainter_adapter_shapes_and_reproducibility(tmp_path: Path):
    sketch_root = Path(os.environ.get("SKETCHINPAINTER_ROOT", ""))
    if not sketch_root.is_dir():
        pytest.skip("SKETCHINPAINTER_ROOT is not available")
    image = np.full((64, 64, 3), 180, dtype=np.uint8)
    edge = np.full((64, 64), 255, dtype=np.uint8)
    cv2.line(edge, (8, 32), (55, 32), 0, 2)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[16:48, 16:48] = 255
    image_path, edge_path = tmp_path / "image.png", tmp_path / "edge.png"
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    mask_path, token_path = mask_dir / "mask.png", tmp_path / "tokens.npy"
    cv2.imwrite(str(image_path), image)
    cv2.imwrite(str(edge_path), edge)
    cv2.imwrite(str(mask_path), mask)
    np.save(token_path, np.arange(1024, dtype=np.int64) % 2887, allow_pickle=False)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({
        "sample_id": "artbench:train/example.png",
        "dataset": "artbench",
        "rel_key": "train/example.png",
        "split": "train",
        "image_path": str(image_path),
        "edge_path": str(edge_path),
        "token_path": str(token_path),
    }) + "\n", encoding="utf-8")
    dataset = SketchInpainterPDDPDataset(
        manifest_path=str(manifest), phase="train", mask_dirs=[str(mask_dir)],
        sketchinpainter_root=str(sketch_root), global_seed=17,
        mask_count_probs={1: 1.0}, random_mask_rotate_90=False,
        sketch_overrides={"sigma_mean": 0.0, "sigma_std": 0.0, "move_prob": 0.0,
                          "cp_sigma_mean": 0.0, "cp_sigma_std": 0.0},
    )
    first, second = dataset[0], dataset[0]
    assert first["image"].shape == (3, 256, 256)
    assert first["obj_mask"].shape == (256, 256)
    assert first["sketch"].shape == (3, 224, 224)
    assert first["quantized_image"].shape == (1024,)
    assert first["condition_seed"] == second["condition_seed"]
    assert np.array_equal(first["obj_mask"], second["obj_mask"])
    assert np.array_equal(first["sketch"], second["sketch"])
    batch = next(iter(DataLoader(dataset, batch_size=1, num_workers=0)))
    assert tuple(batch["image"].shape) == (1, 3, 256, 256)
    assert tuple(batch["obj_mask"].shape) == (1, 256, 256)
    assert tuple(batch["sketch"].shape) == (1, 3, 224, 224)
    assert tuple(batch["quantized_image"].shape) == (1, 1024)
