import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from image_synthesis.data.sketchinpainter_dataset import (
    crop_and_pad_sketch,
    bbox_crop_and_pad_sketch,
    expanded_mask_bbox,
    prepare_pddp_sketch,
    resize_full_sketch,
    load_hole_mask,
    load_jsonl,
    load_vq_tokens,
    SketchInpainterPDDPDataset,
    stable_seed,
)
from scripts.sketchinpainter.build_manifest import build_rows, stable_split, training_root
from scripts.sketchinpainter.validate_preprocessed_manifest import validate_manifest_caches


class EpochProbeDataset(SketchInpainterPDDPDataset):
    """Minimal worker probe for the production shared-epoch implementation."""

    def __init__(self):
        self._epoch = torch.zeros((), dtype=torch.int64).share_memory_()

    def __len__(self):
        return 4

    def __getitem__(self, index):
        return stable_seed(11, self.epoch, index)


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


def test_bbox_crop_retains_all_lines_inside_expanded_box_and_discards_outside():
    sketch = np.full((100, 160), 255, dtype=np.uint8)
    hole = np.zeros_like(sketch)
    hole[40:60, 70:90] = 1
    # Bbox scale 2.0 gives x=[60,100), y=[30,70).
    sketch[35:65, 62] = 0       # inside bbox, outside free-form mask
    sketch[35:65, 50] = 0       # outside bbox
    original_mask = hole.copy()
    result, bbox = bbox_crop_and_pad_sketch(sketch, hole, output_size=224, bbox_scale=2.0)
    assert bbox == (60, 30, 100, 70)
    assert result.shape == (224, 224, 3)
    assert result[:, :30].min() < 128
    assert np.array_equal(hole, original_mask)
    # Only one black component was inside the crop.
    assert (result[..., 0] < 128).sum() < 224 * 20


def test_bbox_crop_multicomponent_border_padding_and_training_inference_identity():
    sketch = np.full((80, 160), 255, dtype=np.uint8)
    sketch[0:10, 1] = 0
    sketch[60:75, 150] = 0
    mask = np.zeros((80, 160), dtype=np.uint8)
    mask[0:4, 0:5] = 1
    mask[60:80, 145:160] = 1
    bbox = expanded_mask_bbox(mask, bbox_scale=1.2)
    assert bbox[0] == 0 and bbox[1] == 0 and bbox[2:] == (160, 80)
    training_value, training_meta = prepare_pddp_sketch(
        sketch, mask, scope="bbox_crop", output_size=224, bbox_scale=1.2
    )
    inference_value, inference_meta = prepare_pddp_sketch(
        sketch.copy(), mask.copy(), scope="bbox_crop", output_size=224, bbox_scale=1.2
    )
    assert np.array_equal(training_value, inference_value)
    assert training_meta == inference_meta
    assert np.all(training_value[:50] == 255)  # vertical white padding for a 2:1 crop


def test_bbox_crop_empty_mask_and_invalid_scope_fail():
    value = np.full((8, 8), 255, dtype=np.uint8)
    with pytest.raises(ValueError, match="empty"):
        bbox_crop_and_pad_sketch(value, np.zeros_like(value))
    with pytest.raises(ValueError, match="scope"):
        prepare_pddp_sketch(value, np.ones_like(value), scope="unknown")


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


def test_three_dataset_manifest_excludes_coco_test_and_has_unique_ids(tmp_path: Path):
    output = tmp_path / "output"
    roots = {}
    for name in ("artbench", "coco", "mural1"):
        root = tmp_path / name
        (root / "train" / "1" / "images").mkdir(parents=True)
        (root / "train" / "1" / "images" / f"{name}.png").write_bytes(b"image")
        (root / "test" / "1" / "images").mkdir(parents=True)
        (root / "test" / "1" / "images" / "leak.png").write_bytes(b"image")
        roots[name] = training_root(root)
    rows = [row for name in sorted(roots) for row in build_rows(name, roots[name], output, 1.0)]
    assert {row["dataset"] for row in rows} == {"artbench", "coco", "mural1"}
    assert len(rows) == 3
    assert len({row["sample_id"] for row in rows}) == 3
    assert all("test" not in Path(row["rel_key"]).parts for row in rows)
    assert [row["split"] for row in rows] == [stable_split(row["sample_id"], 1.0) for row in rows]


def test_dataset_weights_produce_expected_effective_length(tmp_path: Path):
    manifest = tmp_path / "manifest.jsonl"
    rows = []
    for dataset, count in (("artbench", 10), ("coco", 20), ("mural1", 5)):
        for index in range(count):
            rows.append({
                "sample_id": f"{dataset}:{index}", "dataset": dataset, "split": "train",
                "image_path": "missing", "edge_path": "missing", "token_path": "missing",
            })
    manifest.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    cv2.imwrite(str(mask_dir / "mask.png"), np.ones((4, 4), dtype=np.uint8) * 255)
    dataset = SketchInpainterPDDPDataset(
        manifest_path=str(manifest), mask_dirs=[str(mask_dir)], sketchinpainter_root=str(tmp_path),
        validate_files=False, dataset_weights={"artbench": 1.0, "coco": 0.43369, "mural1": 0.1},
    )
    assert len(dataset) == round(10 + 20 * 0.43369 + 5 * 0.1)


def test_persistent_workers_observe_epoch_changes_and_repeat_same_epoch():
    dataset = EpochProbeDataset()
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2, persistent_workers=True)
    dataset.set_epoch(3)
    first = next(iter(loader)).tolist()
    dataset.set_epoch(3)
    repeated = next(iter(loader)).tolist()
    dataset.set_epoch(4)
    changed = next(iter(loader)).tolist()
    assert first == repeated
    assert first != changed


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


def test_manifest_cache_validation_is_per_row_not_global_count(tmp_path: Path):
    edge = tmp_path / "edge.png"
    token = tmp_path / "token.npy"
    cv2.imwrite(str(edge), np.full((8, 8), 255, dtype=np.uint8))
    np.save(token, np.arange(1024, dtype=np.int64), allow_pickle=False)
    manifest = tmp_path / "manifest.jsonl"
    row = {"sample_id": "coco:one", "dataset": "coco", "edge_path": str(edge), "token_path": str(token)}
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    report = validate_manifest_caches(manifest)
    assert report["valid"] and report["rows"] == 1 and report["datasets"] == {"coco": 1}
    token.write_bytes(b"corrupt")
    report = validate_manifest_caches(manifest)
    assert not report["valid"] and len(report["errors"]) == 1


def test_real_sketchinpainter_adapter_shapes_and_reproducibility(tmp_path: Path):
    sketch_root_value = os.environ.get("SKETCHINPAINTER_ROOT")
    if not sketch_root_value:
        pytest.skip("SKETCHINPAINTER_ROOT is not available")
    sketch_root = Path(sketch_root_value)
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
