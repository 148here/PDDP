from pathlib import Path

import numpy as np
import pytest

from scripts.sketchinpainter.batch_inference import (
    apply_checkpoint_weights,
    core_key_mismatches,
    raw_image_stats,
    read_condition_ids,
)
from scripts.sketchinpainter.prepare_pretrained_smoke import select_conditions


def _difficulty_rows():
    rows = []
    sizes = {"artbench": 50, "coco": 50, "mural1": 17}
    rank_offset = {"Easy": 0, "Hard": 100}
    for dataset, size in sizes.items():
        for difficulty in ("Easy", "Hard"):
            for index in range(size):
                source_id = f"{dataset}:source_{difficulty.lower()}_{index:03d}"
                rows.append({
                    "dataset_name": dataset,
                    "difficulty": difficulty,
                    "source_id": source_id,
                    "difficulty_rank": rank_offset[difficulty] + index,
                    "difficulty_score": float(rank_offset[difficulty] + index),
                    "condition_ids": [f"{source_id}:round_001", f"{source_id}:round_002"],
                })
    return rows


def test_deterministic_smoke_selection_and_quotas():
    rows = _difficulty_rows()
    first = select_conditions(rows)
    second = select_conditions(list(reversed(rows)))
    assert first == second
    assert len(first) == 6
    assert {name: sum(row["dataset_name"] == name for row in first)
            for name in ("artbench", "coco", "mural1")} == {
                "artbench": 2, "coco": 2, "mural1": 2,
            }
    assert {name: sum(row["difficulty"] == name for row in first)
            for name in ("Easy", "Hard")} == {"Easy": 3, "Hard": 3}
    assert all(row["condition_id"].endswith(":round_001") for row in first)
    assert [row["selection_index"] for row in first] == [24, 24, 24, 24, 8, 8]


def test_core_checkpoint_mismatch_filter_only_allows_external_vq():
    keys = [
        "content_codec.vq.encoder.conv_in.weight",
        "transformer.transformer.blocks.0.attn.key.weight",
        "condition_codec.some_weight",
    ]
    assert core_key_mismatches(keys) == keys[1:]


class _LoadTarget:
    def __init__(self, mismatch=([], [])):
        self.mismatch = mismatch
        self.loaded = []

    def load_state_dict(self, value, strict=False):
        self.loaded.append((value, strict))
        return self.mismatch


class _FakeModel(_LoadTarget):
    def __init__(self, mismatch=([], []), ema_mismatch=([], [])):
        super().__init__(mismatch)
        self.ema = _LoadTarget(ema_mismatch)

    def get_ema_model(self):
        return self.ema


def test_checkpoint_loader_applies_ema_after_base_model():
    model = _FakeModel()
    audit = apply_checkpoint_weights(model, {"model": {"base": 1}, "ema": {"ema": 2}})
    assert model.loaded == [({"base": 1}, False)]
    assert model.ema.loaded == [({"ema": 2}, False)]
    assert audit["ema_present"] and audit["ema_applied"]


def test_checkpoint_loader_rejects_core_or_ema_mismatch():
    with pytest.raises(RuntimeError, match="critical model key"):
        apply_checkpoint_weights(_FakeModel(mismatch=(["transformer.required"], [])),
                                 {"model": {}, "ema": {}})
    with pytest.raises(RuntimeError, match="EMA"):
        apply_checkpoint_weights(_FakeModel(ema_mismatch=(["required"], [])),
                                 {"model": {}, "ema": {}})


def test_raw_output_degradation_flags():
    normal = np.arange(256, dtype=np.float32).reshape(16, 16)
    normal_stats = raw_image_stats(normal)
    assert normal_stats["finite"]
    assert not normal_stats["near_constant"]
    assert not normal_stats["near_all_black"]
    assert not normal_stats["near_all_white"]
    assert raw_image_stats(np.zeros((8, 8, 3), dtype=np.float32))["near_all_black"]
    assert raw_image_stats(np.full((8, 8, 3), 255, dtype=np.float32))["near_all_white"]
    assert not raw_image_stats(np.array([np.nan], dtype=np.float32))["finite"]


def test_condition_id_file_rejects_duplicates(tmp_path: Path):
    path = tmp_path / "ids.txt"
    path.write_text("one\none\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate condition IDs"):
        read_condition_ids(path)
