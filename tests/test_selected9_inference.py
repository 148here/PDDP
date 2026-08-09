import hashlib
import json
from pathlib import Path

from PIL import Image

from scripts.sketchinpainter.selected9_inference import (
    MAX_PDDP_SEED,
    build_rows,
    comparison_seed,
    load_manifest,
    write_plan,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(tmp_path: Path) -> Path:
    samples = []
    for index in range(9):
        root = tmp_path / f"sample_{index}"
        root.mkdir()
        rgb = root / "rgb.png"
        edge = root / "edge.png"
        mask = root / "mask.png"
        sketch = root / "sketch.png"
        prompt = root / "prompt.txt"
        metadata = root / "metadata.json"
        Image.new("RGB", (512, 512), (index, 20, 30)).save(rgb)
        Image.new("L", (512, 512), 10).save(edge)
        binary = Image.new("L", (512, 512), 0)
        for x in range(100, 200):
            for y in range(100, 200):
                binary.putpixel((x, y), 255)
        binary.save(mask)
        Image.new("RGB", (512, 512), "white").save(sketch)
        prompt.write_text("a prompt", encoding="utf-8")
        metadata.write_text("{}", encoding="utf-8")
        paths = {"rgb": rgb, "edge": edge, "mask": mask, "sketch": sketch,
                 "prompt": prompt, "source_metadata": metadata}
        samples.append({
            "sample_id": f"sample_{index}", "dataset_name": ("mural1", "artbench", "coco")[index // 3],
            "rel_key": f"test/images/{index}.png", "paths": {key: str(value) for key, value in paths.items()},
            "sha256": {key: _sha(value) for key, value in paths.items()},
        })
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"expected_sample_count": 9, "global_seed": 20260425,
                                "mask_convention": "white_is_hole", "samples": samples}), encoding="utf-8")
    return path


def test_selected9_plan_has_exact_inputs_and_three_distinct_rounds(tmp_path: Path):
    manifest_path = _manifest(tmp_path)
    payload, samples = load_manifest(manifest_path)
    rows = build_rows(samples, global_seed=payload["global_seed"], num_rounds=3)
    assert len(rows) == 27
    assert len({row["condition_id"] for row in rows}) == 27
    assert len({row["source_id"] for row in rows}) == 9
    assert {sum(row["dataset_name"] == name for row in rows) for name in ("mural1", "artbench", "coco")} == {9}
    for sample in samples:
        selected = [row for row in rows if row["source_id"] == sample.sample_id]
        assert len({row["comparison_seed"] for row in selected}) == 3
        assert all(row["effective_seed"] == row["comparison_seed"] % MAX_PDDP_SEED for row in selected)
        assert all(row["assets"]["base_sketch"]["sha256"] == sample.hashes["sketch"] for row in selected)


def test_seed_policy_matches_ominicontrol_formula():
    seed = comparison_seed(20260425, 1, "mural1", "test/1/images/000625.jpg")
    assert seed == 3986938990293272934


def test_write_plan_records_27_conditions(tmp_path: Path):
    manifest_path = _manifest(tmp_path)
    payload, samples = load_manifest(manifest_path)
    rows = build_rows(samples, global_seed=payload["global_seed"], num_rounds=3)
    plan_root = tmp_path / "plan"
    write_plan(plan_root, manifest_path, payload, rows)
    assert len((plan_root / "selected9_conditions.jsonl").read_text(encoding="utf-8").splitlines()) == 27
    audit = json.loads((plan_root / "input_audit.json").read_text(encoding="utf-8"))
    assert audit["sample_count"] == 9 and audit["condition_count"] == 27
    input_index = json.loads((plan_root / "input_index.json").read_text(encoding="utf-8"))
    assert input_index["sample_count"] == 9
    assert len(list((plan_root / "inputs").glob("*/rgb.png"))) == 9
    for sample in input_index["samples"]:
        assert len(sample["assets"]) == 6
        assert all(_sha(Path(asset["path"])) == asset["sha256"] for asset in sample["assets"].values())
