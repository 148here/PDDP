"""Select deterministic canonical smoke conditions and build a contact sheet."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def safe_id(condition_id: str) -> str:
    return hashlib.sha256(condition_id.encode("utf-8")).hexdigest()[:24]


def select_conditions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for dataset in ("artbench", "coco", "mural1"):
        for difficulty in ("Easy", "Hard"):
            group = sorted(
                (row for row in rows if row["dataset_name"] == dataset and row["difficulty"] == difficulty),
                key=lambda row: (int(row["difficulty_rank"]), str(row["source_id"])),
            )
            if not group:
                raise ValueError(f"No difficulty rows for {dataset}/{difficulty}")
            chosen = group[(len(group) - 1) // 2]
            round_one = [value for value in chosen["condition_ids"] if value.endswith(":round_001")]
            if len(round_one) != 1:
                raise ValueError(f"Expected exactly one round_001 condition for {chosen['source_id']}")
            selected.append({
                "dataset_name": dataset,
                "difficulty": difficulty,
                "source_id": chosen["source_id"],
                "difficulty_rank": chosen["difficulty_rank"],
                "difficulty_score": chosen["difficulty_score"],
                "group_size": len(group),
                "selection_index": (len(group) - 1) // 2,
                "condition_id": round_one[0],
            })
    return selected


def _fit(image: Image.Image, size: int = 256) -> Image.Image:
    return image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)


def build_contact_sheet(selection: list[dict[str, Any]], output_root: Path, destination: Path) -> dict[str, Any]:
    if len(selection) != 6:
        raise ValueError(f"Expected six smoke selections, got {len(selection)}")
    dataset_counts = {name: sum(row["dataset_name"] == name for row in selection)
                      for name in ("artbench", "coco", "mural1")}
    difficulty_counts = {name: sum(row["difficulty"] == name for row in selection)
                         for name in ("Easy", "Hard")}
    if dataset_counts != {"artbench": 2, "coco": 2, "mural1": 2}:
        raise ValueError(f"Invalid smoke dataset quota: {dataset_counts}")
    if difficulty_counts != {"Easy": 3, "Hard": 3}:
        raise ValueError(f"Invalid smoke difficulty quota: {difficulty_counts}")
    cell, gap, label_width, header_height = 256, 10, 120, 32
    columns = ["GT", "Hole mask", "PDDP sketch", "Raw 256", "Composite"]
    width = label_width + len(columns) * (cell + gap) + gap
    row_height = cell + gap
    height = header_height + len(selection) * row_height + gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for column_index, label in enumerate(columns):
        draw.text((label_width + gap + column_index * (cell + gap) + 8, 9), label, fill="black")
    rows_summary: list[dict[str, Any]] = []
    for row_index, selected in enumerate(selection):
        item_dir = output_root / selected["dataset_name"] / "round_001" / safe_id(selected["condition_id"])
        metadata_path = item_dir / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        gt = _fit(Image.open(metadata["inputs"]["gt_rgb"]["path"]))
        native_mask = _fit(Image.open(metadata["inputs"]["hole_mask"]["path"]).convert("L"))
        sketch = _fit(Image.open(item_dir / "pddp_sketch.png"))
        raw = _fit(Image.open(item_dir / "raw_256.png"))
        composite = _fit(Image.open(item_dir / "generated.png"))
        images = [gt, native_mask, sketch, raw, composite]
        y = header_height + row_index * row_height
        draw.text((8, y + 105), f"{selected['dataset_name']}\n{selected['difficulty']}", fill="black")
        for column_index, image in enumerate(images):
            x = label_width + gap + column_index * (cell + gap)
            canvas.paste(image, (x, y))
        flags = {key: value for key, value in metadata["raw_stats"].items()
                 if key in {"finite", "near_constant", "near_all_black", "near_all_white"}}
        rows_summary.append({**selected, "raw_stats": metadata["raw_stats"], "flags": flags,
                             "metadata_path": str(metadata_path)})
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)
    serious = [row for row in rows_summary if not row["flags"]["finite"] or row["flags"]["near_constant"]
               or row["flags"]["near_all_black"] or row["flags"]["near_all_white"]]
    summary = {
        "schema_version": 1,
        "count": len(rows_summary),
        "datasets": dataset_counts,
        "difficulties": difficulty_counts,
        "automated_serious_flag_count": len(serious),
        "contact_sheet": str(destination),
        "rows": rows_summary,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    select_parser = subparsers.add_parser("select")
    select_parser.add_argument("--difficulty-index", type=Path, required=True)
    select_parser.add_argument("--selection-json", type=Path, required=True)
    select_parser.add_argument("--condition-id-file", type=Path, required=True)
    sheet_parser = subparsers.add_parser("sheet")
    sheet_parser.add_argument("--selection-json", type=Path, required=True)
    sheet_parser.add_argument("--output-root", type=Path, required=True)
    sheet_parser.add_argument("--contact-sheet", type=Path, required=True)
    sheet_parser.add_argument("--summary-json", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "select":
        selected = select_conditions(read_jsonl(args.difficulty_index))
        args.selection_json.parent.mkdir(parents=True, exist_ok=True)
        args.selection_json.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
        args.condition_id_file.write_text("\n".join(row["condition_id"] for row in selected) + "\n", encoding="utf-8")
        print(json.dumps(selected, ensure_ascii=False, indent=2))
    else:
        selection = json.loads(args.selection_json.read_text(encoding="utf-8"))
        summary = build_contact_sheet(selection, args.output_root, args.contact_sheet)
        args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
