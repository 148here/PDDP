"""CPU-only import and configuration validation for the phase-1 setup."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sketchinpainter-root", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    import cv2
    import torch
    import torchvision
    from image_synthesis.data.sketchinpainter_dataset import crop_and_pad_sketch
    from image_synthesis.utils.io import load_yaml_config

    sys.path.insert(0, str(args.sketchinpainter_root.resolve()))
    from dataset.makesketch import make_sketch_from_edge

    config = load_yaml_config("configs/sketchinpainter_finetune.yaml")
    report = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "opencv": cv2.__version__,
        "cuda_visible": torch.cuda.is_available(),
        "config_model": config["model"]["target"],
        "dataset_target": config["dataloader"]["train_datasets"][0]["target"],
        "sketch_callable": callable(make_sketch_from_edge),
        "crop_callable": callable(crop_and_pad_sketch),
    }
    print(json.dumps(report, indent=2))
    if report["cuda_visible"]:
        raise RuntimeError("CPU validation unexpectedly sees a CUDA device")


if __name__ == "__main__":
    main()
