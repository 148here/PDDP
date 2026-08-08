"""Extract per-sample 32x32 VQ tokens with collision-safe paths."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--vq-checkpoint", type=Path, required=True)
    parser.add_argument("--vq-config", type=Path, default=Path("configs/vqvae_openimages.yaml"))
    parser.add_argument("--mapping", type=Path, default=Path("help_folder/statistics/taming_vqvae_2887.pt"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    with args.manifest.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    missing = [row for row in rows if not Path(row["token_path"]).is_file()]
    print(json.dumps({"conditions": len(rows), "missing_tokens": len(missing), "execute": args.execute}, indent=2))
    if not args.execute:
        return
    if not args.vq_checkpoint.is_file():
        raise FileNotFoundError(args.vq_checkpoint)
    import numpy as np
    import torch
    from PIL import Image
    from image_synthesis.modeling.codecs.image_codec.taming_gumbel_vqvae import TamingGumbelVQVAE

    codec = TamingGumbelVQVAE(
        config_path=str(args.vq_config), ckpt_path=str(args.vq_checkpoint), mapping_path=str(args.mapping)
    ).to(args.device).eval()
    for index, row in enumerate(rows, 1):
        output = Path(row["token_path"])
        if args.resume and output.is_file():
            continue
        image = Image.open(row["image_path"]).convert("RGB").resize((256, 256), Image.Resampling.BICUBIC)
        tensor = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1).unsqueeze(0).to(args.device)
        with torch.no_grad():
            tokens = codec.get_tokens(tensor)["token"].reshape(-1).cpu().numpy().astype(np.int64)
        if tokens.size != 1024:
            raise ValueError(f"Expected 1024 tokens for {row['sample_id']}, got {tokens.size}")
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output, tokens, allow_pickle=False)
        if index % 100 == 0:
            print(f"{index}/{len(rows)}")


if __name__ == "__main__":
    main()
