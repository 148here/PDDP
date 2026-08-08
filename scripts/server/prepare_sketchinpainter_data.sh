#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/cpfs01/projects-SSD/cfff-27504eab520e_SSD/zwz_42312/yza/PDDP_sketch_inpainting}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/zwz_42312/PDDPoutputs}"
SKETCH_ROOT="${SKETCH_ROOT:-/cpfs01/projects-SSD/cfff-27504eab520e_SSD/zwz_42312/yza/SketchInpainter}"
ARTBENCH_ROOT="${ARTBENCH_ROOT:-/cpfs01/projects-SSD/cfff-27504eab520e_SSD/zwz_42312/yza/data/artbench/export_512}"
MURAL1_ROOT="${MURAL1_ROOT:-/cpfs01/projects-SSD/cfff-27504eab520e_SSD/zwz_42312/yza/data/mural1}"
MUGE_ROOT="${MUGE_ROOT:-${SKETCH_ROOT}/external/UAED_MuGE}"
MUGE_CKPT="${MUGE_CKPT:-/cpfs01/projects-SSD/cfff-27504eab520e_SSD/zwz_42312/yza/data/temp_for_weight/muge-epoch-19-checkpoint.pth}"
VQ_CKPT="${VQ_CKPT:-/home/zwz_42312/temp/downloaded_checkpoints/last.ckpt}"
MANIFEST="${OUTPUT_ROOT}/preprocessed/training_manifest.jsonl"

cd "${PROJECT_ROOT}"
python scripts/sketchinpainter/build_manifest.py --artbench-root "${ARTBENCH_ROOT}" --mural1-root "${MURAL1_ROOT}" --output-root "${OUTPUT_ROOT}/preprocessed" --manifest "${MANIFEST}"
python scripts/sketchinpainter/precompute_muge_edges.py --manifest "${MANIFEST}" --sketchinpainter-root "${SKETCH_ROOT}" --muge-source-root "${MUGE_ROOT}" --muge-checkpoint "${MUGE_CKPT}" --device cuda:0 --execute --resume
python scripts/sketchinpainter/extract_vq_tokens.py --manifest "${MANIFEST}" --vq-checkpoint "${VQ_CKPT}" --device cuda:0 --execute --resume
