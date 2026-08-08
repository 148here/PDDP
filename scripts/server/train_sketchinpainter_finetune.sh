#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/cpfs01/projects-SSD/cfff-27504eab520e_SSD/zwz_42312/yza/PDDP_sketch_inpainting}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/zwz_42312/PDDPoutputs}"
PDDP_CKPT="${PDDP_CKPT:-/home/zwz_42312/temp/downloaded_checkpoints/000297e_1343979iter.pth}"

cd "${PROJECT_ROOT}"
test -f "${PDDP_CKPT}"
mkdir -p "${OUTPUT_ROOT}/train"
export CUDA_VISIBLE_DEVICES=0
python train.py --config_file configs/sketchinpainter_finetune.yaml --name sketchinpainter_finetune --output "${OUTPUT_ROOT}/train" --load_path "${PDDP_CKPT}" --gpu 0 --seed 20260425 --cudnn_deterministic "$@"
