#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/cpfs01/projects-SSD/cfff-27504eab520e_SSD/zwz_42312/yza/PDDP_sketch_inpainting}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/zwz_42312/PDDPoutputs}"
CANONICAL="${CANONICAL:-/home/zwz_42312/SketchInpainter_outputs/difficulty_grouped_evaluation_v1/canonical_conditions.jsonl}"
CHECKPOINT="${CHECKPOINT:?set CHECKPOINT to the fine-tuned PDDP checkpoint}"
RUN_NAME="${RUN_NAME:-pddp_sketchinpainter_finetuned}"

cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES=0
python scripts/sketchinpainter/batch_inference.py --canonical-conditions "${CANONICAL}" --config configs/sketchinpainter_finetune.yaml --checkpoint "${CHECKPOINT}" --output-root "${OUTPUT_ROOT}/inference/${RUN_NAME}" --device cuda:0 --sketch-scope bbox_crop --bbox-scale 1.2 --verify-hashes --execute "$@"
