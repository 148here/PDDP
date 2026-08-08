#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/cpfs01/projects-SSD/cfff-27504eab520e_SSD/zwz_42312/yza/PDDP_sketch_inpainting}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/zwz_42312/PDDPoutputs/smoke/pretrained_canonical_v1}"
CANONICAL="${CANONICAL:-/home/zwz_42312/SketchInpainter_outputs/difficulty_grouped_evaluation_v1/canonical_conditions.jsonl}"
DIFFICULTY_INDEX="${DIFFICULTY_INDEX:-/home/zwz_42312/SketchInpainter_outputs/difficulty_grouped_evaluation_v1/difficulty_index.jsonl}"
PDDP_CKPT="${PDDP_CKPT:-/home/zwz_42312/temp/downloaded_checkpoints/000297e_1343979iter.pth}"
VQ_CKPT="${VQ_CKPT:-/home/zwz_42312/temp/downloaded_checkpoints/last.ckpt}"
PYTHON_BIN="${PYTHON_BIN:-/home/zwz_42312/conda_envs/pddp_sketch_inpainting/bin/python}"

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_ROOT}"
echo "f16d0c6519601840b8f17645a5e3cea048ff135b0bb50c700fc0a97b695dcfe1  ${PDDP_CKPT}" | sha256sum -c -
echo "5cd6c74810ab97e00e942c25403f73afc081e8b19987b31ec0d9ff5b68e7ab14  ${VQ_CKPT}" | sha256sum -c -
"${PYTHON_BIN}" scripts/sketchinpainter/prepare_pretrained_smoke.py select \
  --difficulty-index "${DIFFICULTY_INDEX}" \
  --selection-json "${OUTPUT_ROOT}/selection.json" \
  --condition-id-file "${OUTPUT_ROOT}/condition_ids.txt"
export CUDA_VISIBLE_DEVICES=0
"${PYTHON_BIN}" scripts/sketchinpainter/batch_inference.py \
  --canonical-conditions "${CANONICAL}" \
  --condition-id-file "${OUTPUT_ROOT}/condition_ids.txt" \
  --config configs/sketchinpainter_finetune.yaml \
  --checkpoint "${PDDP_CKPT}" \
  --output-root "${OUTPUT_ROOT}" \
  --device cuda:0 --truncation 0.85 --verify-hashes --execute
"${PYTHON_BIN}" scripts/sketchinpainter/prepare_pretrained_smoke.py sheet \
  --selection-json "${OUTPUT_ROOT}/selection.json" \
  --output-root "${OUTPUT_ROOT}" \
  --contact-sheet "${OUTPUT_ROOT}/contact_sheet.png" \
  --summary-json "${OUTPUT_ROOT}/smoke_summary.json"
