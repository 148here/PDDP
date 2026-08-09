#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/zwz_42312/PDDPoutputs}"
MANIFEST="${OUTPUT_ROOT}/preprocessed/training_manifest.jsonl"
PYTHON_BIN="${PYTHON_BIN:-/home/zwz_42312/conda_envs/pddp_sketch_inpainting/bin/python}"

test -f "${MANIFEST}"
"${PYTHON_BIN}" scripts/sketchinpainter/validate_preprocessed_manifest.py --manifest "${MANIFEST}"

date -Is > "${OUTPUT_ROOT}/train_control/formal_started_at.txt"
bash scripts/server/train_sketchinpainter_finetune.sh --log_frequency 50
date -Is > "${OUTPUT_ROOT}/train_control/formal_completed_at.txt"
