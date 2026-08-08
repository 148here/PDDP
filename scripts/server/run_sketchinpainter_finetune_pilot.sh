#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/cpfs01/projects-SSD/cfff-27504eab520e_SSD/zwz_42312/yza/PDDP_sketch_inpainting}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/zwz_42312/PDDPoutputs}"
CONTROL_ROOT="${OUTPUT_ROOT}/train_control"

cd "${PROJECT_ROOT}"
mkdir -p "${CONTROL_ROOT}"
date -Is > "${CONTROL_ROOT}/pipeline_started_at.txt"
bash scripts/server/prepare_sketchinpainter_data.sh
date -Is > "${CONTROL_ROOT}/preprocessing_completed_at.txt"
bash scripts/server/train_sketchinpainter_finetune.sh --log_frequency 10
date -Is > "${CONTROL_ROOT}/pilot_completed_at.txt"
