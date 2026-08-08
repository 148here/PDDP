#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/zwz_42312/PDDPoutputs}"
MANIFEST="${OUTPUT_ROOT}/preprocessed/training_manifest.jsonl"
EXPECTED=52964

test -f "${MANIFEST}"
edge_count="$(find "${OUTPUT_ROOT}/preprocessed/muge_edges" -type f | wc -l)"
token_count="$(find "${OUTPUT_ROOT}/preprocessed/vq_tokens" -type f | wc -l)"
if [[ "${edge_count}" -ne "${EXPECTED}" || "${token_count}" -ne "${EXPECTED}" ]]; then
  echo "Incomplete preprocessing: edges=${edge_count}, tokens=${token_count}, expected=${EXPECTED}" >&2
  exit 1
fi

date -Is > "${OUTPUT_ROOT}/train_control/formal_started_at.txt"
bash scripts/server/train_sketchinpainter_finetune.sh --log_frequency 50
date -Is > "${OUTPUT_ROOT}/train_control/formal_completed_at.txt"
