#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/cpfs01/projects-SSD/cfff-27504eab520e_SSD/zwz_42312/yza/PDDP_sketch_inpainting}"
CONDA_BIN="${CONDA_BIN:-/cpfs01/projects-SSD/cfff-27504eab520e_SSD/zwz_42312/miniconda3/bin/conda}"
ENV_PREFIX="${ENV_PREFIX:-/home/zwz_42312/conda_envs/pddp_sketch_inpainting}"

if [[ -x "${ENV_PREFIX}/bin/python" ]]; then
  echo "Environment already exists: ${ENV_PREFIX}"
  "${ENV_PREFIX}/bin/python" --version
  exit 0
fi

"${CONDA_BIN}" env create -p "${ENV_PREFIX}" -f "${PROJECT_ROOT}/environment.server.yml"
"${ENV_PREFIX}/bin/python" --version
