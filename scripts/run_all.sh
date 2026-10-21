#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
config_path=${1:-configs/default.yaml}
python src/prepare_data.py --config "$config_path"
python src/extract_clip.py --config "$config_path"
python src/train_diffusion.py --config "$config_path" --direction both
python src/train_diffusion.py --config "$config_path" --direction both --unconditional
python src/run_experiments.py --config "$config_path" --exp all
python src/plot_results.py --config "$config_path"
