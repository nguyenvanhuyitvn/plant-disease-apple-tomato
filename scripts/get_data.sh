#!/usr/bin/env bash
# Kiểm tra checksum và giải nén gói dữ liệu vào project.
#
#   bash scripts/get_data.sh /đường/dẫn/thư-mục-chứa-gói
#
# Thư mục chứa gói cần có: plantdx_data_processed.tar, SHA256SUMS và (tuỳ chọn) plantdx_hf_weights.tar.
set -euo pipefail
cd "$(dirname "$0")/.."
SRC=${1:?"cách dùng: bash scripts/get_data.sh <thư mục chứa gói>"}
SRC=$(cd "$SRC" && pwd)

echo "== Kiểm tra checksum"
(cd "$SRC" && sha256sum --ignore-missing -c SHA256SUMS)

echo "== Giải nén dữ liệu -> data/processed"
tar -xf "$SRC/plantdx_data_processed.tar" --no-same-owner
echo "   $(find data/processed -type f | wc -l) file"

if [ -f "$SRC/plantdx_hf_weights.tar" ]; then
  HUB="${HF_HOME:-$HOME/.cache/huggingface}/hub"
  echo "== Trọng số pretrained -> $HUB"
  mkdir -p "$HUB" && tar -xf "$SRC/plantdx_hf_weights.tar" -C "$HUB" --no-same-owner
fi
echo "Xong. Kiểm tra: PYTHONPATH=src .venv/bin/python scripts/check_env.py"
