#!/usr/bin/env bash
# Cài môi trường trên Ubuntu 24.04 + GPU NVIDIA (Python 3.12 của hệ thống).
#
#   bash scripts/setup.sh
#
# torch 2.15.0 trên PyPI đi kèm CUDA 13.2 (đã gồm cuDNN, cuBLAS...): KHÔNG cần cài CUDA Toolkit,
# chỉ cần driver NVIDIA >= 580 (GPU Blackwell như RTX PRO / RTX 50xx cần driver mới này).
set -euo pipefail
cd "$(dirname "$0")/.."
PYBIN=${PYBIN:-python3}
MIN_DRIVER=580

echo "== Hệ điều hành"
grep -E '^(PRETTY_NAME)=' /etc/os-release || true

echo "== Python"
$PYBIN -c 'import sys; v=sys.version_info; print(sys.version.split()[0]); assert v >= (3, 12), "cần Python >= 3.12"'
if ! $PYBIN -c 'import ensurepip, venv' 2>/dev/null; then
  echo "Thiếu gói venv. Chạy: sudo apt install python3.12-venv   rồi chạy lại script này."; exit 1
fi

echo "== Driver NVIDIA"
if ! command -v nvidia-smi >/dev/null; then
  echo "Không thấy nvidia-smi. Cài driver: sudo ubuntu-drivers install   (hoặc sudo apt install nvidia-driver-580-open), rồi khởi động lại."
  exit 1
fi
nvidia-smi --query-gpu=index,name,driver_version,memory.total,compute_cap --format=csv,noheader
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
if [ "$DRV" -lt "$MIN_DRIVER" ]; then
  echo "Driver $DRV < $MIN_DRIVER: torch (CUDA 13.2) sẽ không dùng được GPU."
  echo "Nâng driver: sudo apt install nvidia-driver-580-open (hoặc mới hơn), rồi khởi động lại."
  exit 1
fi

echo "== Môi trường ảo .venv"
[ -d .venv ] || $PYBIN -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

echo "== Kiểm tra"
PYTHONPATH=src .venv/bin/python scripts/check_env.py
