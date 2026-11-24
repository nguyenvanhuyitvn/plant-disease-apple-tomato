#!/usr/bin/env bash
# Tạo bộ gói để chép sang server (chạy được trên macOS lẫn Linux):
#   plantdx-species-linux_code.tar.gz   mã nguồn (bản đã commit)
#   plantdx_data_processed.tar          data/processed – ảnh đã làm sạch + manifest (khoảng 3,8 GB)
#   plantdx_hf_weights.tar              trọng số ImageNet của timm (nếu có trong cache)
#   SHA256SUMS                          để get_data.sh kiểm tra khi giải nén
#
#   bash scripts/make_dist.sh [thư mục đích]      # mặc định ../plantdx-species-linux-dist
set -euo pipefail
cd "$(dirname "$0")/.."
DST=${1:-../plantdx-species-linux-dist}
mkdir -p "$DST"; DST=$(cd "$DST" && pwd)
export COPYFILE_DISABLE=1                                   # macOS: không thêm file ._*
XA=(--exclude .DS_Store --exclude "._*"); tar --version 2>/dev/null | grep -q bsdtar && XA+=(--no-mac-metadata)

git diff --quiet HEAD || echo "CHÚ Ý: còn thay đổi chưa commit – gói mã nguồn chỉ chứa bản đã commit"
git archive --format=tar.gz --prefix=plantdx-species-linux/ -o "$DST/plantdx-species-linux_code.tar.gz" HEAD

[ -f data/processed/species/manifest_apple.csv ] || { echo "thiếu data/processed"; exit 1; }
tar ${XA[@]+"${XA[@]}"} -cf "$DST/plantdx_data_processed.tar" data/processed

HUB="${HF_HOME:-$HOME/.cache/huggingface}/hub"
W=(models--timm--convnext_tiny.in12k_ft_in1k models--timm--mobilenetv3_large_100.ra_in1k models--timm--resnet50.a1_in1k)
if [ -d "$HUB/${W[0]}" ]; then tar ${XA[@]+"${XA[@]}"} --exclude "*/blobs" -chf "$DST/plantdx_hf_weights.tar" -C "$HUB" "${W[@]}"; fi

cd "$DST"
files=(plantdx-species-linux_code.tar.gz plantdx_data_processed.tar)
[ -f plantdx_hf_weights.tar ] && files+=(plantdx_hf_weights.tar)
if command -v sha256sum >/dev/null; then sha256sum "${files[@]}" > SHA256SUMS; else shasum -a 256 "${files[@]}" > SHA256SUMS; fi
ls -lh "${files[@]}" SHA256SUMS
