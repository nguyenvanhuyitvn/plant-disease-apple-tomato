#!/usr/bin/env bash
# Gói cập nhật Tomato-Village cho server đã chạy xong lượt đầu (74 lần):
#   plantdx_tv_update.tar   mã nguồn đã sửa + ảnh TV đã làm sạch + manifest cà chua mới (đường dẫn tính từ gốc project)
#   SHA256SUMS_tv
#
#   bash scripts/make_tv_update.sh [thư mục đích]      # mặc định ../plantdx-species-linux-dist
# Trên server, trong thư mục plantdx-species-linux (chỉ sau khi logs/run.log báo ALL DONE):
#   sha256sum -c SHA256SUMS_tv && tar xf plantdx_tv_update.tar
set -euo pipefail
cd "$(dirname "$0")/.."
DST=${1:-../plantdx-species-linux-dist}
mkdir -p "$DST"; DST=$(cd "$DST" && pwd)
export COPYFILE_DISABLE=1                                   # macOS: không thêm file ._*
XA=(--exclude .DS_Store --exclude "._*"); tar --version 2>/dev/null | grep -q bsdtar && XA+=(--no-mac-metadata)

SP=data/processed/species
FILES=(
  src/plantdx/data.py
  src/plantdx/calibrate.py
  src/plantdx/summarize.py
  src/plantdx/eval_extra.py
  scripts/run_experiments.sh
  scripts/prepare_tomato_village.py
  scripts/make_tv_update.sh
  app/streamlit_app.py
  README.md
  data/processed/tomatovillage
  $SP/manifest_tomato.csv
  $SP/manifest_tomato_base.csv
  $SP/manifest_tomato_village_ood.csv
  $SP/stats_tomato.csv
  $SP/qa/tv_near_pairs.csv
  $SP/qa/tv_cross_matches.csv
  $SP/qa/tv_removed.csv
  $SP/qa/samples_tomatovillage.jpg
)
for f in "${FILES[@]}"; do [ -e "$f" ] || { echo "thiếu $f (chạy scripts/prepare_tomato_village.py trước)"; exit 1; }; done
grep -q tomatovillage $SP/manifest_tomato.csv || { echo "manifest_tomato.csv chưa có TV"; exit 1; }

tar ${XA[@]+"${XA[@]}"} -cf "$DST/plantdx_tv_update.tar" "${FILES[@]}"
cd "$DST"
if command -v sha256sum >/dev/null; then sha256sum plantdx_tv_update.tar > SHA256SUMS_tv; else shasum -a 256 plantdx_tv_update.tar > SHA256SUMS_tv; fi
ls -lh plantdx_tv_update.tar SHA256SUMS_tv
