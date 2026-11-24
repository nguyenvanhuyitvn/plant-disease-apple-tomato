#!/usr/bin/env bash
# Thực nghiệm (bản Linux + GPU NVIDIA): táo, cà chua; teacher ConvNeXt-T -> student MobileNetV3-L.
# 6 thiết lập, 74 lần chạy (8 teacher/tham chiếu + 66 MobileNetV3). Chạy lại được: lần chạy đã có metrics.json sẽ bị bỏ qua.
#
#   bash scripts/run_experiments.sh           # chạy thật; dừng rồi chạy lại được
#   nohup bash scripts/run_experiments.sh > logs/run.log 2>&1 &     # chạy nền, thoát SSH vẫn chạy
#   QUICK=1 bash scripts/run_experiments.sh   # chạy thử toàn bộ luồng trên ít ảnh (khoảng 15 phút) -> runs_smoke/
#
# Biến môi trường (tuỳ chọn):
#   PY=python                  trình thông dịch Python (mặc định .venv/bin/python; dùng PY=python với conda)
#   WORKERS=8                  số tiến trình đọc ảnh mỗi lần huấn luyện (mặc định 8; tăng nếu GPU chờ CPU)
#   AMP=0                      tắt bfloat16 tự động (mặc định bật)
#   SEEDS="0 1 2"              seed của MobileNetV3 ở thiết lập chính (A-all, T-all)
#   SEEDS_AUX="0 1 2"          seed ở thiết lập phụ (A-LODO-*, T0); SEEDS_AUX=0 để rút gọn
#   ONLY="A-all T-all"         chỉ chạy các thiết lập này (vd. chia cho nhiều GPU bằng CUDA_VISIBLE_DEVICES)
#   SKIP_POST=1                bỏ bước hiệu chỉnh / tổng hợp / xuất mô hình (chạy riêng sau khi mọi GPU xong)
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
export PYTHONPATH=src PYTHONWARNINGS=ignore

OUT=runs; EXTRA=()
if [ "${QUICK:-0}" = 1 ]; then
  OUT=runs_smoke
  EXTRA=(--limit 48 --epochs 1 --samples-per-epoch 128 --batch 32 --workers 2)
fi
SEEDS=${SEEDS:-"0 1 2"}
SEEDS_AUX=${SEEDS_AUX:-"0 1 2"}
ONLY=${ONLY:-}
MN=mobilenetv3_large_100

TRAIN_OPTS=(--workers "${WORKERS:-8}")
[ "${AMP:-1}" = 0 ] && TRAIN_OPTS+=(--no-amp)
mkdir -p logs
train() { $PY -m plantdx.train --out "$OUT" "${TRAIN_OPTS[@]}" ${EXTRA[@]+"${EXTRA[@]}"} "$@"; }

# setting  species  train-domains  main(1 = có đủ ablation + ResNet-50)
SETTINGS=(
  "A-all      apple  pv,pp,pd  1"
  "T-all      tomato pv,pd     1"
  "A-LODO-PD  apple  pv,pp     0"
  "A-LODO-PP  apple  pv,pd     0"
  "A-LODO-PV  apple  pp,pd     0"   # không có ảnh PV -> không có cặp nền, "proposed" chỉ còn KD
  "T0         tomato pv        0"   # PV -> PlantDoc (kiểu bài báo gốc)
)

for row in "${SETTINGS[@]}"; do
  read -r SET SP DOMS MAIN <<< "$row"
  if [ -n "$ONLY" ] && [[ " $ONLY " != *" $SET "* ]]; then continue; fi
  common=(--species "$SP" --setting "$SET" --train-domains "$DOMS")
  # 1) teacher / mô hình tham chiếu (1 seed)
  train "${common[@]}" --model convnext_tiny --config convnext_tiny --seed 0
  [ "$MAIN" = 1 ] && train "${common[@]}" --model resnet50 --config resnet50 --seed 0
  TEACHER="$OUT/$SP/$SET/convnext_tiny/seed0/best.pt"
  # 2) student MobileNetV3-L
  if [ "$MAIN" = 1 ]; then SEED_LIST=$SEEDS; else SEED_LIST=$SEEDS_AUX; fi
  for S in $SEED_LIST; do
    train "${common[@]}" --model $MN --config mnv3_erm      --seed "$S"
    train "${common[@]}" --model $MN --config mnv3_mixstyle --seed "$S" --mixstyle
    if [ "$MAIN" = 1 ]; then
      train "${common[@]}" --model $MN --config mnv3_kd   --seed "$S" --teacher "$TEACHER" --alpha 0.5
      train "${common[@]}" --model $MN --config mnv3_cons --seed "$S" --beta 1
    fi
    train "${common[@]}" --model $MN --config proposed --seed "$S" --teacher "$TEACHER" --alpha 0.5 --beta 1
  done
done

if [ "${SKIP_POST:-0}" = 1 ]; then echo "TRAIN DONE (bỏ qua bước sau huấn luyện)"; exit 0; fi

# 3) hiệu chỉnh + lọc ảnh lạ cho mọi lần chạy
OODMAX=2000; [ "${QUICK:-0}" = 1 ] && OODMAX=64
for d in "$OUT"/*/*/*/seed*; do
  [ -f "$d/metrics.json" ] && [ ! -f "$d/calib.json" ] && $PY -m plantdx.calibrate --run "$d" --ood-max $OODMAX
done

# 4) tổng hợp bảng
RES=results; [ "${QUICK:-0}" = 1 ] && RES=results_smoke
$PY -m plantdx.summarize --runs "$OUT" --out "$RES"

# 5) xuất mô hình triển khai (đề xuất, thiết lập dùng cả các miền, seed 0)
if [ "${QUICK:-0}" = 1 ]; then DEP=(--out models_smoke --calib 32 --ood-max 64); else DEP=(); fi
for ck in "$OUT/apple/A-all/proposed/seed0" "$OUT/tomato/T-all/proposed/seed0"; do
  if [ -f "$ck/metrics.json" ]; then $PY -m plantdx.deploy --ckpt "$ck/best.pt" ${DEP[@]+"${DEP[@]}"}; fi
done
echo "ALL DONE"
