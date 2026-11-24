# plantdx-species-linux – chẩn đoán bệnh lá táo / cà chua (bản cho server Linux + GPU NVIDIA)

Teacher **ConvNeXt-Tiny** chưng cất sang student **MobileNetV3-Large**, thêm **loss nhất quán nền**, hiệu chỉnh
độ tin cậy và lọc ảnh lạ. Dữ liệu: PlantVillage, Plant Pathology 2020 + 2021, PlantDoc. Mô hình xuất ONNX INT8
(khoảng 5 MB), xem kết quả bằng **Streamlit**.

Bản này dành cho **Ubuntu 24.04, Python 3.12, GPU NVIDIA** (đã nhắm tới GPU Blackwell, mã thiết bị 2bb3).
So với bản `plantdx-species` (Mac), bản Linux có thêm:
- **GPU NVIDIA:** ưu tiên CUDA, bật TF32, cuDNN benchmark và channels_last.
- **bfloat16 tự động (AMP)** khi huấn luyện và suy luận; tắt bằng `AMP=0`.
- **Script cho server:** cài đặt (`setup.sh`), kiểm tra môi trường (`check_env.py`), giải nén dữ liệu có
  kiểm tra checksum (`get_data.sh`).

Thuật toán, siêu tham số và cách chia dữ liệu giữ nguyên như bản Mac.

```
                 ┌─ Teacher ConvNeXt-Tiny (fine-tune trên đúng các miền huấn luyện, rồi đóng băng)
ảnh x ───────────┤                                                  │ logits/T
(lấy mẫu cân     └─ Student MobileNetV3-Large ── CE(y) ── KD(T=4) ──┘
 bằng miền×lớp)              │
ảnh x_nền (lá PV trên nền ───┘── nhất quán nền KL(sg[p(x)] ‖ p(x_nền))
thực địa tổng hợp)
loss = (1−α)·CE + α·T²·KD + β·Cons      (α = 0,5; T = 4; β = 1, tăng dần trong epoch đầu)
→ hiệu chỉnh T*, ngưỡng năng lượng τ_E (ảnh lạ), ngưỡng tin cậy τ_c (chưa chắc) → ONNX INT8 → Streamlit
```

## 1. Yêu cầu

| | |
|---|---|
| Hệ điều hành | Ubuntu 24.04 (các bản Linux x86_64 khác có glibc ≥ 2.28 cũng được) |
| Python | 3.12–3.14, kèm gói `python3.12-venv` |
| GPU | NVIDIA, **driver ≥ 580**. torch 2.15.0 trên PyPI kèm sẵn CUDA 13.2 và cuDNN, nên **không cần cài CUDA Toolkit** |
| Ổ đĩa | khoảng 10 GB: 3,8 GB dữ liệu, khoảng 4 GB thư viện, phần còn lại cho checkpoint |

## 2. Cài đặt

Chép lên server ba thứ: thư mục project này (hoặc `plantdx-species-linux_code.tar.gz`), gói dữ liệu
`plantdx_data_processed.tar` cùng `SHA256SUMS`, và (tuỳ chọn) `plantdx_hf_weights.tar`.

```bash
sudo apt install -y python3.12-venv            # nếu chưa có
tar xzf plantdx-species-linux_code.tar.gz && cd plantdx-species-linux
bash scripts/get_data.sh ../                   # kiểm tra checksum, giải nén data/processed và trọng số pretrained
bash scripts/setup.sh                          # kiểm tra driver, tạo .venv, cài thư viện, chạy check_env.py
```

`check_env.py` in ra GPU, compute capability, khả năng bfloat16 và tốc độ nhân ma trận. Nó cũng kiểm tra đủ
128.988 ảnh trong manifest và trọng số pretrained đã có chưa. Script thoát với mã 1 nếu còn lỗi.

## 3. Chạy thí nghiệm

```bash
QUICK=1 bash scripts/run_experiments.sh && rm -rf runs_smoke results_smoke models_smoke   # chạy thử vài phút

nohup bash scripts/run_experiments.sh > logs/run.log 2>&1 &     # chạy thật, thoát SSH vẫn chạy
tail -f logs/run.log                                            # theo dõi
nvidia-smi -l 5                                                 # mức dùng GPU
```

Dừng giữa chừng rồi chạy lại vẫn được: lần chạy nào đã có `metrics.json` sẽ được bỏ qua.

| Biến môi trường | Ý nghĩa |
|---|---|
| `SEEDS="0 1 2"` | seed của MobileNetV3 ở thiết lập chính A-all, T-all |
| `SEEDS_AUX="0 1 2"` | seed ở thiết lập phụ A-LODO-*, T0; `SEEDS_AUX=0` để rút gọn |
| `ONLY="A-all T-all"` | chỉ chạy các thiết lập này |
| `SKIP_POST=1` | chỉ huấn luyện, bỏ bước hiệu chỉnh / tổng hợp / xuất mô hình |
| `WORKERS=8` | số tiến trình đọc ảnh; tăng lên 12–16 nếu GPU dưới 80% mà CPU còn rảnh |
| `AMP=0` | tắt bfloat16 (FP32/TF32) |
| `PY=python` | dùng Python của conda hoặc hệ thống thay cho `.venv` |

Nhiều GPU: mỗi GPU chạy một nhóm thiết lập, cuối cùng chạy bước sau huấn luyện.

```bash
CUDA_VISIBLE_DEVICES=0 ONLY="A-all A-LODO-PD A-LODO-PV" SKIP_POST=1 nohup bash scripts/run_experiments.sh > logs/g0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 ONLY="T-all T0 A-LODO-PP"        SKIP_POST=1 nohup bash scripts/run_experiments.sh > logs/g1.log 2>&1 &
bash scripts/run_experiments.sh        # khi cả hai xong: hiệu chỉnh, tổng hợp, xuất mô hình
```

### Thiết lập thí nghiệm

| Thiết lập | Loài | Huấn luyện | Cấu hình |
|---|---|---|---|
| A-all | táo | PV + PP + PD | ConvNeXt-T, ResNet-50, MNv3 ERM / MixStyle / KD / nhất quán nền / đề xuất |
| T-all | cà chua | PV + PD | như trên |
| A-LODO-PD, A-LODO-PP, A-LODO-PV | táo | bỏ một miền | ConvNeXt-T, MNv3 ERM / MixStyle / đề xuất |
| T0 | cà chua | PV | ConvNeXt-T, MNv3 ERM / MixStyle / đề xuất |

- MNv3 chạy 3 seed; ConvNeXt-T và ResNet-50 chạy 1 seed. Tổng 74 lần chạy.
- Mỗi lần chạy: 10 epoch × 7.680 ảnh, batch 64, AdamW lr 5e-4, cosine.
- A-LODO-PV không có ảnh PlantVillage nên không có cặp ảnh nền: cấu hình "đề xuất" ở thiết lập này chỉ còn KD.

## 4. Kết quả và ứng dụng

```
runs/<loài>/<thiết lập>/<cấu hình>/seed<n>/   best.pt, history.csv, metrics.json, calib.json, logits test
results/        runs.csv, summary.csv, summary.md (F1 trung bình ± độ lệch chuẩn)
models/<loài>/  model.onnx (INT8), model_fp32.onnx, labels.txt, meta.json
```

Mang kết quả về máy cá nhân (không cần mang `best.pt` nếu chỉ xem bảng và dùng app):

```bash
rsync -av --exclude 'best.pt' <user>@<server>:~/plantdx-species-linux/{runs,results,models} ./
```

Streamlit trên server (cấu hình trong `.streamlit/config.toml`, chỉ nghe trên localhost):

```bash
.venv/bin/streamlit run app/streamlit_app.py      # trên server
ssh -L 8501:localhost:8501 <user>@<server>        # trên máy cá nhân, rồi mở http://localhost:8501
```

## 5. Cấu trúc

```
plantdx-species-linux/
├── scripts/
│   ├── setup.sh             cài môi trường (kiểm tra driver ≥ 580, .venv, requirements)
│   ├── check_env.py         kiểm tra GPU / CUDA / bfloat16 / dữ liệu / trọng số
│   ├── get_data.sh          kiểm tra SHA256 + giải nén gói dữ liệu
│   ├── run_experiments.sh   toàn bộ thí nghiệm + hiệu chỉnh + tổng hợp + xuất mô hình
│   ├── prepare_data.py      (chỉ khi làm sạch lại từ dữ liệu gốc) bước 1: PlantVillage + PlantDoc
│   ├── prepare_species.py   (chỉ khi làm sạch lại) bước 2: táo + cà chua + Plant Pathology
│   └── class_mapping.csv
├── src/plantdx/   data.py, background.py, models.py, train.py, calibrate.py, summarize.py, deploy.py, metrics.py
├── app/           streamlit_app.py, inference.py (ONNX Runtime, không cần PyTorch)
├── data/processed/   ảnh đã làm sạch + manifest (từ gói dữ liệu, không đưa vào git)
└── .streamlit/config.toml
```

Huấn luyện chỉ cần `data/processed`. Muốn làm sạch lại từ đầu thì cần thêm dữ liệu gốc trong `data/raw/` (khoảng
19 GB, gồm PlantVillage, PlantDoc, Plant Pathology), rồi chạy `prepare_data.py` và `prepare_species.py`. Quy trình
làm sạch được mô tả trong `BaoCao/LamSachDuLieu_TaoCaChua.docx`.

## 6. Lỗi thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| `torch.cuda.is_available()` = False | `nvidia-smi` phải chạy được; driver ≥ 580; khởi động lại sau khi cài driver |
| `no kernel image is available` | bản torch không có mã cho GPU: kiểm tra `check_env.py`, cài đúng `torch==2.15.0` |
| `DataLoader worker ... killed` / `shm` | trong Docker thêm `--shm-size=8g`; hoặc giảm `WORKERS` |
| GPU dùng dưới 50% | tăng `WORKERS`; đặt dữ liệu trên SSD cục bộ thay vì ổ mạng |
| Hết bộ nhớ GPU | không xảy ra với batch 64 trên GPU ≥ 16 GB; nếu GPU nhỏ hơn, giảm `--batch` trong script |

## Dữ liệu và giấy phép

- PlantVillage – Mohanty, Hughes & Salathé (2016), CC BY-SA 3.0
- PlantDoc – Singh et al. (2020), CC BY 4.0
- Plant Pathology 2020 / 2021 – Thapa et al. (2020, 2021), cuộc thi FGVC7 / FGVC8 trên Kaggle; dùng theo điều lệ
  cuộc thi (nghiên cứu, phi thương mại)
