"""Kiểm tra server trước khi chạy thí nghiệm: GPU, CUDA, bfloat16, dữ liệu, trọng số pretrained.

  PYTHONPATH=src .venv/bin/python scripts/check_env.py
Thoát với mã 1 nếu có lỗi khiến thí nghiệm không chạy được.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
errors, warns = [], []


def ok(msg):
    print(f"  [OK]   {msg}")


def warn(msg):
    warns.append(msg)
    print(f"  [CHÚ Ý] {msg}")


def err(msg):
    errors.append(msg)
    print(f"  [LỖI]  {msg}")


print("== PyTorch / GPU")
import torch  # noqa: E402

ok(
    f"torch {torch.__version__}, CUDA build {torch.version.cuda}, cuDNN {torch.backends.cudnn.version()}"
)
if not torch.cuda.is_available():
    err("torch không thấy GPU (kiểm tra driver NVIDIA >= 580 và `nvidia-smi`)")
else:
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        ok(
            f"GPU {i}: {p.name}, {p.total_memory / 2**30:.0f} GB, compute capability {p.major}.{p.minor}"
        )
        arch = f"sm_{p.major}{p.minor}"
        if (
            arch not in torch.cuda.get_arch_list()
            and f"compute_{p.major}{p.minor}" not in torch.cuda.get_arch_list()
        ):
            err(
                f"bản torch này không có mã cho {arch} (có: {torch.cuda.get_arch_list()})"
            )
    if torch.cuda.is_bf16_supported():
        ok("hỗ trợ bfloat16 (AMP)")
    else:
        warn("GPU không hỗ trợ bfloat16: chạy với AMP=0")
    x = torch.randn(4096, 4096, device="cuda")
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(20):
        x @ x
    torch.cuda.synchronize()
    ok(
        f"nhân ma trận 4096² FP32/TF32: {20 * 2 * 4096**3 / (time.perf_counter() - t) / 1e12:.1f} TFLOPS"
    )

print("== CPU / bộ nhớ chia sẻ")
n = os.cpu_count() or 1
ok(f"{n} CPU") if n >= 8 else warn(f"chỉ {n} CPU: đặt WORKERS={max(1, n - 1)}")
shm = Path("/dev/shm")
if shm.exists():
    st = os.statvfs(shm)
    gb = st.f_bsize * st.f_blocks / 2**30
    ok(f"/dev/shm {gb:.0f} GB") if gb >= 4 else warn(
        f"/dev/shm chỉ {gb:.1f} GB (Docker: thêm --shm-size=8g)"
    )

print("== Dữ liệu")
sp = ROOT / "data" / "processed" / "species"
need = [
    sp / "manifest_apple.csv",
    sp / "manifest_tomato.csv",
    ROOT / "data" / "processed" / "manifest.csv",
]
missing = [str(p.relative_to(ROOT)) for p in need if not p.exists()]
if missing:
    err(
        f"thiếu {missing} – giải nén gói dữ liệu: bash scripts/get_data.sh <thư mục chứa gói>"
    )
else:
    import pandas as pd

    bad = 0
    for f in need:
        m = pd.read_csv(f, keep_default_na=False)
        paths = [p for c in ["path", "mask_path"] if c in m for p in m[c] if p]
        lost = [p for p in paths if not (ROOT / "data" / p).exists()]
        bad += len(lost)
        (ok if not lost else err)(
            f"{f.name}: {len(paths)} ảnh, thiếu {len(lost)}"
            + (f" (vd. {lost[0]})" if lost else "")
        )

print("== Trọng số pretrained (timm / Hugging Face)")
hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
for w in [
    "convnext_tiny.in12k_ft_in1k",
    "mobilenetv3_large_100.ra_in1k",
    "resnet50.a1_in1k",
]:
    if (hub / f"models--timm--{w}").exists():
        ok(f"đã có {w}")
    else:
        warn(
            f"chưa có {w}: sẽ tự tải khi chạy (cần Internet), hoặc giải nén plantdx_hf_weights.tar"
        )

print()
if errors:
    print(f"CÓ {len(errors)} LỖI – sửa trước khi chạy thí nghiệm.")
    sys.exit(1)
print(
    "Sẵn sàng."
    + (f" ({len(warns)} chú ý)" if warns else "")
    + " Chạy thử: QUICK=1 bash scripts/run_experiments.sh"
)
