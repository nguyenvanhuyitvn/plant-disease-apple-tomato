"""Xuất mô hình theo loài để triển khai (Streamlit, Android): ONNX FP32 + INT8 hỗn hợp, kèm
hiệu chỉnh (T*, τ_E, τ_c) tính lại trên chính mô hình INT8 được triển khai.

  python -m plantdx.deploy --ckpt runs/apple/A-all/proposed/seed0/best.pt
Kết quả: models/<loài>/ {model.onnx, model_fp32.onnx, labels.txt, meta.json}
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                      quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process

from .calibrate import evaluate_calibration, fit_calibration
from .data import ROOT, VI_NAMES, PairDataset, build_transforms, domains_of, load_species, ood_sets, select
from .models import load_ckpt

MODELS = ROOT / "models"


def to_numpy_batches(df, tf, batch=32):
    ds = PairDataset(df, tf)
    for i in range(0, len(ds), batch):
        xs, ys = zip(*(ds[j] for j in range(i, min(i + batch, len(ds)))), strict=True)
        yield torch.stack(xs).numpy(), np.array(ys)


class Calib(CalibrationDataReader):
    """Cấp từng ảnh cho bước hiệu chỉnh lượng tử hoá INT8."""

    def __init__(self, df, tf, input_name):
        self.it = ({input_name: x[k:k + 1]} for x, _ in to_numpy_batches(df, tf) for k in range(len(x)))

    def get_next(self):
        return next(self.it, None)


def latency(path, size, threads, runs=200, warmup=20):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    name = s.get_inputs()[0].name
    x = np.random.rand(1, 3, size, size).astype(np.float32)
    for _ in range(warmup):
        s.run(None, {name: x})
    t = []
    for _ in range(runs):
        t0 = time.perf_counter()
        s.run(None, {name: x})
        t.append((time.perf_counter() - t0) * 1000)
    return {"mean_ms": float(np.mean(t)), "p95_ms": float(np.percentile(t, 95))}


def onnx_logits(path, df, tf):
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out = [s.run(None, {"input": x})[0] for x, _ in to_numpy_batches(df, tf, 64)]
    return np.concatenate(out) if out else np.zeros((0, 1), np.float32)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--calib", type=int, default=384, help="số ảnh hiệu chỉnh INT8")
    ap.add_argument("--target-acc", type=float, default=0.9)
    ap.add_argument("--ood-max", type=int, default=1500)
    ap.add_argument("--out", default=str(MODELS))
    args = ap.parse_args(argv)

    model, ck = load_ckpt(args.ckpt)
    sp, classes = ck["species"], ck["classes"]
    size = ck["args"]["img_size"]
    train_doms = ck["args"]["train_domains"].split(",")
    out = Path(args.out) / sp
    out.mkdir(parents=True, exist_ok=True)
    tf = build_transforms(size, ck["mean"], ck["std"], False)
    m = load_species(sp)

    fp32 = out / "model_fp32.onnx"
    torch.onnx.export(model, (torch.randn(1, 3, size, size),), str(fp32), input_names=["input"],
                      output_names=["logits"], dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=17, dynamo=False)
    pre = out / "_pre.onnx"
    quant_pre_process(str(fp32), str(pre))
    g = onnx.load(str(pre)).graph
    convs = [n.name for n in g.node if n.op_type in ("Conv", "Gemm")]
    keep_fp32 = sorted({convs[0], convs[-1]} | {n.name for n in g.node if n.op_type == "Gemm"})
    cal_df = select(m, train_doms, "train").sample(frac=1, random_state=0).groupby("dataset").head(
        args.calib // len(train_doms))
    q = out / "model.onnx"
    quantize_static(str(pre), str(q), Calib(cal_df, tf, "input"), quant_format=QuantFormat.QDQ, per_channel=True,
                    weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8,
                    calibrate_method=CalibrationMethod.Percentile, op_types_to_quantize=["Conv", "Gemm"],
                    nodes_to_exclude=keep_fp32,
                    extra_options={"CalibPercentile": 99.99, "CalibMaxIntermediateOutputs": 50})
    pre.unlink()

    # hiệu chỉnh trên chính mô hình INT8
    val = select(m, train_doms, "val")
    ood_v = ood_sets(sp, "val").sample(frac=1, random_state=0).head(args.ood_max)
    ood_t = ood_sets(sp, "test").sample(frac=1, random_state=0).head(args.ood_max)
    cal = fit_calibration(onnx_logits(q, val, tf), val.label.to_numpy(), onnx_logits(q, ood_v, tf), args.target_acc)
    t_lg, t_y = {}, {}
    for d in domains_of(sp):
        df = select(m, [d], "test")
        t_lg[d], t_y[d] = onnx_logits(q, df, tf), df.label.to_numpy()
    cal["test"] = evaluate_calibration(cal, t_lg, t_y, onnx_logits(q, ood_t, tf))

    meta = {"species": sp, "classes": classes, "names_vi": [VI_NAMES[c] for c in classes],
            "model": ck["args"]["model"], "config": ck["args"]["config"], "setting": ck["args"]["setting"],
            "source_ckpt": os.path.relpath(Path(args.ckpt).resolve(), ROOT),
            "img_size": size, "resize": int(size * 1.14), "mean": ck["mean"], "std": ck["std"],
            "size_mb": {"int8": q.stat().st_size / 1e6, "fp32": fp32.stat().st_size / 1e6},
            "latency_ms_1thread": latency(q, size, 1, runs=100), **cal}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    (out / "labels.txt").write_text("\n".join(classes))
    print(json.dumps({k: meta[k] for k in ["species", "T", "tau_energy", "tau_conf", "size_mb"]}, ensure_ascii=False))
    print(json.dumps(cal["test"], ensure_ascii=False))


if __name__ == "__main__":
    main()
