"""Suy luận bằng ONNX Runtime (không cần PyTorch) – dùng chung cho Streamlit và kiểm thử.

Tiền xử lý giống lúc đánh giá: cạnh ngắn -> meta["resize"], cắt giữa meta["img_size"], chuẩn hoá ImageNet.
Quyết định 3 mức:
  not_leaf  : năng lượng < τ_E  -> "không phải lá của loài này"
  uncertain : xác suất lớn nhất (đã hiệu chỉnh) < τ_c -> "chưa chắc, hãy chụp lại"
  ok        : trả về tên bệnh
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
MODELS = Path(os.environ.get("PLANTDX_MODELS", ROOT / "models"))   # đổi thư mục mô hình khi chạy thử


def _logsumexp(z):
    m = z.max()
    return float(m + np.log(np.exp(z - m).sum()))


@dataclass
class Prediction:
    species: str
    status: str            # ok | uncertain | not_leaf
    label: str
    name_vi: str
    prob: float
    energy: float
    probs: np.ndarray
    ms: float


class SpeciesModel:
    def __init__(self, folder: Path):
        self.meta = json.loads((folder / "meta.json").read_text())
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        self.sess = ort.InferenceSession(str(folder / "model.onnx"), so, providers=["CPUExecutionProvider"])
        self.mean = np.array(self.meta["mean"], np.float32)[:, None, None]
        self.std = np.array(self.meta["std"], np.float32)[:, None, None]

    @property
    def species(self):
        return self.meta["species"]

    def preprocess(self, img: Image.Image) -> np.ndarray:
        img = ImageOps.exif_transpose(img).convert("RGB")
        w, h = img.size
        r, s = self.meta["resize"], self.meta["img_size"]
        if w <= h:
            nw, nh = r, int(r * h / w)
        else:
            nw, nh = int(r * w / h), r
        img = img.resize((nw, nh), Image.BILINEAR)
        left, top = int(round((nw - s) / 2)), int(round((nh - s) / 2))
        img = img.crop((left, top, left + s, top + s))
        x = np.asarray(img, np.float32).transpose(2, 0, 1) / 255.0
        return ((x - self.mean) / self.std)[None].astype(np.float32)

    def predict(self, img: Image.Image) -> Prediction:
        x = self.preprocess(img)
        t0 = time.perf_counter()
        logits = self.sess.run(None, {"input": x})[0][0].astype(np.float64)
        ms = (time.perf_counter() - t0) * 1000
        T = self.meta["T"]
        z = logits / T
        probs = np.exp(z - _logsumexp(z))
        energy = T * _logsumexp(z)
        k = int(probs.argmax())
        if energy < self.meta["tau_energy"]:
            status = "not_leaf"
        elif probs[k] < self.meta["tau_conf"]:
            status = "uncertain"
        else:
            status = "ok"
        return Prediction(self.species, status, self.meta["classes"][k], self.meta["names_vi"][k], float(probs[k]),
                          float(energy), probs, ms)


def available_models() -> dict[str, SpeciesModel]:
    out = {}
    for sp in ["apple", "tomato"]:
        f = MODELS / sp
        if (f / "meta.json").exists() and (f / "model.onnx").exists():
            out[sp] = SpeciesModel(f)
    return out


def auto_predict(models: dict[str, SpeciesModel], img: Image.Image) -> tuple[Prediction, dict]:
    """Chạy mọi mô hình; chọn loài có năng lượng vượt ngưỡng τ_E nhiều nhất.
    Nếu không mô hình nào vượt ngưỡng -> ảnh không phải lá táo/cà chua."""
    preds = {sp: m.predict(img) for sp, m in models.items()}
    margin = {sp: p.energy - models[sp].meta["tau_energy"] for sp, p in preds.items()}
    best = max(margin, key=margin.get)
    return preds[best], preds
