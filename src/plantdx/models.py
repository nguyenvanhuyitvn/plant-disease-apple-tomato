"""Mô hình: tạo/nạp checkpoint, MixStyle, suy luận theo batch. Dùng chung cho train, calibrate, deploy.

Bản Linux: ưu tiên GPU NVIDIA (CUDA), bật TF32 + cuDNN benchmark, suy luận/huấn luyện với bfloat16 (AMP).
"""

from __future__ import annotations

import contextlib
import random

import numpy as np
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import PairDataset


def device():
    """CUDA nếu có (server Linux); MPS/CPU chỉ để chạy thử trên máy khác."""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    print("CẢNH BÁO: không thấy GPU, chạy bằng CPU (rất chậm)", flush=True)
    return torch.device("cpu")


def autocast(dev, enabled=True):
    """bfloat16 tự động trên GPU (Blackwell/Ampere trở lên); tắt trên CPU."""
    if not enabled or dev.type not in ("cuda", "mps"):
        return contextlib.nullcontext()
    return torch.autocast(device_type=dev.type, dtype=torch.bfloat16)


def to_device(model, dev):
    """channels_last giúp conv nhanh hơn trên GPU NVIDIA khi dùng AMP."""
    model = model.to(dev)
    return model.to(memory_format=torch.channels_last) if dev.type == "cuda" else model


def batch_to(x, dev):
    x = x.to(dev, non_blocking=True)
    return (
        x.contiguous(memory_format=torch.channels_last)
        if dev.type == "cuda" and x.dim() == 4
        else x
    )


def make_model(name: str, n_classes: int, pretrained=True):
    return timm.create_model(name, pretrained=pretrained, num_classes=n_classes)


# ----------------------------------------------------------------------------- MixStyle
class MixStyle(nn.Module):
    """Zhou et al., ICLR 2021: trộn thống kê (mean, std) theo kênh giữa các ảnh trong batch,
    mô phỏng ảnh của "miền mới". Chỉ hoạt động khi huấn luyện."""

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
        super().__init__()
        self.p, self.alpha, self.eps = p, alpha, eps
        self.beta = torch.distributions.Beta(alpha, alpha)

    def forward(self, x):
        if random.random() > self.p:
            return x
        b = x.size(0)
        mu = x.mean(dim=[2, 3], keepdim=True).detach()
        sig = (x.var(dim=[2, 3], keepdim=True) + self.eps).sqrt().detach()
        lam = self.beta.sample((b, 1, 1, 1)).to(x.device, x.dtype)
        perm = torch.randperm(b, device=x.device)
        mu2 = lam * mu + (1 - lam) * mu[perm]
        sig2 = lam * sig + (1 - lam) * sig[perm]
        return (x - mu) / sig * sig2 + mu2


def attach_mixstyle(model: nn.Module, name: str):
    """Chèn MixStyle sau các khối nông (nơi lưu "phong cách" ảnh) bằng forward hook."""
    if name.startswith("mobilenetv3"):
        targets = [model.blocks[1], model.blocks[2]]
    elif name.startswith("convnext"):
        targets = [model.stages[0], model.stages[1]]
    elif name.startswith("resnet"):
        targets = [model.layer1, model.layer2]
    else:
        raise ValueError(f"chưa hỗ trợ MixStyle cho {name}")
    ms = MixStyle()
    for t in targets:
        t.register_forward_hook(lambda mod, _inp, out: ms(out) if mod.training else out)
    return ms


# ----------------------------------------------------------------------------- suy luận
def eval_loader(df, tf, workers=4, batch=128):
    return DataLoader(
        PairDataset(df, tf),
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        persistent_workers=workers > 0,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def predict_logits(model, dl, dev, amp=True):
    model.eval()
    out = []
    for x, _ in dl:
        with autocast(dev, amp):
            out.append(model(batch_to(x, dev)).float().cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, 1), np.float32)


def load_ckpt(path, dev=None):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = make_model(ck["args"]["model"], len(ck["classes"]), pretrained=False)
    m.load_state_dict(ck["model"])
    if dev is not None:
        m = to_device(m, dev)
    return m.eval(), ck
