"""Huấn luyện theo đề xuất mới: teacher ConvNeXt-Tiny -> student MobileNetV3-Large,
chưng cất tri thức (KD) + loss nhất quán nền, so với MixStyle và ERM. Một loài, nhiều miền.

  loss = (1-α)·CE(y, s(x)) + α·T²·KL(softmax(t(x)/T) ‖ softmax(s(x)/T)) + β(t)·KL(sg[p_s(x)] ‖ p_s(x_bg))

Ví dụ:
  # teacher (tham chiếu)
  python -m plantdx.train --species apple --setting A-all --train-domains pv,pp,pd \
         --model convnext_tiny --config convnext_tiny --seed 0
  # đề xuất
  python -m plantdx.train --species apple --setting A-all --train-domains pv,pp,pd \
         --model mobilenetv3_large_100 --config proposed --seed 0 \
         --teacher runs/apple/A-all/convnext_tiny/seed0/best.pt --alpha 0.5 --beta 1
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from timm.data import resolve_data_config
from torch.utils.data import DataLoader

from .data import (
    CLASSES,
    ROOT,
    PairDataset,
    balanced_sampler,
    build_transforms,
    domains_of,
    load_species,
    select,
)
from .metrics import basic_metrics, bootstrap_ci
from .models import (
    attach_mixstyle,
    autocast,
    batch_to,
    device,
    eval_loader,
    load_ckpt,
    make_model,
    predict_logits,
    to_device,
)

RUNS = ROOT / "runs"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ----------------------------------------------------------------------------- loss
def kd_loss(s_logits, t_logits, T):
    return (
        F.kl_div(
            F.log_softmax(s_logits / T, 1),
            F.softmax(t_logits / T, 1),
            reduction="batchmean",
        )
        * T
        * T
    )


def consistency_loss(s_x, s_bg):
    """KL(sg[p(x)] ‖ p(x_bg)): dự đoán trên ảnh nền thực địa phải giống dự đoán trên ảnh gốc."""
    return F.kl_div(
        F.log_softmax(s_bg, 1), F.softmax(s_x, 1).detach(), reduction="batchmean"
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True, choices=list(CLASSES))
    ap.add_argument(
        "--setting", required=True, help="tên thiết lập, vd A-all, A-LODO-PD, T-all, T0"
    )
    ap.add_argument("--train-domains", required=True, help="miền huấn luyện: pv,pp,pd")
    ap.add_argument("--model", default="mobilenetv3_large_100")
    ap.add_argument(
        "--config", default="", help="tên cấu hình (thư mục kết quả); mặc định = model"
    )
    ap.add_argument("--teacher", default="", help="checkpoint teacher (bật chưng cất)")
    ap.add_argument("--alpha", type=float, default=0.5, help="trọng số KD")
    ap.add_argument("--kd-T", type=float, default=4.0, help="nhiệt độ KD")
    ap.add_argument(
        "--beta", type=float, default=0.0, help="trọng số nhất quán nền (0 = tắt)"
    )
    ap.add_argument(
        "--beta-warm", type=float, default=1.0, help="số epoch tăng dần β từ 0"
    )
    ap.add_argument("--mixstyle", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--samples-per-epoch", type=int, default=7680)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--no-amp", action="store_true", help="tắt bfloat16 tự động (chạy FP32/TF32)"
    )
    ap.add_argument(
        "--limit", type=int, default=0, help="chỉ dùng N ảnh mỗi tập (chạy thử nhanh)"
    )
    ap.add_argument("--out", default=str(RUNS))
    args = ap.parse_args(argv)
    args.config = args.config or args.model

    out = (
        Path(args.out) / args.species / args.setting / args.config / f"seed{args.seed}"
    )
    out.mkdir(parents=True, exist_ok=True)
    if (out / "metrics.json").exists():
        print(f"Bỏ qua {out}: đã có kết quả")
        return
    set_seed(args.seed)
    dev = device()
    classes = CLASSES[args.species]
    m = load_species(args.species)
    train_doms = args.train_domains.split(",")

    def lim(d):
        if not args.limit:
            return d
        return (
            d.sample(frac=1, random_state=0)
            .groupby("dataset")
            .head(args.limit)
            .reset_index(drop=True)
        )

    model = to_device(make_model(args.model, len(classes)), dev)
    amp = not args.no_amp
    if args.mixstyle:
        attach_mixstyle(model, args.model)
    cfg = resolve_data_config({}, model=model)
    mean, std = cfg["mean"], cfg["std"]
    tf_train = build_transforms(args.img_size, mean, std, True)
    tf_eval = build_transforms(args.img_size, mean, std, False)

    teacher = None
    if args.teacher:
        teacher, tck = load_ckpt(args.teacher, dev)
        assert tck["classes"] == classes, "teacher khác danh sách lớp"
        assert tck["mean"] == list(mean) or tck["mean"] == mean, (
            "teacher và student phải cùng chuẩn hoá"
        )
        for p in teacher.parameters():
            p.requires_grad_(False)

    pair = args.beta > 0
    train_df = lim(select(m, train_doms, "train"))
    sampler = balanced_sampler(train_df, args.samples_per_epoch, args.seed)
    dl = DataLoader(
        PairDataset(train_df, tf_train, args.img_size, pair=pair),
        batch_size=args.batch,
        sampler=sampler,
        num_workers=args.workers,
        drop_last=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers else None,
        pin_memory=dev.type == "cuda",
    )
    val_sets = {f"{d}_val": lim(select(m, [d], "val")) for d in train_doms}
    val_loaders = {k: eval_loader(df, tf_eval) for k, df in val_sets.items()}

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * len(dl)
    warm = max(1, len(dl) // 2)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (
            (s + 1) / warm
            if s < warm
            else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm)))
        ),
    )
    beta_steps = max(1, int(args.beta_warm * len(dl)))

    history, best, step, t0 = [], -1.0, 0, time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        agg = {"loss": 0.0, "ce": 0.0, "kd": 0.0, "cons": 0.0, "acc": 0, "n": 0}
        for batch in dl:
            if pair:
                x, x_bg, has, y = batch
            else:
                x, y = batch
            x, y = batch_to(x, dev), y.to(dev, non_blocking=True)
            with autocast(dev, amp):
                logits = model(x)
            logits = logits.float()
            ce = F.cross_entropy(logits, y, label_smoothing=0.1)
            loss, kd, cons = (
                ce,
                torch.zeros((), device=dev),
                torch.zeros((), device=dev),
            )
            if teacher is not None:
                with torch.no_grad(), autocast(dev, amp):
                    t_logits = teacher(x)
                t_logits = t_logits.float()
                kd = kd_loss(logits, t_logits, args.kd_T)
                loss = (1 - args.alpha) * ce + args.alpha * kd
            if pair and has.any():
                idx = has.bool().to(dev)
                with autocast(dev, amp):
                    bg_logits = model(batch_to(x_bg, dev)[idx])
                cons = consistency_loss(logits[idx], bg_logits.float())
                loss = loss + args.beta * min(1.0, step / beta_steps) * cons
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            n = len(y)
            for k, v in [("loss", loss), ("ce", ce), ("kd", kd), ("cons", cons)]:
                agg[k] += v.item() * n
            agg["acc"] += (logits.argmax(1) == y).sum().item()
            agg["n"] += n
        rec = {
            "epoch": ep,
            **{f"train_{k}": agg[k] / agg["n"] for k in ["loss", "ce", "kd", "cons"]},
            "train_acc": agg["acc"] / agg["n"],
            "lr": sched.get_last_lr()[0],
            "time_min": (time.time() - t0) / 60,
        }
        f1s = []
        for k, df in val_sets.items():
            lg = predict_logits(model, val_loaders[k], dev, amp)
            r = basic_metrics(df.label.to_numpy(), lg.argmax(1), len(classes))
            rec[f"{k}_acc"], rec[f"{k}_f1"] = r["accuracy"], r["macro_f1"]
            f1s.append(r["macro_f1"])
        rec["val_f1_mean"] = float(
            np.mean(f1s)
        )  # chọn mô hình: trung bình F1 trên val của các miền huấn luyện
        history.append(rec)
        print(
            json.dumps(
                {k: round(v, 4) if isinstance(v, float) else v for k, v in rec.items()}
            ),
            flush=True,
        )
        if rec["val_f1_mean"] > best:
            best = rec["val_f1_mean"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "classes": classes,
                    "species": args.species,
                    "mean": list(mean),
                    "std": list(std),
                    "epoch": ep,
                },
                out / "best.pt",
            )
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)

    # ----- đánh giá trên test của mọi miền của loài (gồm cả miền không huấn luyện)
    model, ck = load_ckpt(out / "best.pt", dev)
    res = {
        "best_epoch": ck["epoch"],
        "train_minutes": (time.time() - t0) / 60,
        "args": vars(args),
        "train_domains": train_doms,
        "heldout_domains": [d for d in domains_of(args.species) if d not in train_doms],
        "test": {},
    }
    for d in domains_of(args.species) + ["pdfull"]:
        df = lim(select(m, [d], "test"))
        if df.empty:
            continue
        lg = predict_logits(model, eval_loader(df, tf_eval), dev, amp)
        y, yhat = df.label.to_numpy(), lg.argmax(1)
        r = basic_metrics(y, yhat, len(classes))
        r.update(bootstrap_ci(y, yhat))
        res["test"][d] = r
        pd.DataFrame({"path": df.path, "y": y, "pred": yhat}).to_csv(
            out / f"pred_{d}_test.csv", index=False
        )
        np.save(out / f"logits_{d}_test.npy", lg.astype(np.float32))
        np.save(out / f"y_{d}_test.npy", y.astype(np.int64))
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    print(
        "TEST",
        {
            d: (round(r["accuracy"], 4), round(r["macro_f1"], 4))
            for d, r in res["test"].items()
        },
    )


if __name__ == "__main__":
    main()
