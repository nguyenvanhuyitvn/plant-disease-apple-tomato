"""Hiệu chỉnh độ tin cậy và lọc ảnh lạ cho một mô hình theo loài.

  - Temperature scaling: tìm T* tối thiểu NLL trên val (các miền huấn luyện).
  - Ngưỡng năng lượng τ_E (score = logsumexp(logits/T*)): giữ 95% ảnh đúng loài trên val.
    Ảnh có score < τ_E -> "không phải lá táo/cà chua".
  - Ngưỡng tin cậy τ_c: nhỏ nhất sao cho độ chính xác trên phần ảnh được trả lời ≥ target (val).
    Ảnh có xác suất lớn nhất < τ_c -> "chưa chắc, hãy chụp lại".
  - Cà chua: thêm "ood_near_tv" – lá Tomato-Village mắc bệnh ngoài 8 lớp, so với ảnh test TV đúng lớp.

  python -m plantdx.calibrate --run runs/apple/A-all/proposed/seed0
Kết quả: <run>/calib.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp, softmax
from sklearn.metrics import roc_auc_score

from .data import build_transforms, domains_of, load_species, near_ood_sets, ood_sets, select


# ----------------------------------------------------------------------------- hàm trên mảng logits
def nll(logits, y, T):
    lp = logits / T - logsumexp(logits / T, axis=1, keepdims=True)
    return float(-lp[np.arange(len(y)), y].mean())


def fit_temperature(logits, y):
    grid = np.exp(np.linspace(np.log(0.3), np.log(10), 400))
    return float(grid[int(np.argmin([nll(logits, y, t) for t in grid]))])


def ece(probs, y, bins=15):
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def aurc(probs, y):
    """Diện tích dưới đường coverage–risk (càng nhỏ càng tốt)."""
    order = np.argsort(-probs.max(1))
    err = (probs.argmax(1)[order] != y[order]).astype(float)
    risk = np.cumsum(err) / np.arange(1, len(err) + 1)
    return float(risk.mean())


def energy_score(logits, T):
    return T * logsumexp(logits / T, axis=1)


def conf_threshold(probs, y, target):
    """τ_c nhỏ nhất để độ chính xác trên phần được trả lời ≥ target; kèm độ phủ."""
    conf, ok = probs.max(1), probs.argmax(1) == y
    for t in np.linspace(0, 0.99, 100):
        keep = conf >= t
        if keep.sum() >= 10 and ok[keep].mean() >= target:
            return float(t), float(keep.mean())
    return 0.99, float((conf >= 0.99).mean())


def ood_metrics(s_id, s_ood, tau):
    lab = np.r_[np.ones(len(s_id)), np.zeros(len(s_ood))]
    sc = np.r_[s_id, s_ood]
    t95 = np.percentile(s_id, 5)
    return {
        "auroc": float(roc_auc_score(lab, sc)),
        "fpr_at_95tpr": float((s_ood >= t95).mean()),
        "id_kept_at_tau": float((s_id >= tau).mean()),
        "ood_rejected_at_tau": float((s_ood < tau).mean()),
    }


def fit_calibration(val_logits, val_y, ood_val_logits, target_acc=0.9, tpr=0.95):
    T = fit_temperature(val_logits, val_y)
    s_val = energy_score(val_logits, T)
    tau_e = float(np.percentile(s_val, 100 * (1 - tpr)))
    tau_c, cov = conf_threshold(softmax(val_logits / T, 1), val_y, target_acc)
    rep = {
        "T": T,
        "tau_energy": tau_e,
        "tau_conf": tau_c,
        "target_acc": target_acc,
        "val_coverage": cov,
        "val_nll_before": nll(val_logits, val_y, 1.0),
        "val_nll_after": nll(val_logits, val_y, T),
        "val_ece_before": ece(softmax(val_logits, 1), val_y),
        "val_ece_after": ece(softmax(val_logits / T, 1), val_y),
    }
    if ood_val_logits is not None and len(ood_val_logits):
        rep["ood_val"] = ood_metrics(s_val, energy_score(ood_val_logits, T), tau_e)
    return rep


def evaluate_calibration(cal, test_logits: dict, test_y: dict, ood_test_logits, near_logits=None):
    """Báo cáo trên test: ECE trước/sau, AURC, độ phủ & độ chính xác khi áp τ_E và τ_c, OOD.
    near_logits: ảnh lạ khó của Tomato-Village, so với ảnh test miền "tv" (cùng miền chụp)."""
    T, rep, s_id_all = cal["T"], {}, []
    for d, lg in test_logits.items():
        y = test_y[d]
        p0, p1 = softmax(lg, 1), softmax(lg / T, 1)
        s = energy_score(lg, T)
        s_id_all.append(s)
        keep = (s >= cal["tau_energy"]) & (p1.max(1) >= cal["tau_conf"])
        rep[d] = {
            "ece_before": ece(p0, y),
            "ece_after": ece(p1, y),
            "aurc": aurc(p1, y),
            "acc_all": float((p1.argmax(1) == y).mean()),
            "coverage": float(keep.mean()),
            "acc_answered": float((p1.argmax(1)[keep] == y[keep]).mean())
            if keep.any()
            else None,
        }
    if ood_test_logits is not None and len(ood_test_logits):
        rep["ood_test"] = ood_metrics(
            np.concatenate(s_id_all),
            energy_score(ood_test_logits, T),
            cal["tau_energy"],
        )
    if near_logits is not None and len(near_logits) and "tv" in test_logits:
        T = cal["T"]
        r = ood_metrics(energy_score(test_logits["tv"], T), energy_score(near_logits, T), cal["tau_energy"])
        # bị chặn bởi một trong hai ngưỡng: "không phải lá cà chua" hoặc "chưa chắc"
        held = (energy_score(near_logits, T) < cal["tau_energy"]) | (
            softmax(near_logits / T, 1).max(1) < cal["tau_conf"]
        )
        r["rejected_any"] = float(held.mean())
        r["n"] = int(len(near_logits))
        rep["ood_near_tv"] = r
    return rep


# ----------------------------------------------------------------------------- chạy cho một run PyTorch
def main(argv=None):
    from .models import device, eval_loader, load_ckpt, predict_logits  # cần PyTorch

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--target-acc", type=float, default=0.9)
    ap.add_argument("--ood-max", type=int, default=2000)
    args = ap.parse_args(argv)
    run = Path(args.run)
    dev = device()
    model, ck = load_ckpt(run / "best.pt", dev)
    sp, train_doms = ck["species"], ck["args"]["train_domains"].split(",")
    amp = not ck["args"].get(
        "no_amp", False
    )  # suy luận cùng độ chính xác số như lúc huấn luyện
    tf = build_transforms(ck["args"]["img_size"], ck["mean"], ck["std"], False)
    m = load_species(sp)
    val = select(m, train_doms, "val")
    val_lg = predict_logits(model, eval_loader(val, tf), dev, amp)
    ood_v = ood_sets(sp, "val").sample(frac=1, random_state=0).head(args.ood_max)
    ood_t = ood_sets(sp, "test").sample(frac=1, random_state=0).head(args.ood_max)
    ood_v_lg = predict_logits(model, eval_loader(ood_v, tf), dev, amp)
    ood_t_lg = predict_logits(model, eval_loader(ood_t, tf), dev, amp)
    cal = fit_calibration(val_lg, val.label.to_numpy(), ood_v_lg, args.target_acc)
    t_lg, t_y = {}, {}
    for d in domains_of(sp) + ["pdfull"]:
        f = run / f"logits_{d}_test.npy"
        if f.exists():
            t_lg[d] = np.load(f)
            t_y[d] = np.load(run / f"y_{d}_test.npy")
    near = near_ood_sets(sp)
    near_lg = predict_logits(model, eval_loader(near, tf), dev, amp) if len(near) else None
    cal["test"] = evaluate_calibration(cal, t_lg, t_y, ood_t_lg, near_lg)
    (run / "calib.json").write_text(json.dumps(cal, indent=2))
    print(
        json.dumps(
            {
                k: cal[k]
                for k in [
                    "T",
                    "tau_energy",
                    "tau_conf",
                    "val_ece_before",
                    "val_ece_after",
                ]
            }
        )
    )
    print(json.dumps(cal["test"].get("ood_test", {})))
    if "ood_near_tv" in cal["test"]:
        print("near-OOD TV", json.dumps(cal["test"]["ood_near_tv"]))


if __name__ == "__main__":
    main()
