"""Chấm thêm các lần chạy đã có trên miền mới mà không huấn luyện lại (vd. Tomato-Village "tv" cho cà chua).

Với mỗi lần chạy có metrics.json nhưng chưa có kết quả trên miền mới:
  - suy luận trên test của miền đó, ghi pred_<d>_test.csv, logits_<d>_test.npy, y_<d>_test.npy
  - thêm vào metrics.json: test[<d>], cập nhật heldout_domains theo domains_of() hiện tại
  - xoá calib.json để bước hiệu chỉnh (plantdx.calibrate) chạy lại với miền mới và ảnh lạ khó

  python -m plantdx.eval_extra --runs runs/tomato              # mọi thiết lập cà chua
  python -m plantdx.eval_extra --runs runs/tomato --dry-run    # chỉ liệt kê
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import build_transforms, domains_of, load_species, select
from .metrics import basic_metrics, bootstrap_ci


def eval_run(run: Path, domains: list[str] | None, dry: bool, dev=None) -> list[str]:
    res = json.loads((run / "metrics.json").read_text())
    args = res["args"]
    sp = args["species"]
    todo = [d for d in (domains or domains_of(sp)) if d not in res["test"]]
    if not todo:
        return []
    m = load_species(sp)
    sets = {}
    for d in todo:
        df = select(m, [d], "test")
        if args.get("limit"):  # lần chạy thử (QUICK=1): cùng cách lấy mẫu như train.py
            df = df.sample(frac=1, random_state=0).groupby("dataset").head(args["limit"]).reset_index(drop=True)
        if len(df):
            sets[d] = df
    if dry or not sets:
        return list(sets)

    from .models import eval_loader, load_ckpt, predict_logits  # cần PyTorch

    model, ck = load_ckpt(run / "best.pt", dev)
    tf = build_transforms(args["img_size"], ck["mean"], ck["std"], False)
    amp = not args.get("no_amp", False)
    n_cls = len(ck["classes"])
    for d, df in sets.items():
        lg = predict_logits(model, eval_loader(df, tf), dev, amp)
        y, yhat = df.label.to_numpy(), lg.argmax(1)
        r = basic_metrics(y, yhat, n_cls)
        r.update(bootstrap_ci(y, yhat))
        res["test"][d] = r
        pd.DataFrame({"path": df.path, "y": y, "pred": yhat}).to_csv(run / f"pred_{d}_test.csv", index=False)
        np.save(run / f"logits_{d}_test.npy", lg.astype(np.float32))
        np.save(run / f"y_{d}_test.npy", y.astype(np.int64))
    res["heldout_domains"] = [d for d in domains_of(sp) if d not in res["train_domains"]]
    res.setdefault("added_later", []).extend(sets)
    (run / "metrics.json").write_text(json.dumps(res, indent=2))
    (run / "calib.json").unlink(missing_ok=True)
    return list(sets)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="thư mục, vd runs/tomato hoặc runs")
    ap.add_argument("--domains", default="", help="mặc định: mọi miền của loài còn thiếu, vd tv")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    doms = [d for d in args.domains.split(",") if d] or None
    files = sorted(Path(args.runs).glob("**/seed*/metrics.json"))
    dev = None
    if not args.dry_run:
        from .models import device

        dev = device()
    n = 0
    for f in files:
        added = eval_run(f.parent, doms, args.dry_run, dev)
        if added:
            n += 1
            r = json.loads(f.read_text())["test"]
            info = {d: round(r[d]["macro_f1"], 4) for d in added if d in r}
            print(("[dry] " if args.dry_run else "") + f"{f.parent}: {added} {info}")
    print(f"{n}/{len(files)} lần chạy được chấm thêm")


if __name__ == "__main__":
    main()
