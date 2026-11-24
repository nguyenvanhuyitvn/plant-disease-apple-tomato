"""Tổng hợp kết quả runs/ -> results/{runs.csv, summary.csv, summary.md}.

  python -m plantdx.summarize
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .data import ROOT

CONFIG_ORDER = ["convnext_tiny", "resnet50", "mnv3_erm", "mnv3_mixstyle", "mnv3_kd", "mnv3_cons", "proposed"]
CONFIG_LABEL = {"convnext_tiny": "ConvNeXt-T (teacher)", "resnet50": "ResNet-50", "mnv3_erm": "MNv3-L ERM",
                "mnv3_mixstyle": "MNv3-L + MixStyle", "mnv3_kd": "MNv3-L + KD", "mnv3_cons": "MNv3-L + nhất quán nền",
                "proposed": "MNv3-L + KD + nhất quán nền (đề xuất)"}


def collect(runs: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(runs.glob("*/*/*/seed*/metrics.json")):
        sp, setting, config, seed = f.parts[-5:-1]
        r = json.loads(f.read_text())
        cal = json.loads((f.parent / "calib.json").read_text()) if (f.parent / "calib.json").exists() else {}
        held = set(r.get("heldout_domains", []))
        if "pd" in held:
            held.add("pdfull")      # ảnh nguyên PlantDoc cũng chưa thấy khi bỏ miền PlantDoc
        for d, v in r["test"].items():
            row = {"species": sp, "setting": setting, "config": config, "seed": int(seed[4:]), "domain": d,
                   "heldout": d in held,"acc": v["accuracy"], "f1": v["macro_f1"],
                   "n": v["n"], "minutes": r["train_minutes"]}
            c = cal.get("test", {}).get(d)
            if c:
                row.update({"ece_before": c["ece_before"], "ece_after": c["ece_after"], "aurc": c["aurc"],
                            "coverage": c["coverage"], "acc_answered": c["acc_answered"]})
            if "ood_test" in cal.get("test", {}):
                row["ood_auroc"] = cal["test"]["ood_test"]["auroc"]
                row["ood_fpr95"] = cal["test"]["ood_test"]["fpr_at_95tpr"]
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    num = [c for c in ["acc", "f1", "ece_after", "aurc", "coverage", "acc_answered", "ood_auroc", "ood_fpr95", "minutes"]
           if c in df]
    g = df.groupby(["species", "setting", "config", "domain", "heldout"])
    s = g[num].agg(["mean", "std"])
    s.columns = [f"{a}_{b}" for a, b in s.columns]
    s["seeds"] = g.size()
    s = s.reset_index()
    s["config_order"] = s.config.map({c: i for i, c in enumerate(CONFIG_ORDER)}).fillna(99)
    return s.sort_values(["species", "setting", "domain", "config_order"]).drop(columns="config_order")


def to_markdown(s: pd.DataFrame) -> str:
    lines = ["# Kết quả theo loài (F1 macro %, trung bình ± độ lệch chuẩn qua các seed)", ""]
    for (sp, setting), sub in s.groupby(["species", "setting"], sort=False):
        piv = sub.assign(cell=[f"{m * 100:.1f} ± {0 if pd.isna(sd) else sd * 100:.1f}" + (" *" if h else "")
                               for m, sd, h in zip(sub.f1_mean, sub.f1_std, sub.heldout, strict=True)]) \
                 .pivot_table(index="config", columns="domain", values="cell", aggfunc="first")
        piv = piv.reindex([c for c in CONFIG_ORDER if c in piv.index] + [c for c in piv.index if c not in CONFIG_ORDER])
        piv.index = [CONFIG_LABEL.get(c, c) for c in piv.index]
        lines += [f"## {sp} – {setting}", "", piv.to_markdown(), ""]
    lines += ["`*` = miền không dùng khi huấn luyện (kiểm tra tổng quát hoá)."]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args(argv)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df = collect(Path(args.runs))
    if df.empty:
        print("Chưa có kết quả nào trong", args.runs); return
    df.to_csv(out / "runs.csv", index=False)
    s = summarize(df)
    s.to_csv(out / "summary.csv", index=False)
    (out / "summary.md").write_text(to_markdown(s))
    print(f"{len(df)} dòng, {df.groupby(['species', 'setting', 'config', 'seed']).ngroups} lần chạy -> {out}")


if __name__ == "__main__":
    main()
