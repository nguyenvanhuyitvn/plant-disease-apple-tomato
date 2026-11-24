"""Bước 2 – dữ liệu theo loài (táo, cà chua), thêm Plant Pathology 2020 + 2021.

Chạy (sau bước 1 scripts/prepare_data.py):
    .venv/bin/python scripts/prepare_species.py --data-dir data --workers 8

Đầu vào:
  data/processed/manifest.csv                 PV + PD đã làm sạch (từ prepare_data.py)
  data/raw/plantvillage/...                    để lấy thêm lớp Apple___Black_rot
  data/raw/plantpathology/2020/train.csv + images/          (Kaggle plant-pathology-2020-fgvc7)
  data/raw/plantpathology/2021/train.csv + train_images/    (Kaggle plant-pathology-2021-fgvc8)

Đầu ra (data/processed/species/):
  manifest_apple.csv, manifest_tomato.csv     cột: path,dataset,split,unified_class,group_id,mask_path,source_file
  stats_apple.csv, stats_tomato.csv           số ảnh theo lớp x miền x tập
  removed_log.csv                             ảnh bị loại và lý do
  qa/                                         ảnh kiểm tra trực quan, các cặp trùng chéo
Ảnh Plant Pathology được thu nhỏ (cạnh ngắn PP_SHORT_SIDE) vào data/processed/plantpathology/.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_data import (PHASH_MAX_DIST, SEED, UnionFind, contact_sheet,  # noqa: E402
                          hash_many, leaf_key, load_leaf_map, near_pairs, phash_int,
                          prepare_plantvillage, val_split)

PP_SHORT_SIDE = 448        # đủ cho RandomResizedCrop 224, nhỏ hơn ảnh gốc ~30 lần
PP_GROUP_DIST = 6          # Hamming <= 6 coi là cùng lá/cùng cảnh -> cùng nhóm
PP_TEST_FRAC = 0.15
PP_VAL_FRAC = 0.15

APPLE = ["apple_healthy", "apple_scab", "apple_rust", "apple_black_rot"]
TOMATO = ["tomato_healthy", "tomato_bacterial_spot", "tomato_early_blight", "tomato_late_blight",
          "tomato_leaf_mold", "tomato_septoria_leaf_spot", "tomato_yellow_leaf_curl_virus",
          "tomato_mosaic_virus"]

# nhãn đơn của Plant Pathology -> lớp chung; nhãn khác (complex, powdery_mildew, multiple_diseases) bị loại
PP_MAP = {"healthy": "apple_healthy", "scab": "apple_scab", "rust": "apple_rust",
          "frog_eye_leaf_spot": "apple_black_rot"}   # mắt ếch do nấm Botryosphaeria obtusa, cùng tác nhân thối đen


# ----------------------------------------------------------------------------- Plant Pathology
def read_pp_labels(raw: Path, removed: list) -> pd.DataFrame:
    rows = []
    t20 = pd.read_csv(raw / "2020" / "train.csv")
    for r in t20.itertuples():
        labs = [c for c in ["healthy", "multiple_diseases", "rust", "scab"] if getattr(r, c) == 1]
        rows.append(("pp2020", raw / "2020" / "images" / f"{r.image_id}.jpg", " ".join(labs)))
    t21 = pd.read_csv(raw / "2021" / "train.csv")
    for r in t21.itertuples():
        rows.append(("pp2021", raw / "2021" / "train_images" / r.image, r.labels))
    df = pd.DataFrame(rows, columns=["release", "raw_path", "labels"])
    single = df.labels.str.split().str.len() == 1
    df["unified_class"] = df.labels.map(PP_MAP).where(single)
    for (rel, lab), n in df[df.unified_class.isna()].groupby(["release", "labels"]).size().items():
        removed.append(("plantpathology", "", f"{n} anh {rel} nhan '{lab}'", "nhan khong dung (nhieu benh/khong anh xa)"))
    df = df[df.unified_class.notna()].reset_index(drop=True)
    missing = ~df.raw_path.map(Path.exists)
    for r in df[missing].itertuples():
        removed.append(("plantpathology", "", str(r.raw_path), "khong tim thay file"))
    return df[~missing].reset_index(drop=True)


def _resize_one(args):
    src, dst = args
    dst = Path(dst)
    try:
        if not dst.exists():
            with Image.open(src) as im:
                im.draft("RGB", (PP_SHORT_SIDE * 2, PP_SHORT_SIDE * 2))   # giải mã JPEG ở độ phân giải thấp
                im = ImageOps.exif_transpose(im.convert("RGB"))
                w, h = im.size
                s = PP_SHORT_SIDE / min(w, h)
                if s < 1:
                    im = im.resize((round(w * s), round(h * s)), Image.BICUBIC)
                dst.parent.mkdir(parents=True, exist_ok=True)
                im.save(dst, quality=92)
        with Image.open(dst) as im:
            return str(dst), str(imagehash.phash(im)), None
    except Exception as e:
        return str(dst), None, f"{type(e).__name__}: {e}"


def resize_pp(df: pd.DataFrame, data_dir: Path, workers: int, removed: list) -> pd.DataFrame:
    rel = [Path("processed/plantpathology") / c / f"{r}_{Path(p).stem}.jpg"
           for c, r, p in zip(df.unified_class, df.release, df.raw_path, strict=True)]
    jobs = [(str(p), str(data_dir / q)) for p, q in zip(df.raw_path, rel, strict=True)]
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_resize_one, jobs, chunksize=16))
    df = df.copy()
    df["path"] = [str(q) for q in rel]
    df["phash"] = [r[1] for r in res]
    bad = df.phash.isna()
    for r, (_, _, err) in zip(df[bad].itertuples(), [x for x in res if x[1] is None], strict=True):
        removed.append(("plantpathology", "", str(r.raw_path), f"anh hong: {err}"))
    print(f"[PP] thu nho {len(df)} anh -> {PP_SHORT_SIDE}px, {int(bad.sum())} anh loi")
    return df[~bad].reset_index(drop=True)


def dedup_and_group_pp(df: pd.DataFrame, removed: list, qa: Path) -> pd.DataFrame:
    """Bản 2021 chứa lại phần lớn ảnh 2020 -> bỏ bản 2020 trùng; rồi gom nhóm ảnh gần giống."""
    ph = [phash_int(h) for h in df.phash]
    pairs = near_pairs(ph, None, PP_GROUP_DIST)
    pd.DataFrame([(df.path[i], df.path[j], d) for i, j, d in pairs],
                 columns=["a", "b", "hamming"]).to_csv(qa / "pp_near_pairs.csv", index=False)
    rel = df.release.to_numpy()
    drop = set()
    for i, j, d in pairs:
        if d <= PHASH_MAX_DIST and rel[i] != rel[j]:
            k = i if rel[i] == "pp2020" else j       # giữ bản 2021 (nhãn được rà soát lại)
            drop.add(k)
    for k in sorted(drop):
        removed.append(("plantpathology", "", df.path[k], "trung voi anh ban 2021"))
    keep = np.array([k not in drop for k in range(len(df))])
    idx = np.nonzero(keep)[0]
    pos = {k: n for n, k in enumerate(idx)}
    uf = UnionFind(len(idx))
    for i, j, _d in pairs:
        if i in pos and j in pos:
            uf.union(pos[i], pos[j])
    out = df.iloc[idx].reset_index(drop=True)
    out["group_id"] = ["pp_" + str(uf.find(n)) for n in range(len(out))]
    # nhóm có nhãn mâu thuẫn -> loại cả nhóm
    mixed = out.groupby("group_id").unified_class.nunique()
    mixed = set(mixed[mixed > 1].index)
    for r in out[out.group_id.isin(mixed)].itertuples():
        removed.append(("plantpathology", "", r.path, "anh gan trung nhung nhan khac nhau"))
    out = out[~out.group_id.isin(mixed)].reset_index(drop=True)
    print(f"[PP] bo {len(drop)} anh 2020 trung 2021, {len(mixed)} nhom nhan mau thuan; "
          f"con {len(out)} anh / {out.group_id.nunique()} nhom")
    return out


def split_pp(df: pd.DataFrame) -> pd.DataFrame:
    """70/15/15 theo nhóm, phân tầng theo lớp."""
    df = df.copy()
    sg = StratifiedGroupKFold(n_splits=round(1 / PP_TEST_FRAC), shuffle=True, random_state=SEED)
    _, te = next(sg.split(df, df.unified_class, df.group_id))
    df["split"] = "train"
    df.loc[df.index[te], "split"] = "test"
    rest = df[df.split == "train"]
    sg2 = StratifiedGroupKFold(n_splits=round((1 - PP_TEST_FRAC) / PP_VAL_FRAC), shuffle=True, random_state=SEED)
    _, va = next(sg2.split(rest, rest.unified_class, rest.group_id))
    df.loc[rest.index[va], "split"] = "val"
    return df


# ----------------------------------------------------------------------------- PV: lớp thối đen
def pv_black_rot(raw: Path, out: Path) -> pd.DataFrame:
    mapping = pd.DataFrame({"unified_class": ["apple_black_rot"], "plantvillage_class": ["Apple___Black_rot"]})
    df = prepare_plantvillage(raw, out, mapping, [])
    df = df.reset_index(drop=True)
    # leaf-map.json ghi lớp này là "Apple_Frogeye Spot" -> prepare_plantvillage không tìm thấy leaf_id
    leaf_map = load_leaf_map(raw / "plantvillage" / "leaf_grouping" / "leaf-map.json")
    leaf = [leaf_map.get(("Apple_Frogeye Spot", leaf_key(f))) for f in df.source_file]
    print(f"[PV] Apple___Black_rot: tim thay leaf_id cho {sum(x is not None for x in leaf)}/{len(df)} anh")
    df["group_id"] = ["pv_" + x if x else g for x, g in zip(leaf, df.group_id, strict=True)]
    # chia gốc của PV có thể đặt cùng lá ở train và test -> bỏ bản train
    cross = df.groupby("group_id").split.nunique()
    df = df[~(df.group_id.isin(cross[cross > 1].index) & (df.split == "train"))].reset_index(drop=True)
    m = df.split == "train"
    is_val = val_split(df[m], "unified_class", "group_id")
    df.loc[is_val[is_val].index, "split"] = "val"
    print(f"[PV] Apple___Black_rot: {df.split.value_counts().to_dict()}")
    return df


# ----------------------------------------------------------------------------- rò rỉ chéo
def cross_leak(apple: pd.DataFrame, data_dir: Path, workers: int, removed: list, qa: Path) -> pd.DataFrame:
    """So PP với PlantDoc (ảnh nguyên) và PlantVillage. PlantDoc thu thập từ Internet nên có thể
    chứa ảnh PP; khi trùng thì bỏ bản PlantDoc (cả ảnh nguyên và lá cắt cùng group_id).
    Trùng PP-PV (hiếm) thì bỏ bản PP."""
    pp = apple[apple.dataset == "plantpathology"].reset_index(drop=True)
    pp_ph = [phash_int(h) for h in pp.phash]
    rows, bad_pd, bad_pp = [], set(), set()
    for ds in ["plantdoc_full", "plantvillage"]:
        o = apple[apple.dataset == ds].reset_index(drop=True)
        h = hash_many([str(data_dir / p) for p in o.path], workers)
        o_ph = [phash_int(h[str(data_dir / p)][2]) for p in o.path]
        for i, j, d in near_pairs(o_ph, pp_ph, PHASH_MAX_DIST):
            rows.append((ds, o.path[i], o.split[i], pp.path[j], pp.split[j], d))
            if ds == "plantdoc_full":
                bad_pd.add(o.group_id[i])
            else:
                bad_pp.add(pp.path[j])
    pd.DataFrame(rows, columns=["other_dataset", "other_path", "other_split", "pp_path", "pp_split", "hamming"]) \
        .to_csv(qa / "cross_pp_matches.csv", index=False)
    for g in sorted(bad_pd):
        removed.append(("plantdoc", "", g, "trung voi anh Plant Pathology (ro ri cheo)"))
    for p in sorted(bad_pp):
        removed.append(("plantpathology", "", p, "trung voi anh PlantVillage (ro ri cheo)"))
    print(f"[Cheo] {len(rows)} cap trung: bo {len(bad_pd)} anh PlantDoc, {len(bad_pp)} anh PP")
    m_pd = apple.dataset.str.startswith("plantdoc") & apple.group_id.isin(bad_pd)
    m_pp = (apple.dataset == "plantpathology") & apple.path.isin(bad_pp)
    return apple[~(m_pd | m_pp)]


# ----------------------------------------------------------------------------- main
def check(man: pd.DataFrame, data_dir: Path, name: str):
    for ds in man.dataset.unique():
        sub = man[man.dataset == ds]
        leak = sub.groupby("group_id").split.nunique()
        assert (leak == 1).all(), f"{name}/{ds}: {int((leak > 1).sum())} nhom nam o nhieu tap"
    assert man.path.map(lambda p: (data_dir / p).exists()).all(), f"{name}: co duong dan khong ton tai"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    data_dir = Path(args.data_dir).resolve()
    raw, out = data_dir / "raw", data_dir / "processed"
    sp = out / "species"
    qa = sp / "qa"
    qa.mkdir(parents=True, exist_ok=True)
    removed: list = []
    cols = ["path", "dataset", "split", "unified_class", "group_id", "mask_path", "source_file"]

    base = pd.read_csv(out / "manifest.csv").fillna({"mask_path": ""})

    # ---- cà chua: PV + PD, 8 lớp chung (Target_Spot, Spider_mites đã bị loại ở prepare_data)
    tomato = base[base.unified_class.isin(TOMATO)].copy()

    # ---- táo: PV + PD + PP, 4 lớp
    pp = read_pp_labels(raw / "plantpathology", removed)
    pp = resize_pp(pp, data_dir, args.workers, removed)
    pp = dedup_and_group_pp(pp, removed, qa)
    pp = split_pp(pp)
    pp["dataset"] = "plantpathology"
    pp["mask_path"] = ""
    pp["source_file"] = pp.release + "/" + pp.raw_path.map(lambda p: Path(p).name)
    pv_br = pv_black_rot(raw, out)
    apple = pd.concat([base[base.unified_class.isin(APPLE)], pv_br, pp], ignore_index=True)
    apple = cross_leak(apple, data_dir, args.workers, removed, qa)

    for name, man, classes in [("apple", apple, APPLE), ("tomato", tomato, TOMATO)]:
        man = man[cols].sort_values(["dataset", "split", "unified_class", "path"]).reset_index(drop=True)
        check(man, data_dir, name)
        man.to_csv(sp / f"manifest_{name}.csv", index=False)
        stats = man.pivot_table(index="unified_class", columns=["dataset", "split"], values="path",
                                aggfunc="count", fill_value=0).reindex(classes)
        stats.to_csv(sp / f"stats_{name}.csv")
        print(f"\n=== {name.upper()} ===")
        print(stats.to_string())
        # ảnh kiểm tra: 8 ảnh/lớp mỗi miền
        sheet = man[man.dataset != "plantdoc_full"].groupby(["dataset", "unified_class"]).head(8)
        contact_sheet([data_dir / p for p in sheet.path], qa / f"samples_{name}.jpg")

    pd.DataFrame(removed, columns=["dataset", "split", "item", "reason"]).to_csv(sp / "removed_log.csv", index=False)
    print("\nLy do loai:")
    print(pd.DataFrame(removed, columns=["dataset", "split", "item", "reason"])
          .groupby(["dataset", "reason"]).size().to_string())


if __name__ == "__main__":
    main()
