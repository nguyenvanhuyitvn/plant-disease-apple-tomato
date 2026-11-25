"""Bước 3 (tuỳ chọn) – thêm Tomato-Village (Variant-a) làm miền thực địa thứ ba cho cà chua.

Chạy (sau scripts/prepare_species.py):
    .venv/bin/python scripts/prepare_tomato_village.py --data-dir data --workers 8

Đầu vào:
  data/raw/tomato_village/Variant-a(Multiclass Classification)/{train,val,test}/<lớp>/*.jpg
      (github.com/mamta-joshi-gehlot/Tomato-Village; Gehlot, Saxena & Gandhi, Multimedia Systems 2023)
  data/processed/species/manifest_tomato.csv   PV + PD (từ prepare_species.py)

Đầu ra:
  data/processed/tomatovillage/<lớp>/*.jpg                      ảnh đã thu nhỏ (cạnh ngắn 448)
  data/processed/species/manifest_tomato_base.csv               bản PV + PD gốc (giữ nguyên, tạo ở lần chạy đầu)
  data/processed/species/manifest_tomato.csv                    PV + PD + TV (3 lớp trùng)
  data/processed/species/manifest_tomato_village_ood.csv        5 lớp TV không có trong nghiên cứu (ảnh lạ khó)
  data/processed/species/stats_tomato.csv, qa/tv_*              thống kê, cặp gần trùng, ảnh bị loại

Các bước:
  1. Ánh xạ lớp: Early_blight, Late_blight, Healthy -> 3 lớp chung; 5 lớp còn lại (sâu vẽ bùa, virus
     héo đốm, thiếu Mg/N/K) không có trong PV/PD -> tập ảnh lạ, chỉ dùng để đánh giá.
  2. Thu nhỏ ảnh, tính pHash; bỏ ảnh hỏng.
  3. Bản tăng cường của tác giả: loại ảnh xoay có góc đen; gom nhóm ảnh gần giống (pHash Hamming <= 6,
     xét cả lật/xoay 90°) để chia tập; cặp có tương quan ảnh xám chuẩn hoá >= 0,95 là cùng một ảnh:
     cùng nhãn -> giữ một bản; khác nhãn -> loại cả cụm.
  4. Rò rỉ chéo với PV/PD cà chua (Hamming <= 4) -> bỏ bản TV (không đổi PV/PD vì các lần chạy đã có dùng chúng).
  5. Chia lại 70/15/15 theo nhóm, phân tầng theo lớp (không dùng cách chia sẵn của tác giả).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_data import (  # noqa: E402
    PHASH_MAX_DIST,
    UnionFind,
    contact_sheet,
    hash_many,
    near_pairs,
    phash_int,
)
from prepare_species import (  # noqa: E402
    PP_GROUP_DIST,
    PP_SHORT_SIDE,
    TOMATO,
    check,
    split_pp,
)

DATASET = "tomatovillage"
RAW_SUB = "tomato_village/Variant-a(Multiclass Classification)"
ID_MAP = {
    "Early_blight": "tomato_early_blight",
    "Late_blight": "tomato_late_blight",
    "Healthy": "tomato_healthy",
}
OOD_MAP = {
    "Leaf Miner": "tv_ood_leaf_miner",
    "Spotted Wilt Virus": "tv_ood_spotted_wilt_virus",
    "Magnesium Deficiency": "tv_ood_magnesium_deficiency",
    "Nitrogen Deficiency": "tv_ood_nitrogen_deficiency",
    "Pottassium Deficiency": "tv_ood_potassium_deficiency",
}
SAME_IMAGE_R = 0.95  # tương quan ảnh xám chuẩn hoá: >= 0,95 coi là cùng một ảnh (kiểm tra bằng mắt: 0,90-0,92 còn cặp ghép sai)
COLS = ["path", "dataset", "split", "unified_class", "group_id", "mask_path", "source_file"]


def list_raw(raw: Path) -> pd.DataFrame:
    rows = []
    for split_dir in sorted(raw.iterdir()):
        if not split_dir.is_dir():
            continue
        for cls_dir in sorted(split_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            name = cls_dir.name
            assert name in ID_MAP or name in OOD_MAP, f"lớp lạ: {name}"
            for f in sorted(cls_dir.iterdir()):
                if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    rows.append((str(f), split_dir.name, name))
    df = pd.DataFrame(rows, columns=["raw_path", "orig_split", "raw_class"])
    df["unified_class"] = df.raw_class.map(lambda c: ID_MAP.get(c, OOD_MAP.get(c)))
    df["is_ood"] = df.raw_class.isin(OOD_MAP)
    df["source_file"] = [
        f"{s}/{c}/{Path(p).name}" for p, s, c in zip(df.raw_path, df.orig_split, df.raw_class, strict=True)
    ]
    return df


def _resize_one(args):
    src, dst = args
    dst = Path(dst)
    try:
        if not dst.exists():
            with Image.open(src) as im:
                im.draft("RGB", (PP_SHORT_SIDE * 2, PP_SHORT_SIDE * 2))
                im = ImageOps.exif_transpose(im.convert("RGB"))
                w, h = im.size
                s = PP_SHORT_SIDE / min(w, h)
                if s < 1:
                    im = im.resize((round(w * s), round(h * s)), Image.BICUBIC)
                dst.parent.mkdir(parents=True, exist_ok=True)
                im.save(dst, quality=92)
        with Image.open(dst) as im:
            return str(imagehash.phash(im)), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def resize(df: pd.DataFrame, data_dir: Path, workers: int, removed: list) -> pd.DataFrame:
    df = df.copy()
    df["path"] = [
        f"processed/{DATASET}/{c}/{s}_{Path(p).stem.replace(' ', '_')}.jpg"
        for c, s, p in zip(df.unified_class, df.orig_split, df.raw_path, strict=True)
    ]
    assert df.path.is_unique, "trùng tên file đích"
    jobs = [(p, str(data_dir / q)) for p, q in zip(df.raw_path, df.path, strict=True)]
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_resize_one, jobs, chunksize=16))
    df["phash"] = [r[0] for r in res]
    for r, (_, err) in zip(df.itertuples(), res, strict=True):
        if r.phash is None:
            removed.append((DATASET, r.orig_split, r.source_file, f"anh hong: {err}"))
    bad = df.phash.isna()
    print(f"[TV] thu nho {len(df)} anh -> {PP_SHORT_SIDE}px, {int(bad.sum())} anh loi")
    return df[~bad].reset_index(drop=True)


DIHEDRAL = [None, Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM, Image.ROTATE_90,
            Image.ROTATE_180, Image.ROTATE_270, Image.TRANSPOSE, Image.TRANSVERSE]


def _features(path: str):
    """pHash của 8 phép lật/xoay 90°, ảnh xám 48x48 chuẩn hoá (không đổi khi chỉnh sáng/tương phản)
    của 8 phép đó, và cờ "góc đen" (ảnh đã bị xoay một góc bất kỳ để tăng cường)."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        g = np.asarray(im.convert("L"))
        k = max(4, min(g.shape) // 40)
        corners = [g[:k, :k], g[:k, -k:], g[-k:, :k], g[-k:, -k:]]
        black = sum(c.mean() < 20 for c in corners) >= 2
        small = im.convert("L").resize((48, 48), Image.BILINEAR)
        hashes, vecs = [], []
        for t in DIHEDRAL:
            hashes.append(phash_int(str(imagehash.phash(im if t is None else im.transpose(t)))))
            x = np.asarray(small if t is None else small.transpose(t), dtype=np.float32).ravel()
            vecs.append((x - x.mean()) / (x.std() + 1e-6))
    return hashes, np.stack(vecs), black


def dedup_and_group(df: pd.DataFrame, data_dir: Path, workers: int, removed: list, qa: Path) -> pd.DataFrame:
    """Repo chứa nhiều bản tăng cường của cùng một ảnh: đổi sáng/tương phản/ám màu, lật, xoay 90°,
    và xoay một góc bất kỳ (để lại góc đen, thường kèm phóng to).
      - ảnh có góc đen: loại hết (là bản xoay của ảnh khác, không phải lá mới; so khớp tự động không tin cậy
        vì ảnh còn bị phóng to)
      - cặp ứng viên: pHash <= 6 sau khi xét cả 8 phép lật/xoay 90°. pHash trên ảnh lá đặt giữa nền giấy trơn
        cho nhiều cặp trùng giả, nên cặp chỉ được coi là cùng ảnh khi tương quan ảnh xám chuẩn hoá
        (đã căn theo phép lật/xoay tốt nhất) r >= SAME_IMAGE_R
      - gom nhóm để chia tập: mọi cặp ứng viên (gom thừa chỉ làm việc chia chặt hơn, không gây rò rỉ)
      - cùng ảnh, khác nhãn: loại cả cụm (nhãn mâu thuẫn); cùng ảnh, cùng nhãn: giữ một bản"""
    with ProcessPoolExecutor(workers) as ex:
        feats = list(ex.map(_features, [str(data_dir / p) for p in df.path], chunksize=16))
    black = np.array([f[2] for f in feats])
    for row in df[black].itertuples():
        removed.append((DATASET, row.orig_split, row.source_file, "anh xoay tang cuong (goc den)"))
    keep = np.nonzero(~black)[0]
    df = df.iloc[keep].reset_index(drop=True)
    feats = [feats[k] for k in keep]
    print(f"[TV] bo {int(black.sum())} anh xoay tang cuong (goc den)")

    ph0 = [f[0][0] for f in feats]
    best = {}  # (i, j) -> (hamming, phép biến đổi áp lên j)
    for t in range(len(DIHEDRAL)):
        pt = [f[0][t] for f in feats]
        for i, j, d in near_pairs(ph0, pt, PP_GROUP_DIST):
            if i == j:
                continue
            key = (min(i, j), max(i, j))
            if key not in best or d < best[key][0]:
                best[key] = (d, t, i, j)
    pairs, r = [], []
    for (a, b), (d, t, i, j) in sorted(best.items()):
        pairs.append((a, b, d, t))
        r.append(float(feats[i][1][0] @ feats[j][1][t]) / feats[i][1].shape[1])
    lab = df.unified_class.to_numpy()
    pd.DataFrame(
        [(df.source_file[a], df.source_file[b], d, t, round(x, 4)) for (a, b, d, t), x in zip(pairs, r, strict=True)],
        columns=["a", "b", "hamming", "transform", "corr"],
    ).to_csv(qa / "tv_near_pairs.csv", index=False)

    grp, dup, conflict = UnionFind(len(df)), UnionFind(len(df)), set()
    for (i, j, _d, _t), x in zip(pairs, r, strict=True):
        grp.union(i, j)
        if x >= SAME_IMAGE_R:
            if lab[i] == lab[j]:
                dup.union(i, j)
            else:
                conflict.update((i, j))
    df = df.copy()
    df["group_id"] = [f"tv_{grp.find(k)}" for k in range(len(df))]
    df["dup_id"] = [dup.find(k) for k in range(len(df))]
    bad = df.dup_id.isin({df.dup_id[k] for k in conflict})
    for row in df[bad].itertuples():
        removed.append((DATASET, row.orig_split, row.source_file, "cung anh nhung nhan khac nhau"))
    df = df[~bad]
    first = df.groupby("dup_id").cumcount() == 0
    for row in df[~first].itertuples():
        removed.append((DATASET, row.orig_split, row.source_file, "ban sao / anh tang cuong (giu mot ban)"))
    n_dup = int((~first).sum())
    df = df[first].reset_index(drop=True)
    n_flip = sum(1 for (_a, _b, _d, t), x in zip(pairs, r, strict=True) if t and x >= SAME_IMAGE_R)
    print(
        f"[TV] {len(pairs)} cap ung vien; {int(sum(x >= SAME_IMAGE_R for x in r))} cap cung anh "
        f"({n_flip} cap qua lat/xoay 90); bo {int(bad.sum())} anh nhan mau thuan, {n_dup} ban sao; "
        f"con {len(df)} anh / {df.group_id.nunique()} nhom"
    )
    return df


def cross_leak(tv: pd.DataFrame, base: pd.DataFrame, data_dir: Path, workers: int, removed: list, qa: Path):
    h = hash_many([str(data_dir / p) for p in base.path], workers)
    o_ph = [phash_int(h[str(data_dir / p)][2]) for p in base.path]
    tv_ph = [phash_int(x) for x in tv.phash]
    rows, bad = [], set()
    for i, j, d in near_pairs(o_ph, tv_ph, PHASH_MAX_DIST):
        rows.append((base.dataset.iloc[i], base.path.iloc[i], base.split.iloc[i], tv.source_file[j], d))
        bad.add(j)
    pd.DataFrame(rows, columns=["other_dataset", "other_path", "other_split", "tv_file", "hamming"]).to_csv(
        qa / "tv_cross_matches.csv", index=False
    )
    for j in sorted(bad):
        removed.append((DATASET, tv.orig_split[j], tv.source_file[j], "trung voi anh PV/PD (ro ri cheo)"))
    print(f"[Cheo] {len(rows)} cap trung voi PV/PD: bo {len(bad)} anh TV")
    return tv.drop(index=sorted(bad)).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    data_dir = Path(args.data_dir).resolve()
    sp = data_dir / "processed" / "species"
    qa = sp / "qa"
    qa.mkdir(parents=True, exist_ok=True)
    removed: list = []

    # bản PV + PD gốc: giữ nguyên tuyệt đối (các lần chạy cà chua đã có dùng đúng các dòng này)
    base_f = sp / "manifest_tomato_base.csv"
    if not base_f.exists():
        cur = pd.read_csv(sp / "manifest_tomato.csv", keep_default_na=False)
        assert DATASET not in set(cur.dataset), "manifest_tomato.csv đã có TV nhưng thiếu bản gốc"
        shutil.copy2(sp / "manifest_tomato.csv", base_f)
    base = pd.read_csv(base_f, keep_default_na=False)

    raw = list_raw(data_dir / "raw" / RAW_SUB)
    print(f"[TV] {len(raw)} anh goc:\n{raw.groupby(['raw_class', 'orig_split']).size().unstack().to_string()}")
    tv = resize(raw, data_dir, args.workers, removed)
    tv = dedup_and_group(tv, data_dir, args.workers, removed, qa)
    tv = cross_leak(tv, base, data_dir, args.workers, removed, qa)
    tv["dataset"] = DATASET
    tv["mask_path"] = ""

    ind = split_pp(tv[~tv.is_ood].reset_index(drop=True))
    ood = tv[tv.is_ood].assign(split="test").reset_index(drop=True)

    man = pd.concat([base[COLS], ind[COLS]], ignore_index=True)
    man = man.sort_values(["dataset", "split", "unified_class", "path"]).reset_index(drop=True)
    check(man, data_dir, "tomato")
    assert set(man.unified_class) <= set(TOMATO)
    man.to_csv(sp / "manifest_tomato.csv", index=False)
    ood[COLS + ["raw_class"]].sort_values("path").to_csv(sp / "manifest_tomato_village_ood.csv", index=False)

    stats = man.pivot_table(
        index="unified_class", columns=["dataset", "split"], values="path", aggfunc="count", fill_value=0
    ).reindex(TOMATO)
    stats.to_csv(sp / "stats_tomato.csv")
    print("\n=== TOMATO (PV + PD + TV) ===")
    print(stats.to_string())
    print("\n=== TV: anh la (chi danh gia) ===")
    print(ood.raw_class.value_counts().to_string())

    sheet = pd.concat([ind.groupby("unified_class").head(8), ood.groupby("unified_class").head(8)])
    contact_sheet([data_dir / p for p in sheet.path], qa / "samples_tomatovillage.jpg")

    keep = {data_dir / p for p in pd.concat([ind.path, ood.path])}
    orphan = [f for f in (data_dir / "processed" / DATASET).rglob("*.jpg") if f not in keep]
    for f in orphan:
        f.unlink()
    print(f"\n[TV] xoa {len(orphan)} anh da thu nho nhung bi loai")

    log = pd.DataFrame(removed, columns=["dataset", "split", "item", "reason"])
    log.to_csv(qa / "tv_removed.csv", index=False)
    print("\nLy do loai:")
    print(log.groupby("reason").size().to_string() if len(log) else "(khong co)")


if __name__ == "__main__":
    main()
