"""Bước 1 – làm sạch PlantVillage + PlantDoc (mọi loài, 27 lớp chung trong class_mapping.csv).
Ảnh các loài khác táo/cà chua được dùng làm tập "ảnh lạ" khi chọn ngưỡng năng lượng.
Sau bước này chạy scripts/prepare_species.py (bước 2).

Chạy:  .venv/bin/python scripts/prepare_data.py --data-dir data --workers 8

Đầu vào (data/raw):
  plantvillage/data.zip, plantvillage/splits/color_{train,test}.txt,
  plantvillage/leaf_grouping/leaf-map.json
  plantdoc/detection/  (clone của PlantDoc-Object-Detection-Dataset)

Đầu ra (data/processed):
  manifest.csv           mọi ảnh dùng cho huấn luyện / đánh giá
  removed_log.csv        ảnh bị loại và lý do
  stats_by_class.csv     số ảnh theo lớp x nguồn x tập
  qa/                    ảnh kiểm tra trực quan
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFile
from sklearn.model_selection import StratifiedGroupKFold

ImageFile.LOAD_TRUNCATED_IMAGES = False
SEED = 42
VAL_FOLDS = 7  # 1/7 ~ 14% của train làm validation
CROP_MARGIN = 0.10  # nới bounding box 10% mỗi phía
MIN_BOX_SIDE = 32  # bỏ lá có cạnh nhỏ hơn 32 px
PHASH_MAX_DIST = 4  # khoảng cách Hamming coi là "gần trùng"

SCRIPT_DIR = Path(__file__).resolve().parent


# ----------------------------------------------------------------------------- tiện ích
def safe_name(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return stem[:120]


def file_md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def hash_image(path_str: str):
    """Trả về (path, md5, phash_hex, w, h) hoặc (path, None, lỗi, 0, 0)."""
    p = Path(path_str)
    try:
        with Image.open(p) as im:
            im.load()
            w, h = im.size
            ph = str(imagehash.phash(im.convert("RGB")))
        return path_str, file_md5(p), ph, w, h
    except Exception as e:  # ảnh hỏng / không đọc được
        return path_str, None, f"{type(e).__name__}: {e}", 0, 0


def hash_many(paths: list[str], workers: int) -> dict:
    out = {}
    with ProcessPoolExecutor(workers) as ex:
        for res in ex.map(hash_image, paths, chunksize=64):
            out[res[0]] = res
    return out


def phash_int(h: str) -> int:
    return int(h, 16)


def near_pairs(a: list[int], b: list[int] | None, max_dist: int):
    """Các cặp (i, j) có Hamming <= max_dist. b=None: so trong cùng tập a (i<j)."""
    A = np.array(a, dtype=np.uint64)
    B = A if b is None else np.array(b, dtype=np.uint64)
    pairs = []
    for i in range(len(A)):
        x = np.bitwise_xor(B, A[i])
        # popcount 64-bit
        d = np.zeros(len(B), dtype=np.int64)
        for s in range(0, 64, 8):
            d += POP8[((x >> np.uint64(s)) & np.uint64(255)).astype(np.int64)]
        js = np.nonzero(d <= max_dist)[0]
        for j in js:
            if b is None and j <= i:
                continue
            pairs.append((i, int(j), int(d[j])))
    return pairs


POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.int64)


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def val_split(df: pd.DataFrame, label_col: str, group_col: str) -> pd.Series:
    """Tách validation theo nhóm (không để một nhóm nằm ở hai tập), phân tầng theo lớp."""
    sgkf = StratifiedGroupKFold(n_splits=VAL_FOLDS, shuffle=True, random_state=SEED)
    _, val_idx = next(sgkf.split(df, df[label_col], df[group_col]))
    is_val = pd.Series(False, index=df.index)
    is_val.iloc[val_idx] = True
    return is_val


# ----------------------------------------------------------------------------- PlantDoc
def extract_case_collisions(det: Path) -> dict:
    """Repo PlantDoc có các file chỉ khác chữ hoa/thường (vd. Peach-Leaf.jpg và peach-leaf.jpg).
    Trên macOS/Windows chỉ một bản được checkout, nên lấy từng bản từ git ra thư mục riêng."""
    import subprocess

    ls = subprocess.run(
        ["git", "-C", str(det), "ls-tree", "-r", "--name-only", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    groups = defaultdict(list)
    for p in ls:
        groups[p.lower()].append(p)
    out = {}
    for k, (_key, paths) in enumerate(sorted(groups.items())):
        if len(paths) < 2 or paths[0].lower().endswith(".xml"):
            continue
        for v, p in enumerate(sorted(paths)):
            dst = det.parent / "case_collisions" / f"{k:03d}_{v}" / Path(p).name
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                blob = subprocess.run(
                    ["git", "-C", str(det), "show", f"HEAD:{p}"],
                    capture_output=True,
                    check=True,
                ).stdout
                dst.write_bytes(blob)
            out[(Path(p).parent.name, Path(p).name)] = dst
    print(f"[PlantDoc] tach {len(out)} file trung ten khac hoa/thuong")
    return out


def prepare_plantdoc(
    raw: Path, out: Path, mapping: pd.DataFrame, workers: int, removed: list
):
    det = raw / "plantdoc" / "detection"
    pd2u = dict(
        zip(
            mapping.plantdoc_class.str.lower().str.split().str.join(" "),
            mapping.unified_class,
            strict=True,
        )
    )

    boxes = []
    for split, folder in [("train", "TRAIN"), ("test", "TEST")]:
        df = pd.read_csv(det / f"{split}_labels.csv")
        df["split"] = split
        df["folder"] = folder
        boxes.append(df)
    boxes = pd.concat(boxes, ignore_index=True)
    boxes["class_norm"] = (
        boxes["class"].astype(str).str.lower().str.split().str.join(" ")
    )
    boxes["unified_class"] = boxes["class_norm"].map(pd2u)

    # --- tìm file ảnh thật cho mỗi tên trong CSV
    files = {
        (f.parent.name, f.name): f
        for f in det.glob("*/*")
        if f.suffix.lower() not in {".xml"} and f.is_file()
    }
    files.update(extract_case_collisions(det))
    boxes["img_path"] = [
        str(files.get((fo, fn), ""))
        for fo, fn in zip(boxes.folder, boxes.filename, strict=True)
    ]
    missing = boxes[boxes.img_path == ""]
    for fn, sp in (
        missing[["filename", "split"]].drop_duplicates().itertuples(index=False)
    ):
        removed.append(("plantdoc", sp, fn, "khong tim thay file anh"))
    boxes = boxes[boxes.img_path != ""]

    unmapped = boxes[boxes.unified_class.isna()]
    for (cls, sp), n in unmapped.groupby(["class", "split"]).size().items():
        removed.append(
            ("plantdoc", sp, f"{n} box lop '{cls}'", "lop khong nam trong 27 lop chung")
        )
    boxes = boxes[boxes.unified_class.notna()].copy()

    # --- bảng ảnh (mỗi ảnh gốc một dòng)
    imgs = (
        boxes.groupby(["split", "img_path"])
        .agg(
            classes=("unified_class", lambda s: tuple(sorted(set(s)))),
            csv_w=("width", "first"),
            csv_h=("height", "first"),
            filename=("filename", "first"),
        )
        .reset_index()
    )
    print(f"[PlantDoc] {len(imgs)} anh, {len(boxes)} box sau khi loc lop")

    hashes = hash_many(imgs.img_path.tolist(), workers)
    imgs["md5"] = [hashes[p][1] for p in imgs.img_path]
    imgs["phash"] = [hashes[p][2] for p in imgs.img_path]
    imgs["real_w"] = [hashes[p][3] for p in imgs.img_path]
    imgs["real_h"] = [hashes[p][4] for p in imgs.img_path]

    bad = imgs[imgs.md5.isna()]
    for r in bad.itertuples():
        removed.append(("plantdoc", r.split, r.filename, f"anh loi: {r.phash}"))
    imgs = imgs[imgs.md5.notna()].copy()

    size_bad = imgs[(imgs.real_w != imgs.csv_w) | (imgs.real_h != imgs.csv_h)]
    for r in size_bad.itertuples():
        removed.append(
            (
                "plantdoc",
                r.split,
                r.filename,
                f"kich thuoc anh {r.real_w}x{r.real_h} khac nhan {r.csv_w}x{r.csv_h}",
            )
        )
    imgs = imgs[~imgs.index.isin(size_bad.index)].reset_index(drop=True)

    # --- gom nhóm ảnh trùng / gần trùng (md5 giống hoặc phash gần)
    uf = UnionFind(len(imgs))
    by_md5 = defaultdict(list)
    for i, m in enumerate(imgs.md5):
        by_md5[m].append(i)
    for idx in by_md5.values():
        for j in idx[1:]:
            uf.union(idx[0], j)
    for i, j, _ in near_pairs([phash_int(h) for h in imgs.phash], None, PHASH_MAX_DIST):
        uf.union(i, j)
    imgs["dup_group"] = [uf.find(i) for i in range(len(imgs))]

    keep = np.ones(len(imgs), dtype=bool)
    dup_rows = []
    for g, sub in imgs.groupby("dup_group"):
        if len(sub) == 1:
            continue
        labels = set(sub.classes)
        splits = set(sub.split)
        for r in sub.itertuples():
            dup_rows.append((g, r.split, r.filename, "|".join(r.classes)))
        if len(labels) > 1:  # nhãn mâu thuẫn -> loại cả nhóm
            for r in sub.itertuples():
                keep[r.Index] = False
                removed.append(
                    ("plantdoc", r.split, r.filename, "anh trung nhung nhan mau thuan")
                )
        elif splits == {"train", "test"}:  # rò rỉ -> giữ bản test, bỏ bản train
            for r in sub[sub.split == "train"].itertuples():
                keep[r.Index] = False
                removed.append(
                    (
                        "plantdoc",
                        r.split,
                        r.filename,
                        "trung voi anh trong test (ro ri)",
                    )
                )
            for r in sub[sub.split == "test"].iloc[1:].itertuples():
                keep[r.Index] = False
                removed.append(
                    ("plantdoc", r.split, r.filename, "anh trung lap trong cung tap")
                )
        else:  # trùng trong cùng một tập -> giữ 1
            for r in sub.iloc[1:].itertuples():
                keep[r.Index] = False
                removed.append(
                    ("plantdoc", r.split, r.filename, "anh trung lap trong cung tap")
                )
    pd.DataFrame(
        dup_rows, columns=["dup_group", "split", "filename", "classes"]
    ).to_csv(out / "qa" / "plantdoc_duplicate_groups.csv", index=False)
    imgs = imgs[keep].reset_index(drop=True)
    imgs["source_id"] = [f"pd_{i:05d}" for i in range(len(imgs))]
    print(f"[PlantDoc] con {len(imgs)} anh sau khi loai trung lap")
    return boxes, imgs


def crop_plantdoc(
    boxes: pd.DataFrame, imgs: pd.DataFrame, out: Path, removed: list
) -> pd.DataFrame:
    rows = []
    sid = dict(zip(imgs.img_path, imgs.source_id, strict=True))
    boxes = boxes[boxes.img_path.isin(sid)]
    for img_path, sub in boxes.groupby("img_path"):
        source_id = sid[img_path]
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            W, H = im.size
            for k, b in enumerate(sub.itertuples()):
                x1, y1, x2, y2 = (
                    float(b.xmin),
                    float(b.ymin),
                    float(b.xmax),
                    float(b.ymax),
                )
                w, h = x2 - x1, y2 - y1
                if min(w, h) < MIN_BOX_SIDE:
                    removed.append(
                        (
                            "plantdoc_crop",
                            b.split,
                            f"{b.filename}#{k}",
                            f"box qua nho ({w:.0f}x{h:.0f})",
                        )
                    )
                    continue
                box = (
                    int(max(0, x1 - CROP_MARGIN * w)),
                    int(max(0, y1 - CROP_MARGIN * h)),
                    int(min(W, x2 + CROP_MARGIN * w)),
                    int(min(H, y2 + CROP_MARGIN * h)),
                )
                rel = (
                    Path("processed/plantdoc_crops")
                    / b.split
                    / b.unified_class
                    / f"{source_id}_{k:02d}.jpg"
                )
                dst = out.parent / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                im.crop(box).save(dst, quality=95)
                rows.append(
                    dict(
                        path=str(rel),
                        dataset="plantdoc_crop",
                        split=b.split,
                        unified_class=b.unified_class,
                        group_id=source_id,
                        mask_path="",
                        source_file=Path(img_path).name,
                    )
                )
    return pd.DataFrame(rows)


def plantdoc_full_images(imgs: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Ảnh nguyên của PlantDoc (chỉ ảnh có đúng một lớp) - dùng làm tập đánh giá phụ."""
    rows = []
    for r in imgs.itertuples():
        if len(r.classes) != 1:
            continue
        rel = (
            Path("processed/plantdoc_full")
            / r.split
            / r.classes[0]
            / f"{r.source_id}.jpg"
        )
        dst = out.parent / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(r.img_path) as im:
            im.convert("RGB").save(dst, quality=95)
        rows.append(
            dict(
                path=str(rel),
                dataset="plantdoc_full",
                split=r.split,
                unified_class=r.classes[0],
                group_id=r.source_id,
                mask_path="",
                source_file=Path(r.img_path).name,
            )
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- PlantVillage
def load_leaf_map(path: Path) -> dict:
    """leaf-map.json có dạng {"mary_hl 9155": ["Raspberry___healthy:::12.0", ...]}.
    Khóa là phần tên ảnh sau '___' (chữ thường). Trả về {(lớp, khóa): leaf_id}."""
    data = json.loads(path.read_text())
    out = {}
    for key, entries in data.items():
        for e in entries:
            cls, leaf = e.split(":::")
            out[(cls, key.strip().lower())] = f"{cls}:::{leaf}"
    return out


def leaf_key(split_path: str) -> str:
    stem = Path(split_path).stem
    return stem.split("___", 1)[-1].strip().lower()


def prepare_plantvillage(
    raw: Path, out: Path, mapping: pd.DataFrame, removed: list
) -> pd.DataFrame:
    pv = raw / "plantvillage"
    pv2u = dict(zip(mapping.plantvillage_class, mapping.unified_class, strict=True))
    zf = zipfile.ZipFile(pv / "data.zip")
    names = zf.namelist()
    # Tìm tiền tố thư mục trong zip ứng với "raw/color/..." của file split
    color_entries = [n for n in names if "/color/" in n and not n.endswith("/")]
    prefix = color_entries[0].split("color/")[0]
    print(f"[PlantVillage] zip co {len(names)} muc, tien to '{prefix}'")

    split_rows = []
    for split in ["train", "test"]:
        for line in (pv / "splits" / f"color_{split}.txt").read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            cls = line.split("/")[-2]
            split_rows.append((split, line, cls))
    df = pd.DataFrame(split_rows, columns=["split", "split_path", "pv_class"])
    df["unified_class"] = df.pv_class.map(pv2u)
    dropped = df[df.unified_class.isna()]
    for (c, sp), n in dropped.groupby(["pv_class", "split"]).size().items():
        removed.append(
            (
                "plantvillage",
                sp,
                f"{n} anh lop '{c}'",
                "lop khong nam trong 27 lop chung",
            )
        )
    df = df[df.unified_class.notna()].copy()

    # leaf_id
    leaf_map = load_leaf_map(pv / "leaf_grouping" / "leaf-map.json")
    keys = df.split_path.map(leaf_key)
    df["leaf_id"] = [
        leaf_map.get((c, k)) for c, k in zip(df.pv_class, keys, strict=True)
    ]
    n_no_leaf = int(df.leaf_id.isna().sum())
    df["leaf_id"] = df.leaf_id.fillna("img_" + df.pv_class + "/" + keys)
    df["group_id"] = "pv_" + df.leaf_id.astype(str)
    print(
        f"[PlantVillage] {len(df)} anh; {n_no_leaf} anh khong co leaf_id (dung chinh anh lam nhom)"
    )
    # cách chia gốc có vài lá nằm ở cả train và test -> bỏ bản train của các lá đó
    cross = df.groupby("group_id").split.nunique()
    cross = set(cross[cross > 1].index)
    bad = df[df.group_id.isin(cross) & (df.split == "train")]
    for r in bad.itertuples():
        removed.append(
            (
                "plantvillage",
                "train",
                r.split_path,
                "cung la voi anh trong test (ro ri)",
            )
        )
    df = df.drop(bad.index)

    # giải nén ảnh color của 27 lớp + ảnh segmented tương ứng của tập train
    name_set = set(names)
    seg_index = {}
    for n in names:
        if "/segmented/" in n and not n.endswith("/"):
            key = (n.split("/")[-2], Path(n).stem.replace("_final_masked", "").lower())
            seg_index[key] = n

    paths, masks = [], []
    for r in df.itertuples():
        zname = (
            prefix + r.split_path.split("raw/", 1)[-1]
            if r.split_path.startswith("raw/")
            else prefix + r.split_path
        )
        if zname not in name_set:
            paths.append(None)
            masks.append("")
            continue
        rel = (
            Path("processed/plantvillage/color")
            / r.unified_class
            / safe_name(Path(r.split_path).name)
        )
        dst = out.parent / rel
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(zf.read(zname))
        paths.append(str(rel))
        mrel = ""
        if r.split == "train":
            sname = seg_index.get((r.pv_class, Path(r.split_path).stem.lower()))
            if sname:
                mrel = (
                    Path("processed/plantvillage/segmented")
                    / r.unified_class
                    / safe_name(Path(sname).name)
                )
                mdst = out.parent / mrel
                if not mdst.exists():
                    mdst.parent.mkdir(parents=True, exist_ok=True)
                    mdst.write_bytes(zf.read(sname))
                mrel = str(mrel)
        masks.append(mrel)
    df["path"] = paths
    df["mask_path"] = masks
    miss = df[df.path.isna()]
    for r in miss.itertuples():
        removed.append(
            ("plantvillage", r.split, r.split_path, "khong co trong data.zip")
        )
    df = df[df.path.notna()].copy()
    df["dataset"] = "plantvillage"
    df["source_file"] = df.split_path.map(lambda p: Path(p).name)
    return df[
        [
            "path",
            "dataset",
            "split",
            "unified_class",
            "group_id",
            "mask_path",
            "source_file",
        ]
    ]


# ----------------------------------------------------------------------------- rò rỉ chéo
def cross_dataset_leak(
    manifest: pd.DataFrame, data_dir: Path, workers: int, removed: list
) -> pd.DataFrame:
    """Loại ảnh PlantDoc trùng với PlantVillage:
       - PlantDoc test trùng bất kỳ ảnh PV nào  -> bỏ khỏi PlantDoc test
       - PlantDoc train/val trùng PV test       -> bỏ khỏi PlantDoc train/val
    So trên ảnh nguyên của PlantDoc (group_id) với ảnh color của PV."""
    pv = manifest[manifest.dataset == "plantvillage"]
    pdf = manifest[manifest.dataset == "plantdoc_full"]
    pv_h = hash_many([str(data_dir / p) for p in pv.path], workers)
    pd_h = hash_many([str(data_dir / p) for p in pdf.path], workers)
    pv_ph = [phash_int(pv_h[str(data_dir / p)][2]) for p in pv.path]
    pd_ph = [phash_int(pd_h[str(data_dir / p)][2]) for p in pdf.path]
    pairs = near_pairs(pd_ph, pv_ph, PHASH_MAX_DIST)
    bad_groups = set()
    rows = []
    pv_split = pv.split.to_numpy()
    for i, j, d in pairs:
        r = pdf.iloc[i]
        rows.append((r.group_id, r.split, pv.iloc[j].path, pv_split[j], d))
        if r.split == "test" or pv_split[j] == "test":
            bad_groups.add(r.group_id)
    pd.DataFrame(
        rows,
        columns=["plantdoc_group", "plantdoc_split", "pv_path", "pv_split", "hamming"],
    ).to_csv(data_dir / "processed" / "qa" / "cross_dataset_matches.csv", index=False)
    for g in sorted(bad_groups):
        removed.append(("plantdoc", "", g, "trung voi anh PlantVillage (ro ri cheo)"))
    print(f"[Cheo] {len(pairs)} cap gan trung, loai {len(bad_groups)} anh PlantDoc")
    return manifest[
        ~(
            manifest.dataset.str.startswith("plantdoc")
            & manifest.group_id.isin(bad_groups)
        )
    ]


# ----------------------------------------------------------------------------- QA
def contact_sheet(paths: list[Path], dst: Path, cols=8, size=128):
    if not paths:
        return
    rows = (len(paths) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * size, rows * size), "white")
    for k, p in enumerate(paths):
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                im.thumbnail((size, size))
                sheet.paste(im, ((k % cols) * size, (k // cols) * size))
        except Exception:
            pass
    dst.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dst, quality=85)


def make_qa(
    manifest: pd.DataFrame, boxes: pd.DataFrame, data_dir: Path, mapping: pd.DataFrame
):
    qa = data_dir / "processed" / "qa"
    rng = np.random.default_rng(SEED)
    # 1) lưới ảnh từng lớp: hàng trên PV, hàng dưới PlantDoc
    for cls in mapping.unified_class:
        pv = manifest[
            (manifest.dataset == "plantvillage") & (manifest.unified_class == cls)
        ].path.tolist()
        pc = manifest[
            (manifest.dataset == "plantdoc_crop") & (manifest.unified_class == cls)
        ].path.tolist()
        pick = lambda L, n: (
            [data_dir / L[i] for i in rng.choice(len(L), min(n, len(L)), replace=False)]
            if L
            else []
        )
        contact_sheet(pick(pv, 16) + pick(pc, 24), qa / "class_grids" / f"{cls}.jpg")
    # 2) vẽ box lên vài ảnh PlantDoc gốc
    sample = boxes.drop_duplicates("img_path").sample(12, random_state=SEED).img_path
    for k, ip in enumerate(sample):
        with Image.open(ip) as im:
            im = im.convert("RGB")
            dr = ImageDraw.Draw(im)
            for b in boxes[boxes.img_path == ip].itertuples():
                dr.rectangle(
                    [b.xmin, b.ymin, b.xmax, b.ymax], outline=(255, 0, 0), width=3
                )
                dr.text(
                    (b.xmin + 3, b.ymin + 3), str(b.unified_class), fill=(255, 0, 0)
                )
            im.thumbnail((800, 800))
            (qa / "bbox_check").mkdir(parents=True, exist_ok=True)
            im.save(qa / "bbox_check" / f"{k:02d}.jpg", quality=85)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    data_dir = Path(args.data_dir).resolve()
    raw, out = data_dir / "raw", data_dir / "processed"
    (out / "qa").mkdir(parents=True, exist_ok=True)
    mapping = pd.read_csv(SCRIPT_DIR / "class_mapping.csv")
    mapping.to_csv(out / "class_mapping.csv", index=False)
    removed: list = []

    boxes, pd_imgs = prepare_plantdoc(raw, out, mapping, args.workers, removed)
    crops = crop_plantdoc(boxes, pd_imgs, out, removed)
    full = plantdoc_full_images(pd_imgs, out)
    pv = prepare_plantvillage(raw, out, mapping, removed)

    manifest = pd.concat([pv, crops, full], ignore_index=True)
    manifest = cross_dataset_leak(manifest, data_dir, args.workers, removed)

    # validation: tách từ train theo nhóm, riêng từng nguồn
    manifest = manifest.reset_index(drop=True)
    for ds in ["plantvillage", "plantdoc_crop"]:
        m = (manifest.dataset == ds) & (manifest.split == "train")
        is_val = val_split(manifest[m], "unified_class", "group_id")
        manifest.loc[is_val[is_val].index, "split"] = "val"
    # ảnh nguyên PlantDoc theo cùng phân chia với lá cắt
    val_groups = set(
        manifest[
            (manifest.dataset == "plantdoc_crop") & (manifest.split == "val")
        ].group_id
    )
    m = (manifest.dataset == "plantdoc_full") & manifest.group_id.isin(val_groups)
    manifest.loc[m, "split"] = "val"

    # kiểm tra tự động: không nhóm nào nằm ở hai tập
    for ds in manifest.dataset.unique():
        sub = manifest[manifest.dataset == ds]
        leak = sub.groupby("group_id").split.nunique()
        assert (leak == 1).all(), f"{ds}: {int((leak > 1).sum())} nhom nam o nhieu tap"
    assert manifest.path.map(lambda p: (data_dir / p).exists()).all(), (
        "co duong dan khong ton tai"
    )

    manifest = manifest.sort_values(["dataset", "split", "unified_class", "path"])
    manifest.to_csv(out / "manifest.csv", index=False)
    pd.DataFrame(removed, columns=["dataset", "split", "item", "reason"]).to_csv(
        out / "removed_log.csv", index=False
    )
    stats = manifest.pivot_table(
        index="unified_class",
        columns=["dataset", "split"],
        values="path",
        aggfunc="count",
        fill_value=0,
    )
    stats.to_csv(out / "stats_by_class.csv")
    make_qa(
        manifest, boxes[boxes.img_path.isin(set(pd_imgs.img_path))], data_dir, mapping
    )

    print("\n=== TONG KET ===")
    print(manifest.groupby(["dataset", "split"]).size().unstack(fill_value=0))
    print("\nLy do loai:")
    print(
        pd.DataFrame(removed, columns=["dataset", "split", "item", "reason"])
        .groupby(["dataset", "reason"])
        .size()
    )


if __name__ == "__main__":
    main()
