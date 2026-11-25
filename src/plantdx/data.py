"""Dữ liệu theo loài (táo, cà chua): manifest, tiền xử lý, cặp ảnh (x, x_nền), lấy mẫu cân bằng, ảnh lạ.

Đọc data/processed/species/manifest_{apple,tomato}.csv (tạo bởi scripts/prepare_species.py).
Miền (domain): pv = PlantVillage, pp = Plant Pathology, pd = PlantDoc (lá cắt), pdfull = PlantDoc ảnh nguyên,
tv = Tomato-Village (lá cà chua hái ngoài ruộng, chụp trên nền giấy; scripts/prepare_tomato_village.py).
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision.transforms import v2 as T

from .background import replace_background

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

DOMAINS = {
    "pv": "plantvillage",
    "pp": "plantpathology",
    "pd": "plantdoc_crop",
    "pdfull": "plantdoc_full",
    "tv": "tomatovillage",
}
CLASSES = {
    "apple": ["apple_healthy", "apple_scab", "apple_rust", "apple_black_rot"],
    "tomato": [
        "tomato_healthy",
        "tomato_bacterial_spot",
        "tomato_early_blight",
        "tomato_late_blight",
        "tomato_leaf_mold",
        "tomato_septoria_leaf_spot",
        "tomato_yellow_leaf_curl_virus",
        "tomato_mosaic_virus",
    ],
}
OTHER = {"apple": "tomato", "tomato": "apple"}
VI_NAMES = {
    "apple_healthy": "Táo – Lá khỏe mạnh",
    "apple_scab": "Táo – Bệnh ghẻ (vảy táo)",
    "apple_rust": "Táo – Bệnh gỉ sắt",
    "apple_black_rot": "Táo – Thối đen / đốm mắt ếch",
    "tomato_healthy": "Cà chua – Lá khỏe mạnh",
    "tomato_bacterial_spot": "Cà chua – Đốm vi khuẩn",
    "tomato_early_blight": "Cà chua – Bệnh úa sớm",
    "tomato_late_blight": "Cà chua – Bệnh mốc sương",
    "tomato_leaf_mold": "Cà chua – Bệnh mốc lá",
    "tomato_septoria_leaf_spot": "Cà chua – Đốm lá Septoria",
    "tomato_yellow_leaf_curl_virus": "Cà chua – Virus xoăn vàng lá",
    "tomato_mosaic_virus": "Cà chua – Virus khảm",
}


def _abs(m: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    m = m.copy()
    m["abs_path"] = [str(data_dir / p) for p in m.path]
    m["abs_mask"] = [
        str(data_dir / p) if isinstance(p, str) and p else "" for p in m.mask_path
    ]
    return m


def load_species(species: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    m = pd.read_csv(
        data_dir / "processed" / "species" / f"manifest_{species}.csv",
        keep_default_na=False,
    )
    m["label"] = m.unified_class.map({c: i for i, c in enumerate(CLASSES[species])})
    assert m.label.notna().all(), "manifest có lớp lạ"
    return _abs(m, data_dir)


def domains_of(species: str) -> list[str]:
    return ["pv", "pp", "pd"] if species == "apple" else ["pv", "pd", "tv"]


def select(m: pd.DataFrame, domains: list[str], split: str) -> pd.DataFrame:
    return m[
        m.dataset.isin([DOMAINS[d] for d in domains]) & (m.split == split)
    ].reset_index(drop=True)


def ood_sets(species: str, split: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Ảnh "không phải lá của loài này".
    val : các loài khác táo/cà chua trong manifest 27 lớp (ngô, nho, khoai tây, ...) -> chọn ngưỡng
    test: loài mục tiêu còn lại (táo <-> cà chua) + các loài khác                    -> đánh giá"""
    base = pd.read_csv(data_dir / "processed" / "manifest.csv", keep_default_na=False)
    base = base[
        base.dataset.isin(["plantvillage", "plantdoc_crop"]) & (base.split == split)
    ]
    other = base[~base.unified_class.str.startswith(("apple_", "tomato_"))]
    if split == "test":
        o = load_species(OTHER[species], data_dir)
        o = o[
            o.dataset.isin(["plantvillage", "plantdoc_crop", "plantpathology"])
            & (o.split == "test")
        ]
        other = pd.concat([other, o[other.columns.intersection(o.columns)]])
    other = other.assign(label=-1)
    return _abs(other, data_dir).reset_index(drop=True)


def near_ood_sets(species: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Ảnh lạ "khó": lá cà chua Tomato-Village mắc bệnh/thiếu chất không có trong 8 lớp (sâu vẽ bùa, virus héo
    đốm, thiếu Mg/N/K). Cùng loài, cùng miền chụp với ảnh TV nên khó tách hơn ảnh loài khác. Chỉ dùng để đánh giá;
    táo không có tập này (trả về bảng rỗng)."""
    f = data_dir / "processed" / "species" / "manifest_tomato_village_ood.csv"
    if species != "tomato" or not f.exists():
        return pd.DataFrame(columns=["path", "mask_path", "label", "abs_path", "abs_mask"])
    o = pd.read_csv(f, keep_default_na=False).assign(label=-1)
    return _abs(o, data_dir).reset_index(drop=True)


def build_transforms(img_size: int, mean, std, train: bool):
    """Đánh giá: cạnh ngắn -> 1,14·img_size, cắt giữa. Huấn luyện: crop/lật/xoay/màu nhẹ."""
    norm = [T.ToImage(), T.ToDtype(torch.float32, scale=True), T.Normalize(mean, std)]
    if not train:
        return T.Compose(
            [T.Resize(int(img_size * 1.14)), T.CenterCrop(img_size)] + norm
        )
    aug = [
        T.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(0.2),
        T.RandomRotation(20),
        T.ColorJitter(0.15, 0.15, 0.15, 0.02),
    ]
    return T.Compose(aug + norm)


class PairDataset(Dataset):
    """Trả về (x, x_bg, has_bg, y).
    x_bg là cùng chiếc lá đặt trên nền thực địa tổng hợp (chỉ có với ảnh PlantVillage có mặt nạ);
    dùng cho loss nhất quán nền. Ảnh không có mặt nạ: x_bg = x, has_bg = 0."""

    def __init__(
        self, df: pd.DataFrame, transform, img_size: int = 224, pair: bool = False
    ):
        self.paths, self.masks = df.abs_path.tolist(), df.abs_mask.tolist()
        self.labels = df.label.astype(int).tolist()
        self.tf, self.pair = transform, pair
        self.canvas = int(img_size * 1.3)

    def __len__(self):
        return len(self.paths)

    def _open(self, p):
        img = Image.open(p)
        img.draft("RGB", (self.canvas, self.canvas))
        return img.convert("RGB")

    def __getitem__(self, i):
        x = self.tf(self._open(self.paths[i]))
        if not self.pair:
            return x, self.labels[i]
        bg = None
        if self.masks[i]:
            bg = replace_background(
                self.masks[i],
                self.canvas,
                np.random.default_rng(random.getrandbits(32)),
            )
        if bg is None:
            return x, x, 0, self.labels[i]
        return x, self.tf(bg), 1, self.labels[i]


def balanced_sampler(
    df: pd.DataFrame, num_samples: int, seed: int
) -> WeightedRandomSampler:
    """Mỗi miền chiếm tỉ lệ bằng nhau; trong mỗi miền các lớp có mặt được lấy đều nhau."""
    n_dom = df.dataset.nunique()
    counts = df.groupby(["dataset", "label"]).size()
    n_cls = df.groupby("dataset").label.nunique()
    w = [
        1.0 / (n_dom * n_cls[d] * counts[(d, lab)])
        for d, lab in zip(df.dataset, df.label, strict=True)
    ]
    g = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        torch.tensor(w, dtype=torch.double), num_samples, replacement=True, generator=g
    )
