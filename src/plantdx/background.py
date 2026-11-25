"""Nền thực địa tổng hợp: tách lá PlantVillage (ảnh segmented, nền đen) và đặt lên nền đất / cỏ / tán lá...

Dùng để tạo ảnh x_nền cho loss nhất quán nền. Không dùng ảnh PlantDoc hay Plant Pathology làm nền,
nên không rò rỉ thông tin từ các miền kiểm tra.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

PALETTES = [
    [(92, 64, 40), (130, 96, 60), (60, 45, 30)],  # đất
    [(60, 110, 40), (95, 150, 60), (35, 75, 30)],  # cỏ / tán lá
    [(120, 120, 110), (170, 170, 160), (80, 80, 75)],  # bê tông / đá
    [(150, 170, 200), (200, 210, 225), (110, 130, 160)],  # trời
    [(70, 90, 50), (140, 120, 80), (40, 50, 35)],  # hỗn hợp đất - cỏ
]


def _noise(h, w, scale, rng):
    small = rng.random((max(2, h // scale), max(2, w // scale), 1)).astype(np.float32)
    img = Image.fromarray((small[..., 0] * 255).astype(np.uint8)).resize(
        (w, h), Image.BICUBIC
    )
    return np.asarray(img, dtype=np.float32)[..., None] / 255.0


def random_background(size: int, rng: np.random.Generator) -> Image.Image:
    """Nền nhiễu nhiều tầng theo một bảng màu ngẫu nhiên, đôi khi thêm vài "lá" mờ phía sau."""
    _base, light, dark = (
        np.array(c, np.float32) for c in PALETTES[rng.integers(len(PALETTES))]
    )
    n = (
        0.5 * _noise(size, size, 4, rng)
        + 0.3 * _noise(size, size, 16, rng)
        + 0.2 * _noise(size, size, 64, rng)
    )
    img = dark + (light - dark) * n
    img += rng.random(3) * 40 - 20
    pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    if rng.random() < 0.6:
        d = ImageDraw.Draw(pil)
        for _ in range(rng.integers(3, 12)):
            cx, cy = rng.integers(0, size, 2)
            rx, ry = rng.integers(size // 12, size // 4, 2)
            g = int(rng.integers(70, 170))
            d.ellipse(
                [cx - rx, cy - ry, cx + rx, cy + ry],
                fill=(int(g * 0.5), g, int(g * 0.4)),
            )
        pil = pil.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(2, 6))))
    return pil


def replace_background(
    seg_path: str, size: int, rng: np.random.Generator
) -> Image.Image | None:
    """Tách lá từ ảnh segmented (nền đen) và dán lên nền tổng hợp, có co giãn, xoay, dịch vị trí."""
    try:
        seg = Image.open(seg_path).convert("RGB")
    except Exception:
        return None
    arr = np.asarray(seg)
    mask = Image.fromarray(((arr.max(axis=2) > 25) * 255).astype(np.uint8)).filter(
        ImageFilter.MinFilter(3)
    )
    mask = mask.filter(ImageFilter.GaussianBlur(1.5))
    canvas = random_background(size, rng)
    leaf_size = int(size * rng.uniform(0.55, 0.95))
    leaf = seg.resize((leaf_size, leaf_size), Image.BILINEAR)
    m = mask.resize((leaf_size, leaf_size), Image.BILINEAR)
    angle = float(rng.uniform(-30, 30))
    leaf, m = leaf.rotate(angle, expand=False), m.rotate(angle, expand=False)
    x, y = rng.integers(0, size - leaf_size + 1, 2)
    canvas.paste(leaf, (int(x), int(y)), m)
    return canvas
