"""Подготовка датасета human/transport/stuff из размеченных кадров.

Источник изображений: C:\\Users\\Никита\\Desktop\\human_detector\\data\\human
Источник разметки:    C:\\Users\\Никита\\Desktop\\human_detector\\data\\fire_labels (64 .txt)

Результат:
    dataset/images/{train,val}   - JPEG 1280x960
    dataset/labels/{train,val}   - YOLO txt
    data.yaml
Сплит групповой по локации (первые 3 буквы после train_), чтобы кадры одной
сцены не попадали одновременно в train и val.
"""
from __future__ import annotations

import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
SRC_IMAGES = ROOT / "data" / "human"
SRC_LABELS = ROOT / "data" / "fire_labels"
OUT = ROOT
DATASET = OUT / "dataset"

CLASSES = ["human", "transport", "stuff"]
VAL_LOCATIONS = {"GOR", "TRS"}  # 13 из 64 кадров
TARGET_SIZE = (1280, 960)  # 4:3, как исходники 4000x3000
JPEG_QUALITY = 92


def parse_label(path: Path) -> list[tuple[int, float, float, float, float]]:
    rows = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        assert len(parts) == 5, f"{path.name}:{lineno}: ожидалось 5 значений, got {parts}"
        cls = int(parts[0])
        cx, cy, w, h = (float(v) for v in parts[1:])
        assert 0 <= cls < len(CLASSES), f"{path.name}:{lineno}: класс {cls} вне 0..2"
        assert 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0, f"{path.name}:{lineno}: центр вне [0,1]"
        assert 0.0 < w <= 1.0 and 0.0 < h <= 1.0, f"{path.name}:{lineno}: размер вне (0,1]"
        assert cx - w / 2 >= -1e-6 and cx + w / 2 <= 1 + 1e-6, f"{path.name}:{lineno}: выход по X"
        assert cy - h / 2 >= -1e-6 and cy + h / 2 <= 1 + 1e-6, f"{path.name}:{lineno}: выход по Y"
        rows.append((cls, cx, cy, w, h))
    return rows


def main() -> None:
    label_files = [p for p in SRC_LABELS.glob("*.txt") if p.name != "classes.txt"]
    pairs = []
    for lbl in sorted(label_files):
        img = SRC_IMAGES / f"{lbl.stem}.JPG"
        assert img.exists(), f"нет изображения для {lbl.name}"
        pairs.append((img, lbl))

    assert len(pairs) == 64, f"ожидалось 64 пары, найдено {len(pairs)}"
    assert len({p[0] for p in pairs}) == len(pairs), "дубли изображений"

    if DATASET.exists():
        shutil.rmtree(DATASET)
    for split in ("train", "val"):
        (DATASET / "images" / split).mkdir(parents=True)
        (DATASET / "labels" / split).mkdir(parents=True)

    stats = defaultdict(Counter)
    split_count = Counter()
    for img_path, lbl_path in pairs:
        loc = lbl_path.stem.split("_")[1]
        split = "val" if loc in VAL_LOCATIONS else "train"

        with Image.open(img_path) as im:
            im = im.convert("RGB")
            if im.size != TARGET_SIZE:
                im = im.resize(TARGET_SIZE, Image.LANCZOS)
            im.save(DATASET / "images" / split / f"{lbl_path.stem}.jpg",
                    "JPEG", quality=JPEG_QUALITY, optimize=True)

        shutil.copy2(lbl_path, DATASET / "labels" / split / lbl_path.name)
        split_count[split] += 1
        for row in parse_label(lbl_path):
            stats[split][row[0]] += 1

    assert split_count["train"] + split_count["val"] == 64
    assert split_count["val"] == 13, split_count["val"]

    (OUT / "data.yaml").write_text(
        "\n".join([
            f"path: {DATASET.as_posix()}",
            "train: images/train",
            "val: images/val",
            "names:",
            *[f"  {i}: {name}" for i, name in enumerate(CLASSES)],
            "",
        ]),
        encoding="utf-8",
    )

    print("Датасет готов:", DATASET)
    for split in ("train", "val"):
        total = sum(stats[split].values())
        per_class = ", ".join(f"{CLASSES[c]}={stats[split][c]}" for c in range(len(CLASSES)))
        print(f"  {split}: {split_count[split]} изображений, {total} объектов ({per_class})")
    print("Все проверки разметки пройдены.")


if __name__ == "__main__":
    main()
