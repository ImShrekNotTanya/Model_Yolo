"""Сборка 5-классового датасета для единой SAR-модели.

Классы: 0 human, 1 transport, 2 stuff (в интерфейсе маппится в marker),
3 smoke, 4 fire. postburn (5) отбрасывается: в задаче он не нужен.

Сплит полностью повторяет sar_3class (чистый test GOR/TRS):
  train 134 / val 16 (Egor) / test 13 (Polina GOR/TRS).

Картинки берутся уже ресайзнутые из sar_3class. Метки train/val пересобираются
из исходного sar_source с сохранением smoke/fire; test остаётся 3-классовым,
потому что на этих 13 кадрах smoke/fire не размечены.
"""
from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "sar_3class" / "dataset"
SAR = ROOT / "data" / "sar_source" / "datasets" / "sar"
OUT = ROOT / "sar_5class"
DATASET = OUT / "dataset"

CLASSES = ["human", "transport", "stuff", "smoke", "fire"]


def source_label(stem: str) -> Path:
    for split in ("val", "train"):
        p = SAR / "labels" / split / f"{stem}.txt"
        if p.exists():
            return p
    raise FileNotFoundError(f"нет исходной метки для {stem}")


def filtered_label(path: Path) -> list[str]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        assert len(parts) == 5, f"{path.name}: {line}"
        cls = int(parts[0])
        if cls < len(CLASSES):
            rows.append(" ".join(parts))
    return rows


def yaml_text(split: str) -> str:
    return "\n".join([
        f"path: {DATASET.as_posix()}",
        "train: images/train",
        f"val: images/{split}",
        "names:",
        *[f"  {i}: {name}" for i, name in enumerate(CLASSES)],
        "",
    ])


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    for split in ("train", "val", "test"):
        (DATASET / "images" / split).mkdir(parents=True)
        (DATASET / "labels" / split).mkdir(parents=True)

    stats = defaultdict(Counter)
    zero = defaultdict(list)
    manifest = []

    for split in ("train", "val", "test"):
        for img in sorted((SRC / "images" / split).glob("*.jpg")):
            stem = img.stem
            shutil.copy2(img, DATASET / "images" / split / img.name)
            if split == "test":
                rows = [line.strip() for line in
                        (SRC / "labels" / "test" / f"{stem}.txt").read_text(
                            encoding="utf-8").splitlines() if line.strip()]
            else:
                rows = filtered_label(source_label(stem))
            (DATASET / "labels" / split / f"{stem}.txt").write_text(
                "\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
            for row in rows:
                stats[split][int(row.split()[0])] += 1
            if not rows:
                zero[split].append(stem)
            manifest.append({"stem": stem, "split": split,
                             "source": "egor" if stem.startswith("egor_") else "polina"})

    (OUT / "data.yaml").write_text(yaml_text("val"), encoding="utf-8")
    (OUT / "test.yaml").write_text(yaml_text("test"), encoding="utf-8")
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("sar_5class dataset ready:", DATASET)
    for split in ("train", "val", "test"):
        n = len(list((DATASET / "images" / split).glob("*.jpg")))
        total = sum(stats[split].values())
        per = ", ".join(f"{CLASSES[c]}={stats[split][c]}" for c in range(len(CLASSES)))
        print(f"  {split}: {n} images, {total} boxes ({per}), background={len(zero[split])}")


if __name__ == "__main__":
    main()
