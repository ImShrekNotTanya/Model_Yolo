"""Сборка чистого 3-классового датасета из sar_source/datasets/sar.

Классы: 0 human, 1 transport, 2 stuff. Остальные (smoke/fire/postburn) отбрасываются.

Сплит:
  test  - наши прежние 13 кадров GOR/TRS (Polina), не участвуют в обучении
  val   - 16 Egor-кадров из sar_source val (для early stopping)
  train - остальные 134 кадра (51 Polina + 83 Egor)

Изображения приводятся к max side 1280 с сохранением пропорций.
"""
from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
SAR = ROOT / "data" / "sar_source" / "datasets" / "sar"
OLD_TEST_IMG = ROOT / "dataset" / "images" / "val"
OLD_TEST_LBL = ROOT / "dataset" / "labels" / "val"
OUT = ROOT / "sar_3class"
DATASET = OUT / "dataset"

CLASSES = ["human", "transport", "stuff"]
MAX_SIDE = 1280
QUALITY = 92


def test_mapping() -> dict[str, str]:
    """newstem (polina_XXXX) -> oldstem (train_GOR_...)."""
    old_stems = {p.stem for p in OLD_TEST_LBL.glob("*.txt")}
    found = {}
    report = SAR / "merge_report.txt"
    for line in report.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 4 or not parts[0].startswith("polina_"):
            continue
        newstem, _split, _src, old = parts[:4]
        if Path(old).stem in old_stems:
            found[newstem] = Path(old).stem
    assert len(found) == len(old_stems) == 13, f"test mapping: {len(found)} != 13"
    return found


def filtered_label(path: Path) -> list[str]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        assert len(parts) == 5, f"{path.name}: {line}"
        cls = int(parts[0])
        assert 0 <= cls <= 5, f"{path.name}: class {cls}"
        if cls < len(CLASSES):
            rows.append(" ".join(parts))
    return rows


def save_resized(src: Path, dst: Path) -> None:
    with Image.open(src) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = MAX_SIDE / max(w, h)
        if scale < 1:
            im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst, "JPEG", quality=QUALITY, optimize=True)


def main() -> None:
    mapping = test_mapping()
    test_new = set(mapping)
    test_old = set(mapping.values())

    items = []
    for split in ("train", "val"):
        for img in sorted((SAR / "images" / split).glob("*")):
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            stem = img.stem
            lbl = SAR / "labels" / split / f"{stem}.txt"
            assert lbl.exists(), f"нет разметки: {stem}"
            items.append({"stem": stem, "orig_split": split, "img": img, "lbl": lbl})
    assert len(items) == 163, len(items)

    val_stems = {it["stem"] for it in items
                 if it["orig_split"] == "val" and it["stem"].startswith("egor_")}
    assert len(val_stems) == 16, len(val_stems)

    train_stems = {it["stem"] for it in items} - test_new - val_stems
    assert len(train_stems) == 134, len(train_stems)
    assert not (train_stems & test_new)
    assert not (train_stems & val_stems)
    assert not (val_stems & test_new)

    if OUT.exists():
        shutil.rmtree(OUT)
    for split in ("train", "val", "test"):
        (DATASET / "images" / split).mkdir(parents=True)
        (DATASET / "labels" / split).mkdir(parents=True)

    stats = defaultdict(Counter)
    zero = defaultdict(list)
    manifest = []

    for it in items:
        stem = it["stem"]
        if stem in test_new:
            continue
        split = "val" if stem in val_stems else "train"
        rows = filtered_label(it["lbl"])
        (DATASET / "labels" / split / f"{stem}.txt").write_text(
            "\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        save_resized(it["img"], DATASET / "images" / split / f"{stem}.jpg")
        for row in rows:
            stats[split][int(row.split()[0])] += 1
        if not rows:
            zero[split].append(stem)
        manifest.append({"stem": stem, "split": split,
                         "source": "egor" if stem.startswith("egor_") else "polina",
                         "orig_split": it["orig_split"]})

    # Тест: копируем прежние 13 кадров как есть, чтобы сравнение было честным.
    for old in sorted(test_old):
        shutil.copy2(OLD_TEST_IMG / f"{old}.jpg", DATASET / "images" / "test" / f"{old}.jpg")
        shutil.copy2(OLD_TEST_LBL / f"{old}.txt", DATASET / "labels" / "test" / f"{old}.txt")
        for line in (OLD_TEST_LBL / f"{old}.txt").read_text(encoding="utf-8").splitlines():
            if line.strip():
                stats["test"][int(line.split()[0])] += 1
        manifest.append({"stem": old, "split": "test", "source": "polina",
                         "orig_split": "val"})

    def yaml_text(split: str) -> str:
        return "\n".join([
            f"path: {DATASET.as_posix()}",
            "train: images/train",
            f"val: images/{split}",
            "names:",
            *[f"  {i}: {name}" for i, name in enumerate(CLASSES)],
            "",
        ])

    (OUT / "data.yaml").write_text(yaml_text("val"), encoding="utf-8")
    (OUT / "test.yaml").write_text(yaml_text("test"), encoding="utf-8")
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("sar_3class dataset ready:", DATASET)
    for split in ("train", "val", "test"):
        n = len(list((DATASET / "images" / split).glob("*")))
        total = sum(stats[split].values())
        per = ", ".join(f"{CLASSES[c]}={stats[split][c]}" for c in range(len(CLASSES)))
        print(f"  {split}: {n} images, {total} boxes ({per}), background={len(zero[split])}")
    print("test stems excluded from train/val: OK")


if __name__ == "__main__":
    main()
