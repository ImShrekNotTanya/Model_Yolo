"""Assemble a Firewatch-style chart pack for the 5-class SAR detector.

Firewatch stores the standard Ultralytics training/evaluation figures in
`models/best_model`. This script copies the same kinds of figures from the
detector runs into `charts/firewatch_style`, adds a corrected 5-class labels
chart, and builds a one-image overview.

Run:
    python charts/make_firewatch_charts.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "firewatch_style"
DATASET = ROOT / "sar_5class" / "dataset"
TRAIN_RUN = ROOT / "runs" / "sar_5class_1024"
VAL_RUN = ROOT / "runs" / "sar_5class_final" / "val"
TEST_RUN = ROOT / "runs" / "sar_5class_final" / "test"

CLASSES = ["human", "transport", "marker", "smoke", "fire"]
COLORS = ["#1565c0", "#00acc1", "#90a4ae", "#26a69a", "#283593"]

TRAIN_FILES = [
    "args.yaml",
    "results.csv",
    "results.png",
    "labels.jpg",
    "train_batch0.jpg",
    "train_batch1.jpg",
    "train_batch2.jpg",
    "train_batch850.jpg",
    "train_batch851.jpg",
    "train_batch852.jpg",
]

EVAL_FILES = [
    "BoxF1_curve.png",
    "BoxP_curve.png",
    "BoxPR_curve.png",
    "BoxR_curve.png",
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "val_batch0_labels.jpg",
    "val_batch0_pred.jpg",
    "val_batch1_labels.jpg",
    "val_batch1_pred.jpg",
    "val_batch2_labels.jpg",
    "val_batch2_pred.jpg",
]


def copy_files(src: Path, dst: Path, names: list[str]) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in names:
        source = src / name
        if source.exists():
            shutil.copy2(source, dst / name)


def load_train_labels() -> np.ndarray:
    rows: list[list[float]] = []
    for path in sorted((DATASET / "labels" / "train").glob("*.txt")):
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 5:
                rows.append([float(value) for value in parts[:5]])
    if not rows:
        raise RuntimeError("No train labels found")
    return np.asarray(rows, dtype=float)


def make_labels_chart() -> Path:
    data = load_train_labels()
    cls = data[:, 0].astype(int)
    x, y, w, h = data[:, 1], data[:, 2], data[:, 3], data[:, 4]
    counts = np.bincount(cls, minlength=len(CLASSES))

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    ax = axes[0, 0]
    bars = ax.bar(CLASSES, counts, color=COLORS, edgecolor="white")
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            count,
            str(int(count)),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_title("Объекты по классам (train)")
    ax.set_ylabel("instances")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)

    ax = axes[0, 1]
    ax.hist2d(x, y, bins=40, range=[[0, 1], [0, 1]], cmap="Blues")
    ax.set_title("Центры боксов")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.invert_yaxis()

    ax = axes[1, 0]
    w_hi = max(0.25, float(np.quantile(w, 0.99)) * 1.1)
    h_hi = max(0.25, float(np.quantile(h, 0.99)) * 1.1)
    ax.hist2d(w, h, bins=40, range=[[0, w_hi], [0, h_hi]], cmap="Blues")
    ax.set_title("Размеры боксов")
    ax.set_xlabel("width")
    ax.set_ylabel("height")

    ax = axes[1, 1]
    for index, (name, color) in enumerate(zip(CLASSES, COLORS)):
        mask = cls == index
        ax.scatter(w[mask], h[mask], s=5, alpha=0.45, label=name, color=color, rasterized=True)
    ax.set_title("Классы в пространстве размеров")
    ax.set_xlabel("width")
    ax.set_ylabel("height")
    ax.legend(markerscale=3, fontsize=8, frameon=False)

    fig.suptitle(
        "Разметка train — 5 классов SAR",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = OUT / "training" / "labels_marker.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def make_overview() -> Path:
    items = [
        (OUT / "training" / "results.png", "Training: losses и mAP"),
        (OUT / "training" / "labels_marker.png", "Разметка train"),
        (OUT / "val_egor" / "confusion_matrix.png", "Валидация: матрица ошибок"),
        (OUT / "val_egor" / "BoxPR_curve.png", "Валидация: Precision-Recall"),
        (OUT / "test_polina" / "confusion_matrix.png", "Тест: матрица ошибок"),
        (OUT / "test_polina" / "BoxPR_curve.png", "Тест: Precision-Recall"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    for ax, (path, title) in zip(axes.ravel(), items):
        ax.imshow(mpimg.imread(path))
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

    fig.suptitle(
        "SAR 5-class — графики как в Firewatch",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    path = OUT / "overview.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def write_readme() -> Path:
    text = """# Графики как в Firewatch — SAR 5-class

Это набор стандартных Ultralytics-графиков, аналогичных тем, что лежат в
`Firewatch-main/models/best_model`.

## Откуда взято

| Папка | Источник | Что внутри |
|---|---|---|
| `training` | `runs/sar_5class_1024` | кривые обучения, разметка, train batches |
| `val_egor` | `runs/sar_5class_final/val` | confusion matrix, PR/F1/P/R, val batches |
| `test_polina` | `runs/sar_5class_final/test` | confusion matrix, PR/F1/P/R, val batches |

Модель: **YOLO11s**, 60 эпох, обучение при `imgsz=1024`.
Финальная проверка: `imgsz=1536`.

Стандартные `confusion_matrix.png` и PR-кривые Ultralytics считает при низком
пороге `conf=0.001` (как для mAP), поэтому в матрице много
низкоуверенных ложных срабатываний. Для рабочего порога `conf=0.25`
смотри графики в `detector/error_analysis/`.

## Что смотреть

- `overview.png` — общий обзор шести ключевых графиков;
- `training/results.png` — кривые обучения;
- `training/labels_marker.png` — исправленная разметка с классом `marker`
  (в стандартном `labels.jpg` Ultralytics класс 2 подписан как `stuff`);
- `val_egor/confusion_matrix.png` — ошибки на валидации, все 5 классов;
- `val_egor/BoxPR_curve.png` — Precision-Recall, все 5 классов;
- `test_polina/confusion_matrix.png` — ошибки на тесте;
- `test_polina/BoxPR_curve.png` — Precision-Recall на тесте.

`test_polina` содержит разметку только для `human` и `marker`; для
`transport`, `smoke` и `fire` тестовых метрик нет.

## Пересобрать

```powershell
cd C:\\Users\\Никита\\Desktop\\detector
python charts\\make_firewatch_charts.py
```

Подробный разбор ошибок лежит в `detector/error_analysis/`.
"""
    path = OUT / "README.md"
    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    copy_files(TRAIN_RUN, OUT / "training", TRAIN_FILES)
    copy_files(VAL_RUN, OUT / "val_egor", EVAL_FILES)
    copy_files(TEST_RUN, OUT / "test_polina", EVAL_FILES)

    labels = make_labels_chart()
    overview = make_overview()
    readme = write_readme()

    print(f"labels:   {labels}")
    print(f"overview: {overview}")
    print(f"readme:   {readme}")
    print(f"charts:   {OUT}")


if __name__ == "__main__":
    main()
