"""Финальная оценка и визуальная проверка обученного детектора.

Запуск: python evaluate.py
Требует: runs/human3_1024/weights/best.pt
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")

import torch
from ultralytics import YOLO

torch.set_num_threads(8)

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "runs" / "human3_1024" / "weights" / "best.pt"
CLASSES = ["human", "transport", "stuff"]


def main() -> None:
    assert WEIGHTS.exists(), f"нет весов: {WEIGHTS}"
    model = YOLO(str(WEIGHTS))

    metrics = model.val(
        data=str(ROOT / "data.yaml"),
        split="val",
        imgsz=1024,
        batch=4,
        device="cpu",
        plots=True,
        project=str(ROOT / "runs"),
        name="val_final",
        exist_ok=True,
    )

    # p/r/ap50/ap — массивы только по присутствующим классам, в порядке ap_class_index
    per_class = {
        CLASSES[int(cls_id)]: {
            "precision": float(metrics.box.p[pos]),
            "recall": float(metrics.box.r[pos]),
            "mAP50": float(metrics.box.ap50[pos]),
            "mAP50-95": float(metrics.box.ap[pos]),
        }
        for pos, cls_id in enumerate(metrics.box.ap_class_index)
    }
    summary = {
        "weights": str(WEIGHTS),
        "overall": {
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "mAP50": float(metrics.box.map50),
            "mAP50-95": float(metrics.box.map),
        },
        "per_class": per_class,
    }
    (ROOT / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    # Визуальная проверка на неразмеченных кадрах
    labeled = {p.stem for p in (ROOT / "dataset" / "labels" / "train").glob("*.txt")}
    labeled |= {p.stem for p in (ROOT / "dataset" / "labels" / "val").glob("*.txt")}
    unlabeled = sorted(p for p in (ROOT / "data" / "human").glob("*.JPG") if p.stem not in labeled)
    random.Random(0).shuffle(unlabeled)

    out_dir = ROOT / "predictions"
    out_dir.mkdir(exist_ok=True)
    for img in unlabeled[:8]:
        model.predict(
            source=str(img),
            imgsz=1024,
            conf=0.25,
            device="cpu",
            save=True,
            project=str(out_dir),
            name="unlabeled",
            exist_ok=True,
            verbose=False,
        )
    print(f"Предсказания сохранены в {out_dir / 'unlabeled'}")


if __name__ == "__main__":
    main()
