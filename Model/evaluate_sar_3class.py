"""Оценка обученной sar_3class-модели.

- test (13 кадров GOR/TRS, Polina) при 1024 и 1536
- val  (16 кадров Egor) при 1024 и 1536
- предсказания на неразмеченных Polina-кадрах и на Egor val
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")

import torch
from ultralytics import YOLO

torch.set_num_threads(8)

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "runs" / "sar_3class_1024" / "weights" / "best.pt"
DATA = ROOT / "sar_3class" / "data.yaml"
TEST = ROOT / "sar_3class" / "test.yaml"


def summarize(metrics, names: dict[int, str]) -> dict:
    per = {}
    for pos, cls_id in enumerate(metrics.box.ap_class_index):
        per[names[int(cls_id)]] = {
            "precision": float(metrics.box.p[pos]),
            "recall": float(metrics.box.r[pos]),
            "mAP50": float(metrics.box.ap50[pos]),
            "mAP50-95": float(metrics.box.ap[pos]),
        }
    return {
        "overall": {
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "mAP50": float(metrics.box.map50),
            "mAP50-95": float(metrics.box.map),
        },
        "per_class": per,
    }


def run_val(model, data: Path, imgsz: int, name: str):
    metrics = model.val(
        data=str(data), split="val", imgsz=imgsz, batch=4, device="cpu",
        plots=True, project=str(ROOT / "runs"), name=name, exist_ok=True,
    )
    return summarize(metrics, model.names)


def main() -> None:
    assert WEIGHTS.exists(), f"нет весов: {WEIGHTS}"
    model = YOLO(str(WEIGHTS))

    result = {
        "weights": str(WEIGHTS),
        "test_polina": {},
        "val_egor": {},
    }
    for imgsz in (1024, 1536):
        result["test_polina"][str(imgsz)] = run_val(
            model, TEST, imgsz, f"sar_3class_test_{imgsz}")
        result["val_egor"][str(imgsz)] = run_val(
            model, DATA, imgsz, f"sar_3class_val_{imgsz}")

    (ROOT / "sar_3class_metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # Визуальная проверка: Polina без разметки
    labeled = {p.stem for p in (ROOT / "dataset" / "labels" / "train").glob("*.txt")}
    labeled |= {p.stem for p in (ROOT / "dataset" / "labels" / "val").glob("*.txt")}
    unlabeled = sorted(p for p in (ROOT / "data" / "human").glob("*.JPG")
                       if p.stem not in labeled)[:4]
    if unlabeled:
        model.predict(source=[str(p) for p in unlabeled], imgsz=1024, conf=0.25,
                      device="cpu", save=True, project=str(ROOT / "predictions"),
                      name="sar_3class_polina", exist_ok=True, verbose=False)

    egor = sorted((ROOT / "sar_3class" / "dataset" / "images" / "val").glob("egor_*.jpg"))[:4]
    if egor:
        model.predict(source=[str(p) for p in egor], imgsz=1024, conf=0.25,
                      device="cpu", save=True, project=str(ROOT / "predictions"),
                      name="sar_3class_egor", exist_ok=True, verbose=False)

    print("predictions:", ROOT / "predictions")


if __name__ == "__main__":
    main()
