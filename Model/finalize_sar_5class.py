"""Финализация единой 5-классовой SAR-модели.

- берёт выбранный чекпоинт (по умолчанию last.pt)
- переименовывает класс 2 stuff -> marker
- сохраняет как runs/sar_5class_1024/weights/best_sar.pt
- считает метрики на Polina test и Egor val при 1536
- делает предсказания для визуальной проверки
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")

import torch
from ultralytics import YOLO

torch.set_num_threads(8)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "sar_5class" / "data.yaml"
TEST = ROOT / "sar_5class" / "test.yaml"


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "runs" / "sar_5class_1024" / "weights" / "last.pt"))
    ap.add_argument("--imgsz", type=int, default=1536)
    ap.add_argument("--name", default="best_sar.pt")
    args = ap.parse_args()

    src = Path(args.weights)
    assert src.exists(), f"нет весов: {src}"
    model = YOLO(str(src))
    names = dict(model.names)
    names[2] = "marker"
    model.model.names = names
    if getattr(model, "ckpt", None):
        model.ckpt["model"] = model.model

    out = src.parent / args.name
    model.save(str(out))
    print("saved", out, "names", model.names)

    result = {
        "weights": str(out),
        "imgsz": args.imgsz,
        "classes": list(model.names.values()),
        "test_polina": summarize(
            model.val(data=str(TEST), split="val", imgsz=args.imgsz, batch=4,
                      device="cpu", plots=True, project=str(ROOT / "runs" / "sar_5class_final"),
                      name="test", exist_ok=True, verbose=False),
            model.names),
        "val_egor": summarize(
            model.val(data=str(DATA), split="val", imgsz=args.imgsz, batch=4,
                      device="cpu", plots=True, project=str(ROOT / "runs" / "sar_5class_final"),
                      name="val", exist_ok=True, verbose=False),
            model.names),
    }
    (ROOT / "sar_5class_final_metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    labeled = {p.stem for p in (ROOT / "dataset" / "labels" / "train").glob("*.txt")}
    labeled |= {p.stem for p in (ROOT / "dataset" / "labels" / "val").glob("*.txt")}
    unlabeled = sorted(p for p in (ROOT / "data" / "human").glob("*.JPG")
                       if p.stem not in labeled)[:4]
    if unlabeled:
        model.predict(source=[str(p) for p in unlabeled], imgsz=args.imgsz, conf=0.25,
                      device="cpu", save=True, project=str(ROOT / "predictions"),
                      name="sar_5class_final_polina", exist_ok=True, verbose=False)

    egor = sorted((ROOT / "sar_5class" / "dataset" / "images" / "val").glob("egor_*.jpg"))[:4]
    if egor:
        model.predict(source=[str(p) for p in egor], imgsz=args.imgsz, conf=0.25,
                      device="cpu", save=True, project=str(ROOT / "predictions"),
                      name="sar_5class_final_egor", exist_ok=True, verbose=False)
    print("predictions:", ROOT / "predictions")


if __name__ == "__main__":
    main()
