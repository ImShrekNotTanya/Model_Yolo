"""Оценка единой 5-классовой SAR-модели.

- test: 13 кадров Polina GOR/TRS (чистый holdout)
- val:  16 кадров Egor (тот же домен, что train)
- перебирает best.pt/last.pt при 1024/1536/2048
- сохраняет sar_5class_sweep.json
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
RUN = ROOT / "runs" / "sar_5class_1024"
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


def val(model, data: Path, imgsz: int, name: str) -> dict:
    metrics = model.val(
        data=str(data), split="val", imgsz=imgsz, batch=4, device="cpu",
        plots=False, project=str(ROOT / "runs" / "sar_5class_sweep"), name=name,
        exist_ok=True, verbose=False,
    )
    return summarize(metrics, model.names)


def main() -> None:
    result = {}
    for tag in ("best", "last"):
        weights = RUN / "weights" / f"{tag}.pt"
        assert weights.exists(), f"нет весов: {weights}"
        model = YOLO(str(weights))
        result[tag] = {}
        for imgsz in (1024, 1536, 2048):
            result[tag][str(imgsz)] = {
                "polina": val(model, TEST, imgsz, f"{tag}_polina_{imgsz}"),
                "egor": val(model, DATA, imgsz, f"{tag}_egor_{imgsz}"),
            }
            print(tag, imgsz,
                  "polina", round(result[tag][str(imgsz)]["polina"]["overall"]["mAP50"], 4),
                  "egor", round(result[tag][str(imgsz)]["egor"]["overall"]["mAP50"], 4))
    (ROOT / "sar_5class_sweep.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("saved", ROOT / "sar_5class_sweep.json")


if __name__ == "__main__":
    main()
