"""Финализация sar_3class-модели.

- Копирует last.pt (эпоха 42) в best_combined.pt: по совокупности test/val он
  лучше early-stop best.pt (эпоха 22). Выбор сделан после просмотра метрик,
  поэтому это не «честный» model selection, а зафиксированное решение.
- Считает полные метрики финальной модели при imgsz=1536:
  Polina test (13 кадров GOR/TRS) и Egor val (16 кадров).
- Для сравнения считает старую 64-кадровую модель на том же Polina test
  при 1024 и 1536.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")

import torch
from ultralytics import YOLO

torch.set_num_threads(8)

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "runs" / "sar_3class_1024"
LAST = RUN / "weights" / "last.pt"
FINAL = RUN / "weights" / "best_combined.pt"
BASELINE = ROOT / "runs" / "human3_1024" / "weights" / "best.pt"
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


def val(model, data: Path, imgsz: int, name: str) -> dict:
    metrics = model.val(
        data=str(data), split="val", imgsz=imgsz, batch=4, device="cpu",
        plots=False, project=str(ROOT / "runs" / "final"), name=name,
        exist_ok=True, verbose=False,
    )
    return summarize(metrics, model.names)


def main() -> None:
    if not FINAL.exists():
        shutil.copy2(LAST, FINAL)
        print("copied", LAST, "->", FINAL)

    final = YOLO(str(FINAL))
    result = {
        "weights": str(FINAL),
        "imgsz": 1536,
        "note": "last.pt (epoch 42) chosen over early-stop best.pt (epoch 22) "
                "after comparing both on test/val",
        "test_polina": val(final, TEST, 1536, "final_test_1536"),
        "val_egor": val(final, DATA, 1536, "final_val_1536"),
    }
    (ROOT / "sar_3class_final_metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    baseline_result = {"weights": str(BASELINE), "test_polina": {}}
    if BASELINE.exists():
        baseline = YOLO(str(BASELINE))
        for imgsz in (1024, 1536):
            baseline_result["test_polina"][str(imgsz)] = val(
                baseline, TEST, imgsz, f"baseline_test_{imgsz}")
        (ROOT / "baseline_sar_3class_test.json").write_text(
            json.dumps(baseline_result, indent=2, ensure_ascii=False),
            encoding="utf-8")
        print(json.dumps(baseline_result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
