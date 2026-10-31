"""Обучение единой 5-классовой SAR-модели.

Классы: human, transport, marker (бывший stuff), smoke, fire.
Сплит: sar_5class (134 train / 16 Egor val / 13 Polina GOR-TRS test).

Стартуем с 3-классовой sar_3class-модели: backbone уже видел эти кадры,
голова пересоздаётся под 5 классов.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import torch
from ultralytics import YOLO

torch.set_num_threads(8)

ROOT = Path(__file__).resolve().parent
START = ROOT / "runs" / "sar_3class_1024" / "weights" / "best_combined.pt"


def main() -> None:
    model = YOLO(str(START if START.exists() else ROOT / "yolo11s.pt"))
    model.train(
        data=str(ROOT / "sar_5class" / "data.yaml"),
        epochs=60,
        imgsz=1024,
        batch=8,
        device="cpu",
        workers=2,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=3,
        weight_decay=0.0005,
        mosaic=0.5,
        close_mosaic=10,
        scale=0.5,
        translate=0.1,
        fliplr=0.5,
        patience=20,
        seed=0,
        deterministic=True,
        cache="ram",
        val=True,
        plots=True,
        save=True,
        project=str(ROOT / "runs"),
        name="sar_5class_1024",
        exist_ok=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
