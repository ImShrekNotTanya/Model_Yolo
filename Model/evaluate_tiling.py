"""Сравнение обычного инференса и тайлинга на val-сплите.

Режимы:
  full      - кадр 1280x960 целиком, imgsz=1024 (как при обучении)
  tile640   - 9 тайлов 640x480 из кадра 1280x960 (объекты в 1.6x крупнее)
  tile1024  - 20 тайлов 1024x1024 из оригинала 4000x3000 (объекты в 4x крупнее)

Метрики считаются по ground truth в координатах 1280x960, IoU=0.5,
AP50 - all-point interpolation. Детекции объединяются глобальным NMS.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset"
SRC = ROOT / "data" / "human"
WEIGHTS = ROOT / "runs" / "human3_1024" / "weights" / "best.pt"

CLASSES = ["human", "transport", "stuff"]
IMGSZ = 1024
CONF = 0.001
NMS_IOU = 0.7
MATCH_IOU = 0.5
REPORT_CONF = 0.25
W, H = 1280, 960
ORIG_W, ORIG_H = 4000, 3000


def load_gt(stem: str) -> dict[int, list[np.ndarray]]:
    out = {c: [] for c in range(len(CLASSES))}
    path = DATASET / "labels" / "val" / f"{stem}.txt"
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        cls, cx, cy, bw, bh = line.split()
        cls = int(cls)
        cx, cy, bw, bh = map(float, (cx, cy, bw, bh))
        out[cls].append(np.array([
            (cx - bw / 2) * W, (cy - bh / 2) * H,
            (cx + bw / 2) * W, (cy + bh / 2) * H,
        ], dtype=np.float32))
    return out


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def grid_starts(length: int, tile: int, overlap: int) -> list[int]:
    if tile >= length:
        return [0]
    stride = max(1, tile - overlap)
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def predict_tiles(model: YOLO, image: Image.Image, tile_w: int, tile_h: int,
                  overlap: int, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Возвращает детекции в координатах 1280x960 (scale = W / ширина_источника)."""
    boxes_all, scores_all, classes_all = [], [], []
    w, h = image.size
    for y0 in grid_starts(h, tile_h, overlap):
        for x0 in grid_starts(w, tile_w, overlap):
            crop = image.crop((x0, y0, min(x0 + tile_w, w), min(y0 + tile_h, h)))
            res = model.predict(
                source=crop, imgsz=IMGSZ, conf=CONF, iou=NMS_IOU,
                max_det=300, device="cpu", verbose=False,
            )[0]
            if res.boxes is None or len(res.boxes) == 0:
                continue
            b = res.boxes.xyxy.cpu().numpy().copy()
            b[:, [0, 2]] = (b[:, [0, 2]] + x0) * scale
            b[:, [1, 3]] = (b[:, [1, 3]] + y0) * scale
            boxes_all.append(b)
            scores_all.append(res.boxes.conf.cpu().numpy())
            classes_all.append(res.boxes.cls.cpu().numpy().astype(int))

    if not boxes_all:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int)
    boxes = np.concatenate(boxes_all)
    scores = np.concatenate(scores_all)
    classes = np.concatenate(classes_all)

    keep_all = []
    for c in range(len(CLASSES)):
        idx = np.where(classes == c)[0]
        if len(idx) == 0:
            continue
        keep = nms(
            torch.as_tensor(boxes[idx], dtype=torch.float32),
            torch.as_tensor(scores[idx], dtype=torch.float32),
            NMS_IOU,
        ).cpu().numpy()
        keep_all.append(idx[keep])
    keep = np.concatenate(keep_all) if keep_all else np.zeros(0, int)
    return boxes[keep], scores[keep], classes[keep]


def predict_full(model: YOLO, image: Image.Image) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    res = model.predict(
        source=image, imgsz=IMGSZ, conf=CONF, iou=NMS_IOU,
        max_det=300, device="cpu", verbose=False,
    )[0]
    if res.boxes is None or len(res.boxes) == 0:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int)
    return (
        res.boxes.xyxy.cpu().numpy(),
        res.boxes.conf.cpu().numpy(),
        res.boxes.cls.cpu().numpy().astype(int),
    )


def class_metrics(dets: list[tuple[np.ndarray, np.ndarray]],
                  gts: list[list[np.ndarray]]) -> dict | None:
    n_gt = sum(len(g) for g in gts)
    if n_gt == 0:
        return None

    entries = []
    for i, (b, s) in enumerate(dets):
        for k in range(len(b)):
            entries.append((float(s[k]), i, b[k]))
    entries.sort(key=lambda t: -t[0])

    matched = [set() for _ in gts]
    tps, fps = [], []
    for _, i, box in entries:
        best, best_j = 0.0, -1
        for j, gt in enumerate(gts[i]):
            if j in matched[i]:
                continue
            v = iou_xyxy(box, gt)
            if v > best:
                best, best_j = v, j
        if best >= MATCH_IOU and best_j >= 0:
            matched[i].add(best_j)
            tps.append(1)
            fps.append(0)
        else:
            tps.append(0)
            fps.append(1)

    if not tps:
        return {"AP50": 0.0, "P@0.25": 0.0, "R@0.25": 0.0, "n_gt": n_gt,
                "n_det": 0}

    tp = np.cumsum(tps)
    fp = np.cumsum(fps)
    rec = tp / n_gt
    prec = tp / np.maximum(tp + fp, 1)

    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([1.0], prec, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

    k = int(np.searchsorted(-np.array([e[0] for e in entries]), -REPORT_CONF, side="right"))
    tp_k = int(tp[k - 1]) if k > 0 else 0
    fp_k = int(fp[k - 1]) if k > 0 else 0
    return {
        "AP50": ap,
        "P@0.25": tp_k / max(tp_k + fp_k, 1),
        "R@0.25": tp_k / n_gt,
        "n_gt": n_gt,
        "n_det": len(entries),
    }


def main() -> None:
    assert WEIGHTS.exists(), WEIGHTS
    model = YOLO(str(WEIGHTS))

    stems = sorted(p.stem for p in (DATASET / "labels" / "val").glob("*.txt"))
    gts_per_image = {stem: load_gt(stem) for stem in stems}
    gts = {
        c: [gts_per_image[stem][c] for stem in stems]
        for c in range(len(CLASSES))
    }

    modes = {
        "full": lambda im, orig: predict_full(model, im),
        "tile640": lambda im, orig: predict_tiles(model, im, 640, 480, 128),
        "tile1024": lambda im, orig: predict_tiles(model, orig, 1024, 1024, 128, W / ORIG_W),
    }

    all_results = {}
    for mode, fn in modes.items():
        started = time.perf_counter()
        dets = {c: [] for c in range(len(CLASSES))}
        for stem in stems:
            im = Image.open(DATASET / "images" / "val" / f"{stem}.jpg").convert("RGB")
            orig = None
            if mode == "tile1024":
                orig = Image.open(SRC / f"{stem}.JPG").convert("RGB")
            boxes, scores, classes = fn(im, orig)
            for c in range(len(CLASSES)):
                m = classes == c
                dets[c].append((boxes[m], scores[m]))

        results = {}
        for c, name in enumerate(CLASSES):
            m = class_metrics(dets[c], gts[c])
            if m is not None:
                results[name] = m
        present = [v["AP50"] for v in results.values()]
        all_results[mode] = {
            "per_class": results,
            "mAP50": float(np.mean(present)) if present else None,
            "seconds": round(time.perf_counter() - started, 1),
        }
        print(f"\n[{mode}] {all_results[mode]['seconds']}s, mAP50={all_results[mode]['mAP50']}")
        for name, m in results.items():
            print(f"  {name:<10} AP50={m['AP50']:.3f}  P@.25={m['P@0.25']:.3f}  "
                  f"R@.25={m['R@0.25']:.3f}  gt={m['n_gt']}  det={m['n_det']}")
        if "transport" not in results:
            print("  transport  нет объектов в val")

    (ROOT / "tiling_metrics.json").write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {ROOT / 'tiling_metrics.json'}")


if __name__ == "__main__":
    main()

