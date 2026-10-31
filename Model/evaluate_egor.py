"""Проверка текущей 3-классовой модели на Egor-кадрах (крошечные люди).

Сравниваем полный кадр и тайлинг в нативном разрешении:
  full     - весь кадр 8000x6000 -> 1024
  t2048    - тайлы 2048, люди ~23 px
  t3072    - тайлы 3072, люди ~16 px (масштаб обучения)
  t4096    - тайлы 4096, люди ~12 px
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
SAR = ROOT / "data" / "sar_source" / "datasets" / "sar"
WEIGHTS = ROOT / "runs" / "human3_1024" / "weights" / "best.pt"
CLASSES = ["human", "transport", "stuff"]
IMGSZ = 1024
CONF = 0.001
NMS_IOU = 0.7
MATCH_IOU = 0.5
REPORT_CONF = 0.25


def iou_xyxy(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    u = aa + bb - inter
    return float(inter / u) if u > 0 else 0.0


def grid_starts(length, tile, overlap):
    if tile >= length:
        return [0]
    stride = max(1, tile - overlap)
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def load_gt(stem, w, h):
    out = {c: [] for c in range(len(CLASSES))}
    p = SAR / "labels" / "val" / f"{stem}.txt"
    for line in p.read_text(encoding="utf-8").splitlines():
        q = line.split()
        if len(q) != 5:
            continue
        c = int(q[0])
        if c >= len(CLASSES):
            continue
        _, cx, cy, bw, bh = q
        cx, cy, bw, bh = map(float, (cx, cy, bw, bh))
        out[c].append(np.array([
            (cx - bw / 2) * w, (cy - bh / 2) * h,
            (cx + bw / 2) * w, (cy + bh / 2) * h,
        ], np.float32))
    return out


def nms_global(boxes, scores, classes):
    if len(boxes) == 0:
        return boxes, scores, classes
    keep_all = []
    for c in range(len(CLASSES)):
        idx = np.where(classes == c)[0]
        if len(idx) == 0:
            continue
        keep = nms(torch.as_tensor(boxes[idx], dtype=torch.float32),
                   torch.as_tensor(scores[idx], dtype=torch.float32),
                   NMS_IOU).cpu().numpy()
        keep_all.append(idx[keep])
    keep = np.concatenate(keep_all) if keep_all else np.zeros(0, int)
    return boxes[keep], scores[keep], classes[keep]


def predict_full(model, im):
    r = model.predict(source=im, imgsz=IMGSZ, conf=CONF, iou=NMS_IOU,
                      max_det=300, device="cpu", verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int)
    return (r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
            r.boxes.cls.cpu().numpy().astype(int))


def predict_tiles(model, im, tile, overlap):
    w, h = im.size
    boxes, scores, classes = [], [], []
    for y0 in grid_starts(h, tile, overlap):
        for x0 in grid_starts(w, tile, overlap):
            crop = im.crop((x0, y0, min(x0 + tile, w), min(y0 + tile, h)))
            r = model.predict(source=crop, imgsz=IMGSZ, conf=CONF, iou=NMS_IOU,
                              max_det=300, device="cpu", verbose=False)[0]
            if r.boxes is None or len(r.boxes) == 0:
                continue
            b = r.boxes.xyxy.cpu().numpy().copy()
            b[:, [0, 2]] += x0
            b[:, [1, 3]] += y0
            boxes.append(b)
            scores.append(r.boxes.conf.cpu().numpy())
            classes.append(r.boxes.cls.cpu().numpy().astype(int))
    if not boxes:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int)
    return nms_global(np.concatenate(boxes), np.concatenate(scores), np.concatenate(classes))


def class_metrics(dets, gts):
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
        best, bj = 0.0, -1
        for j, gt in enumerate(gts[i]):
            if j in matched[i]:
                continue
            v = iou_xyxy(box, gt)
            if v > best:
                best, bj = v, j
        if best >= MATCH_IOU and bj >= 0:
            matched[i].add(bj)
            tps.append(1)
            fps.append(0)
        else:
            tps.append(0)
            fps.append(1)
    if not tps:
        return {"AP50": 0.0, "P@.25": 0.0, "R@.25": 0.0, "n_gt": n_gt, "n_det": 0}

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

    scores = np.array([e[0] for e in entries])
    k = int(np.searchsorted(-scores, -REPORT_CONF, side="right"))
    tp_k = int(tp[k - 1]) if k > 0 else 0
    fp_k = int(fp[k - 1]) if k > 0 else 0
    return {"AP50": ap, "P@.25": tp_k / max(tp_k + fp_k, 1),
            "R@.25": tp_k / n_gt, "n_gt": n_gt, "n_det": len(entries)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(WEIGHTS))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    model = YOLO(args.weights)
    stems = sorted(p.stem for p in (SAR / "images" / "val").glob("egor_*.jpg"))
    gt_all = {}
    for stem in stems:
        with Image.open(SAR / "images" / "val" / f"{stem}.jpg") as im:
            w, h = im.size
        gt_all[stem] = load_gt(stem, w, h)

    modes = {
        "full": lambda im: predict_full(model, im),
        "t2048": lambda im: predict_tiles(model, im, 2048, 256),
        "t3072": lambda im: predict_tiles(model, im, 3072, 384),
        "t4096": lambda im: predict_tiles(model, im, 4096, 512),
    }

    out = {}
    for mode, fn in modes.items():
        dets = {c: [] for c in range(len(CLASSES))}
        for stem in stems:
            im = Image.open(SAR / "images" / "val" / f"{stem}.jpg").convert("RGB")
            b, s, c = fn(im)
            for cls in range(len(CLASSES)):
                m = c == cls
                dets[cls].append((b[m], s[m]))
        out[mode] = {}
        for cls, name in enumerate(CLASSES):
            gts = [gt_all[stem][cls] for stem in stems]
            m = class_metrics(dets[cls], gts)
            if m:
                out[mode][name] = m
        print(f"\n[{mode}]")
        for name, m in out[mode].items():
            print(f"  {name:<10} AP50={m['AP50']:.3f}  P@.25={m['P@.25']:.3f}  "
                  f"R@.25={m['R@.25']:.3f}  gt={m['n_gt']}  det={m['n_det']}")

    name = f"egor_metrics{args.tag}.json"
    (ROOT / name).write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
