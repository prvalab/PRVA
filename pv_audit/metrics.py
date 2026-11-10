"""Dataset-level robustness metrics for PRVA.

For every stress level, predictions are matched to a reference (human ground truth, or the model's own detections on
the clean image) and summarised over a set of images:

- recall          share of reference objects still found (same label, IoU >= iou_thr)
- mean_iou        mean IoU of the matched objects
- mean_conf       mean confidence of the matched objects
- precision       share of predictions (score >= score_thr) that match a reference object
- overconfidence  mean confidence of all kept predictions minus their precision (> 0: more confident than correct)

Confidence intervals come from a bootstrap over images (fixed seed: the same input gives the same numbers).
The red line is the first level whose recall CI upper bound falls below `frac` x the clean-image recall.
Results describe the tested model under the tested conditions only; this is not a standard or a certification.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def normalize(pred) -> List[dict]:
    """adapter output -> [{'box': [x1, y1, x2, y2], 'label': ..., 'score': float}]. Accepts the detection form
    {'boxes', 'labels', 'scores'}, a list of {'box', 'label', 'conf'|'score'}, or the legacy {'conf': float}
    (one image-level score, no boxes)."""
    if pred is None:
        return []
    if isinstance(pred, dict) and "boxes" in pred:
        labels = pred.get("labels") or [None] * len(pred["boxes"])
        return [{"box": [float(v) for v in b], "label": l, "score": float(s)}
                for b, l, s in zip(pred["boxes"], labels, pred["scores"])]
    if isinstance(pred, list):
        return [{"box": [float(v) for v in p["box"]], "label": p.get("label"),
                 "score": float(p.get("score", p.get("conf", 0.0)))} for p in pred]
    return []


def match_image(refs: List[dict], preds: List[dict], iou_thr: float = 0.5, score_thr: float = 0.35) -> dict:
    """greedy one-to-one matching by IoU (highest first); returns per-image counts and sums"""
    kept = [p for p in preds if p["score"] >= score_thr]
    pairs = sorted(((iou(r["box"], p["box"]), i, j) for i, r in enumerate(refs) for j, p in enumerate(kept)
                    if r.get("label") == p.get("label") or r.get("label") is None or p.get("label") is None),
                   reverse=True)
    used_r, used_p, ious, confs = set(), set(), [], []
    for v, i, j in pairs:
        if v < iou_thr or i in used_r or j in used_p:
            continue
        used_r.add(i), used_p.add(j)
        ious.append(v), confs.append(kept[j]["score"])
    return {"n_ref": len(refs), "n_pred": len(kept), "tp": len(ious), "iou_sum": float(sum(ious)),
            "conf_sum": float(sum(confs)), "score_sum_all": float(sum(p["score"] for p in kept))}


def summarise(rows: List[dict]) -> dict:
    """per-image match counts -> level metrics"""
    n_ref = sum(r["n_ref"] for r in rows)
    n_pred = sum(r["n_pred"] for r in rows)
    tp = sum(r["tp"] for r in rows)
    prec = tp / n_pred if n_pred else float("nan")
    return {"recall": tp / n_ref if n_ref else float("nan"),
            "precision": prec,
            "mean_iou": sum(r["iou_sum"] for r in rows) / tp if tp else 0.0,
            "mean_conf": sum(r["conf_sum"] for r in rows) / tp if tp else 0.0,
            "overconfidence": (sum(r["score_sum_all"] for r in rows) / n_pred - prec) if n_pred else 0.0,
            "n_ref": n_ref, "n_pred": n_pred, "tp": tp}


def bootstrap_ci(rows: List[dict], key: str, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0):
    """(low, high) of a level metric, resampling images with replacement"""
    if not rows:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = [rows[i] for i in rng.integers(0, len(rows), len(rows))]
        v = summarise(pick)[key]
        if v == v:                                          # skip NaN (no references in the sample)
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2)))


def red_line(levels: Sequence[float], recall: Sequence[float], recall_hi: Sequence[float],
             frac: float = 0.8) -> Optional[float]:
    """first level whose recall CI upper bound is below frac x the recall at the first (clean) level"""
    if not levels:
        return None
    base = recall[0]
    for lv, hi in zip(levels, recall_hi):
        if hi == hi and hi < frac * base:
            return lv
    return None


def evaluate(per_level: Dict[float, List[dict]], n_boot: int = 1000, seed: int = 0, frac: float = 0.8) -> dict:
    """{level: [per-image match rows]} -> {'levels': [...], 'metrics': {...}, 'ci': {...}, 'red_line': level|None}"""
    levels = sorted(per_level)
    out = {"levels": levels, "metrics": {}, "ci": {}}
    for k in ("recall", "precision", "mean_iou", "mean_conf", "overconfidence"):
        out["metrics"][k] = [summarise(per_level[lv])[k] for lv in levels]
    for k in ("recall", "mean_iou", "mean_conf"):
        out["ci"][k] = [bootstrap_ci(per_level[lv], k, n_boot, seed=seed) for lv in levels]
    out["red_line"] = red_line(levels, out["metrics"]["recall"], [c[1] for c in out["ci"]["recall"]], frac)
    out["red_line_rule"] = f"first level with recall CI upper bound < {frac} x clean recall"
    return out
