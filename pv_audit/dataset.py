"""Run PRVA over a set of images and report dataset-level metrics with confidence intervals and a red line.

    from pv_audit import DatasetAudit
    report = DatasetAudit(adapter, model_name="my-detector").run(images, dims=["DPI", "Lux-"], lvls=range(0, 100, 10))
    report.to_json("prva_report.json"); report.plot("prva_report.png")

`images`: list of numpy images (BGR) or (image, ground_truth) pairs; ground truth = [{'box', 'label'}]. Without
ground truth the model's own detections on the clean image are the reference (relative robustness).
"""
from __future__ import annotations

import hashlib
import json
import platform
import time
from typing import Dict, Iterable, List, Optional

import numpy as np

from . import metrics
from .stress import StressSimulator

OPERATORS = {
    "Lux+": StressSimulator.apply_lux_plus,
    "Lux-": StressSimulator.apply_lux_minus,
    "MBlur": StressSimulator.apply_mblur,
    "ISO": StressSimulator.apply_iso_noise,
    "DPI": StressSimulator.apply_dpi,
    "Defocus": StressSimulator.apply_defocus,
}


def _hash(img: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(img).tobytes()).hexdigest()[:16]


class DatasetReport:
    def __init__(self, meta: dict, results: Dict[str, dict]):
        self.meta, self.results = meta, results

    def to_dict(self) -> dict:
        return {"schema": "prva.report/1", "meta": self.meta, "results": self.results}

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=1)

    def to_csv(self, path: str) -> None:
        import pandas as pd
        rows = []
        for dim, r in self.results.items():
            for i, lv in enumerate(r["levels"]):
                row = {"dimension": dim, "level": lv}
                row.update({k: v[i] for k, v in r["metrics"].items()})
                row.update({f"{k}_lo": v[i][0] for k, v in r["ci"].items()})
                row.update({f"{k}_hi": v[i][1] for k, v in r["ci"].items()})
                rows.append(row)
        pd.DataFrame(rows).to_csv(path, index=False)

    def plot(self, path: str) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        dims = list(self.results)
        fig, axes = plt.subplots(len(dims), 1, figsize=(9, 3.2 * len(dims)), squeeze=False)
        for ax, dim in zip(axes[:, 0], dims):
            r = self.results[dim]
            lv = r["levels"]
            for key, col in (("recall", "#1f77b4"), ("mean_conf", "#ff7f0e")):
                ax.plot(lv, r["metrics"][key], color=col, lw=2, label=key)
                if key in r["ci"]:
                    lo = [c[0] for c in r["ci"][key]]
                    hi = [c[1] for c in r["ci"][key]]
                    ax.fill_between(lv, lo, hi, color=col, alpha=0.2)
            ax.plot(lv, r["metrics"]["overconfidence"], color="#7f7f7f", lw=1, ls="--", label="overconfidence")
            if r["red_line"] is not None:
                ax.axvline(r["red_line"], color="red", lw=2, label=f"red line ({r['red_line']})")
            ax.set_title(f"{dim} - {self.meta.get('model_name', 'model')}")
            ax.set_ylim(-0.3, 1.05)
            ax.grid(alpha=0.3)
            ax.legend(loc="lower left", fontsize=8)
        axes[-1, 0].set_xlabel("stress level")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)


class DatasetAudit:
    def __init__(self, adapter, model_name: str = "model", iou_thr: float = 0.5, score_thr: float = 0.35,
                 frac: float = 0.8, n_boot: int = 1000, seed: int = 0):
        self.adapter, self.model_name = adapter, model_name
        self.iou_thr, self.score_thr, self.frac, self.n_boot, self.seed = iou_thr, score_thr, frac, n_boot, seed

    def run(self, images: Iterable, dims: List[str], lvls: Iterable[float]) -> DatasetReport:
        items = [(x if isinstance(x, tuple) else (x, None)) for x in images]
        lvls = sorted(set(float(v) for v in lvls) | {0.0})
        refs = []
        for img, gt in items:                                # reference: ground truth, or the clean prediction
            refs.append(metrics.normalize(gt) if gt is not None else
                        [p for p in metrics.normalize(self.adapter.predict(img)) if p["score"] >= self.score_thr])
        results = {}
        for dim in dims:
            op = OPERATORS[dim]
            per_level = {}
            for lv in lvls:
                rows = []
                for k, ((img, _), ref) in enumerate(zip(items, refs)):
                    st = op(img, lv, seed=self.seed + k) if dim == "ISO" else op(img, lv)
                    rows.append(metrics.match_image(ref, metrics.normalize(self.adapter.predict(st)),
                                                    self.iou_thr, self.score_thr))
                per_level[lv] = rows
            results[dim] = metrics.evaluate(per_level, self.n_boot, self.seed, self.frac)
        from . import __version__
        meta = {"prva_version": __version__, "model_name": self.model_name, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "dims": dims, "levels": lvls, "n_images": len(items), "image_sha256_16": [_hash(i) for i, _ in items],
                "reference": "ground truth" if any(g is not None for _, g in items) else "clean-image predictions",
                "iou_thr": self.iou_thr, "score_thr": self.score_thr, "red_line_frac": self.frac,
                "n_boot": self.n_boot, "seed": self.seed, "python": platform.python_version(),
                "note": "Results describe the tested model under the tested image conditions only; "
                        "not a standard or a certification."}
        return DatasetReport(meta, results)
