import json
import os
import tempfile
import unittest

import numpy as np

from pv_audit import DatasetAudit, metrics
from pv_audit.stress import StressSimulator


class TestMatching(unittest.TestCase):
    def test_greedy_one_to_one(self):
        refs = [{"box": [0, 0, 10, 10], "label": 1}, {"box": [20, 20, 30, 30], "label": 1}]
        preds = [{"box": [0, 0, 10, 10], "label": 1, "score": 0.9}, {"box": [1, 1, 10, 10], "label": 1, "score": 0.8}]
        r = metrics.match_image(refs, preds)
        self.assertEqual(r["tp"], 1)                       # two predictions on one object count once
        self.assertEqual(r["n_pred"], 2)

    def test_label_and_score_threshold(self):
        refs = [{"box": [0, 0, 10, 10], "label": "car"}]
        self.assertEqual(metrics.match_image(refs, [{"box": [0, 0, 10, 10], "label": "person", "score": 0.9}])["tp"], 0)
        self.assertEqual(metrics.match_image(refs, [{"box": [0, 0, 10, 10], "label": "car", "score": 0.2}])["tp"], 0)

    def test_normalize_forms(self):
        a = metrics.normalize({"boxes": [[0, 0, 1, 1]], "labels": [3], "scores": [0.5]})
        b = metrics.normalize([{"box": [0, 0, 1, 1], "label": 3, "conf": 0.5}])
        self.assertEqual(a, b)
        self.assertEqual(metrics.normalize({"conf": 0.9}), [])


class TestSummary(unittest.TestCase):
    def test_overconfidence_sign(self):
        rows = [{"n_ref": 1, "n_pred": 2, "tp": 1, "iou_sum": 0.9, "conf_sum": 0.9, "score_sum_all": 1.8}]
        s = metrics.summarise(rows)
        self.assertAlmostEqual(s["precision"], 0.5)
        self.assertAlmostEqual(s["overconfidence"], 0.9 - 0.5)

    def test_red_line_rule(self):
        self.assertEqual(metrics.red_line([0, 10, 20, 30], [1.0, 0.9, 0.6, 0.1], [1.0, 0.95, 0.75, 0.2], 0.8), 20)
        self.assertIsNone(metrics.red_line([0, 10], [1.0, 0.9], [1.0, 0.95], 0.8))

    def test_bootstrap_is_deterministic(self):
        rows = [{"n_ref": 1, "n_pred": 1, "tp": int(i % 3 != 0), "iou_sum": 0.8, "conf_sum": 0.8, "score_sum_all": 0.8}
                for i in range(30)]
        self.assertEqual(metrics.bootstrap_ci(rows, "recall", 200, seed=1), metrics.bootstrap_ci(rows, "recall", 200, seed=1))
        lo, hi = metrics.bootstrap_ci(rows, "recall", 200, seed=1)
        self.assertTrue(lo <= metrics.summarise(rows)["recall"] <= hi)


class BrightSquare:
    """toy detector: finds the bright square while it stays sharp enough"""
    def predict(self, img):
        g = img.mean(axis=2)
        ys, xs = np.where(g > 200)
        if len(xs) < 50:
            return {"boxes": [], "labels": [], "scores": []}
        return {"boxes": [[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]], "labels": [1], "scores": [min(1.0, len(xs) / 1600)]}


class TestDatasetAudit(unittest.TestCase):
    def images(self):
        out = []
        for k in range(4):
            img = np.full((120, 160, 3), 40, np.uint8)
            img[30 + k:70 + k, 50:90] = 255
            out.append(img)
        return out

    def test_report_has_red_line_and_meta(self):
        rep = DatasetAudit(BrightSquare(), "toy", n_boot=100).run(self.images(), ["Lux-"], range(0, 100, 10))
        r = rep.results["Lux-"]
        self.assertEqual(r["metrics"]["recall"][0], 1.0)
        self.assertIsNotNone(r["red_line"])
        self.assertEqual(rep.meta["n_images"], 4)
        with tempfile.TemporaryDirectory() as d:
            rep.to_json(os.path.join(d, "r.json"))
            rep.to_csv(os.path.join(d, "r.csv"))
            rep.plot(os.path.join(d, "r.png"))
            self.assertEqual(json.load(open(os.path.join(d, "r.json")))["schema"], "prva.report/1")

    def test_iso_is_reproducible(self):
        img = self.images()[0]
        a = StressSimulator.apply_iso_noise(img, 30, seed=5)
        self.assertTrue(np.array_equal(a, StressSimulator.apply_iso_noise(img, 30, seed=5)))
        self.assertTrue(np.array_equal(StressSimulator.apply_iso_noise(img, 0), img))


if __name__ == "__main__":
    unittest.main()
