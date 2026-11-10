# PRVA (Physical Reality Validation & Assessment) 🛡️

**Quantifying the Operational Design Domain (ODD) for Industrial AI.**

[![Field](https://img.shields.io/badge/Industry-Industrial%20Automation-blue.svg)]()
[![Field](https://img.shields.io/badge/Field-Computer%20Vision-blue.svg)]()
[![Standard](https://img.shields.io/badge/Standard-Industrial%20ODD-orange.svg)]()
[![Focus](https://img.shields.io/badge/Focus-AI%20Robustness-gold.svg)]()
[![License](https://img.shields.io/badge/License-Apache%202.0-lightgrey.svg)](https://opensource.org/licenses/Apache-2.0)
[![SmartMfg](https://img.shields.io/badge/Smart%20Mfg-Industry%204.0-critical.svg)]()

[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)]()

---

## 🚩 About
In industrial deployment, AI model performance is often fragile against physical environmental fluctuations. **PRVA (Physical Reality Validation & Assessment)** is an open validation and evaluation method / framework designed to quantify the performance boundaries of vision models under varying physical stresses.

**The Core Philosophy:**
PRVA is an **open method for validating and assessing industrial AI reliability**. We believe that in mission-critical manufacturing, the training process is only the beginning—the **Reliability Loop** is what truly defines production readiness. 



---

## 📌 What PRVA is — and is not
PRVA organises existing ideas into one practical method: the **Operational Design Domain (ODD)** — the conditions a
system is designed to work in —, **SOTIF (ISO 21448, Safety Of The Intended Functionality)** — hazards that come from
the limits of perception rather than from faults —, and **robustness evaluation** of vision models under controlled
degradations. It helps you measure and document where a model stays reliable.
**PRVA is not a standard, not a certification, and not a substitute for functional-safety work** (e.g. ISO 26262,
IEC 61508, ISO 13849). Results describe the tested model under the tested conditions only.

---

## 🏗️ Core Modules
- **PhysCorruptor:** A simulation engine that injects controlled physical variables (Lux, DPI, Motion Blur, Defocus, ISO Noise) into raw samples to stress-test model robustness.
- **Audit Logger:** Outputs standardized performance metrics (CSV/JSON) for deep-dive production auditing.
- **Inference Bridge:** A model-agnostic interface supporting third-party architectures (e.g., YOLO, ResNet) via standard protocols without modifying core model code.
- **Analysis Tool:** Visualization utilities for mapping the **Safe Operating Area (SOA)** and identifying the model's "breaking point."

---

## 📊 Assessment Dimensions

| Physical Factor | Key (`dims`) | Industrial Context | Implementation Logic |
| :--- | :--- | :--- | :--- |
| **Lux** | `Lux+` / `Lux-` | Environmental light fluctuation | Linear scaling & Gamma adjustment (`Lux+`: gain 1 + level/50, gamma 1/(1 + level/100); `Lux-`: gain 1/(1 + level/50), gamma 1 + level/100) |
| **Motion Blur** | `MBlur` | Conveyor/Mechanical vibration | Directional convolution kernels (line of int(level/10)·2+1 px, horizontal by default; `apply_mblur(img, level, angle=…)` for other directions) |
| **DPI/Res** | `DPI` | Working distance variations | Area-based resampling (down to (100 − level)% per side, then back to full size) |
| **Defocus** | `Defocus` | Focal drift | Gaussian-based degradation (sigma = level / 10 px) |
| **ISO Noise** | `ISO` | Low-light sensor gain | Gaussian-Normal injection |

---

> Formerly *PVA (Physical Validator Audit)*. The Python package is still imported as `pv_audit`.

## 🚀 Quick Start

```bash
git clone https://github.com/shef9432/PRVA.git
cd PRVA
pip install -e .
```

Wrap your model in an adapter (one method), then run the audit:

```python
import cv2
from pv_audit import AuditAgent, BaseAdapter


class MyModel(BaseAdapter):
    def predict(self, image):
        # Run your own model here (YOLO, ResNet, ...).
        # Return its raw output and one confidence score for this image.
        result = ...  # your inference
        return {"data": result, "conf": 0.9}


image = cv2.imread("sample.jpg")
agent = AuditAgent(MyModel())

# Stress dimensions: "Lux+", "Lux-", "MBlur", "DPI", "Defocus", "ISO" (see Assessment Dimensions)
report = agent.run(image, dims=["Lux+", "Lux-", "MBlur", "DPI", "Defocus", "ISO"], lvls=[0, 20, 40, 60])

report.to_csv("audit.csv")       # dimension, level, conf
report.plot_trend("audit.png")   # confidence vs. stress level, one line per dimension
```

Run the tests: `python -m unittest discover -s tests`

---

## 📈 Dataset Audit (v0.2): metrics, confidence intervals, red line

`AuditAgent` scores one image. `DatasetAudit` runs the same stressors over a set of images and reports, per level:

| Metric | Meaning |
| :--- | :--- |
| `recall` | share of reference objects still found (same label, IoU ≥ 0.5, score ≥ 0.35) |
| `mean_iou` / `mean_conf` | IoU and confidence of the matched objects |
| `precision` | share of kept predictions that match a reference object |
| `overconfidence` | mean confidence of the kept predictions minus their precision (> 0: more confident than correct) |

The reference is your ground truth (`[{"box": [x1, y1, x2, y2], "label": ...}]` per image) or, without it, the
model's own detections on the clean image. Confidence intervals come from a bootstrap over images; the **red line** is
the first level whose recall upper bound falls below 0.8 × the clean recall (`frac` sets the factor). Every run is
seeded (ISO noise included), and the JSON report (`schema: prva.report/1`) records the PRVA version, levels, seed,
thresholds and a hash of every image.

```python
from pv_audit import DatasetAudit

class MyDetector:                      # return boxes, labels and scores (the legacy {"conf": x} form also works)
    def predict(self, image):
        return {"boxes": [[10, 20, 110, 220]], "labels": ["person"], "scores": [0.91]}

report = DatasetAudit(MyDetector(), model_name="my-detector").run(
    images,                             # [image, ...] or [(image, ground_truth), ...]
    dims=["DPI", "Lux-", "ISO"], lvls=range(0, 100, 10))
report.to_json("prva_report.json"); report.to_csv("prva_report.csv"); report.plot("prva_report.png")
```

## 🗺️ Roadmap: The Evolution of Industrial Reliability

### Phase 1: Physical Audit (The Present)
- **Infrastructure:** Established PhysCorruptor and Audit Logger.
- **Focus:** Quantifying the Safe Operating Area (SOA) under basic optical and environmental stressors.

### Phase 2: Parametric Engineering (In Progress)
- **Golden Reference Mapping:** Integrating high-fidelity "Golden Reference" data to map the optimal synergy between hardware and algorithmic parameters.
- **Degradation Engines:** Developing physically consistent degradation models to simulate complex material distortions and optical reflections.

### Phase 3: Cognitive Defense & Safety (Future)
- **Adversarial Closed-loop:** Implementation of "Anti-Fact" data injection to detect model hallucinations.
- **Re-calibration Engine:** Enabling automatic model rollbacks or re-calibration when the system enters a "Logic Wasteland" (i.e., non-physical confident predictions).
- **Cross-Domain Expansion:**
  - Control-PV: Assessing stability against network jitter and mechanical backlash.
  - Language-PV / Medical-PV: Extending robustness auditing to acoustic and bio-signal domains.

---

## 💬 Feedback & Inquiries
PRVA is an independent technical initiative aimed at making industrial AI predictable. Contributions, technical discussions, or inquiries regarding specific industrial use cases are highly welcome.

### 📜 License & Disclaimer
- **License:** This project is licensed under the [Apache 2.0](LICENSE) License.
- **Disclaimer:** This project is an independent technical initiative developed by **shef9432**. It is intended as a general-purpose tool for the industrial AI community and does not represent the official position of any specific organization or commercial entity.

---


**Author:** [shef9432](https://github.com/shef9432)  
**Connect:** [LinkedIn Profile](https://jp.linkedin.com/in/%E4%BD%B3%E4%BB%81-%E8%83%A1-0a020430a)
