import numpy as np
from .stress import StressSimulator

class AuditEngine:
    def __init__(self, adapter):
        self.adapter = adapter
        self.operators = {
            "Lux+": StressSimulator.apply_lux_plus,
            "Lux-": StressSimulator.apply_lux_minus,
            "MBlur": StressSimulator.apply_mblur,
            "ISO": StressSimulator.apply_iso_noise,
            "DPI": StressSimulator.apply_dpi,
            "Defocus": StressSimulator.apply_defocus
        }

    def run_audit(self, image: np.ndarray, dims: list, lvls: list) -> list:
        report_data = []
        for dim in dims:
            for lvl in lvls:
                stressed_img = (self.operators[dim](image, lvl, seed=0) if dim == "ISO"
                                else self.operators[dim](image, lvl))
                prediction = self.adapter.predict(stressed_img)
                report_data.append({
                    "dimension": dim, 
                    "level": lvl, 
                    "conf": prediction['conf']
                })
        return report_data
