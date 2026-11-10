from .interface import BaseAdapter
from .engine import AuditEngine
from .reporter import AuditReporter
from .dataset import DatasetAudit, DatasetReport

class AuditAgent:
    def __init__(self, adapter):
        self.engine = AuditEngine(adapter)
    
    def run(self, image, dims, lvls):
        raw_data = self.engine.run_audit(image, dims, lvls)
        return AuditReporter(raw_data)

__version__ = "0.2.0"