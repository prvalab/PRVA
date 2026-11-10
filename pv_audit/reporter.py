class AuditReporter:
    """single-image results (AuditAgent.run); pandas / seaborn are imported only when a file is written"""

    def __init__(self, data: list):
        self.data = data

    @property
    def df(self):
        import pandas as pd
        return pd.DataFrame(self.data)

    def to_csv(self, filepath: str):
        self.df.to_csv(filepath, index=False)

    def plot_trend(self, filepath: str):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=self.df, x="level", y="conf", hue="dimension", marker="o")
        plt.title("AI Model Robustness Trend")
        plt.grid(True)
        plt.savefig(filepath)
        plt.close()
