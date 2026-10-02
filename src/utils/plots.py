# графики: важность фичей ранкера и сравнение моделей
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils.report import NAMES


def feature_importance(names, values, path, top=20):
    order = np.argsort(values)[-top:]
    plt.figure(figsize=(7, 6))
    plt.barh(np.array(names)[order], np.array(values)[order])
    plt.title("важность фичей ранкера (PredictionValuesChange)")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def metrics_bar(metrics, metric, path):
    keys = list(metrics)
    vals = [metrics[k][metric] for k in keys]
    plt.figure(figsize=(7, 3.5))
    bars = plt.barh([NAMES[k] for k in keys], vals)
    for b, v in zip(bars, vals):
        plt.text(b.get_width(), b.get_y() + b.get_height() / 2, f" {v:.4f}", va="center")
    plt.title(metric.replace("ndcg", "NDCG").replace("recall", "Recall") + " на тестовом дне")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
