# evaluate.py  —  run after train.py to get full report + plots
import json, matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import torch, numpy as np

def get_preds(model, loader, device):
    model.eval(); ys, ps = [], []
    with torch.no_grad():
        for xb, yb in loader:
            ps += (model(xb.to(device)) > 0).cpu().tolist()
            ys += yb.tolist()
    return ys, ps

def plot_results(results: dict):
    names  = list(results.keys())
    colors = ["#888780", "#EF9F27", "#378ADD", "#7F77DD"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1) val accuracy over epochs
    ax = axes[0]
    for name, c in zip(names, colors):
        ax.plot(results[name]["val_acc"], label=name, color=c, marker="o")
    ax.set_title("Sequence handling: val accuracy")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.legend(); ax.grid(True, alpha=0.3)

    # 2) training loss curves (temporal modeling proxy)
    ax = axes[1]
    for name, c in zip(names, colors):
        ax.plot(results[name]["train_loss"], label=name, color=c, marker="s")
    ax.set_title("Temporal modeling: train loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("BCE Loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    # 3) total training time (bar)
    ax = axes[2]
    times = [results[n]["total_time"] for n in names]
    bars = ax.bar(names, times, color=colors, width=0.5)
    ax.bar_label(bars, fmt="%.0fs")
    ax.set_title("Training speed: total time (5 epochs)")
    ax.set_ylabel("Seconds")

    plt.tight_layout()
    plt.savefig("plots/comparison.png", dpi=150)
    plt.show()