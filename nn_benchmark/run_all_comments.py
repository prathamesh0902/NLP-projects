"""
run_all.py  —  single entry point for the benchmark
Usage:  python run_all.py
"""
import json, os
from data     import get_loaders
from models   import FeedforwardNN, RNNModel, LSTMModel, BiLSTMModel
from train    import train_model
from evaluate import plot_results, print_summary

os.makedirs("plots", exist_ok=True)

# ── 1. Load data ────────────────────────────────────────────────────
train_loader, test_loader, vocab = get_loaders()

# ── 2. Define models in order (FF is the baseline) ─────────────────
MODELS = {
    "FF (baseline)": FeedforwardNN(),
    "RNN":           RNNModel(),
    "LSTM":          LSTMModel(),
    "BiLSTM":        BiLSTMModel(),
}

# ── 3. Train each model and collect results ─────────────────────────
results = {}
for name, model in MODELS.items():
    results[name] = train_model(name, model, train_loader, test_loader)

# ── 4. Save raw results ─────────────────────────────────────────────
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to results.json")

# ── 5. Print summary table and plot ─────────────────────────────────
print_summary(results)
plot_results(results)