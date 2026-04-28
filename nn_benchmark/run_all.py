# run_all.py ────────────────────────────────────────────────────────
from data   import get_loaders
from models import FeedforwardNN, RNNModel, LSTMModel, BiLSTMModel
from train  import train_model
from evaluate import plot_results
import json, os

os.makedirs("plots", exist_ok=True)
train_loader, test_loader, vocab = get_loaders()

MODELS = {
    "FF (baseline)": FeedforwardNN(),
    "RNN":           RNNModel(),
    "LSTM":          LSTMModel(),
    "BiLSTM":        BiLSTMModel(),
}

results = {}
for name, model in MODELS.items():
    print(f"\n=== {name} ===")
    results[name] = train_model(model, train_loader, test_loader)

json.dump(results, open("results.json", "w"), indent=2)
plot_results(results)