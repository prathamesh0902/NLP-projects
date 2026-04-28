# models.py
import torch
import torch.nn as nn

EMB_DIM  = 64
HID_DIM  = 128
VOCAB_SZ = 10_000

# ── Baseline: no sequence awareness ──────────────────────────────
class FeedforwardNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SZ, EMB_DIM, padding_idx=0)
        self.fc  = nn.Sequential(
            nn.Linear(EMB_DIM, HID_DIM), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(HID_DIM, 1)
        )
    def forward(self, x):
        # mean-pool embeddings → bag-of-words, no order
        e = self.emb(x).mean(dim=1)
        return self.fc(e).squeeze(-1)

# ── RNN ───────────────────────────────────────────────────────────
class RNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SZ, EMB_DIM, padding_idx=0)
        self.rnn = nn.RNN(EMB_DIM, HID_DIM, batch_first=True)
        self.fc  = nn.Linear(HID_DIM, 1)
    def forward(self, x):
        _, h = self.rnn(self.emb(x))     # last hidden state
        return self.fc(h.squeeze(0)).squeeze(-1)

# ── LSTM ──────────────────────────────────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb  = nn.Embedding(VOCAB_SZ, EMB_DIM, padding_idx=0)
        self.lstm = nn.LSTM(EMB_DIM, HID_DIM, batch_first=True)
        self.fc   = nn.Linear(HID_DIM, 1)
    def forward(self, x):
        _, (h, _) = self.lstm(self.emb(x))
        return self.fc(h.squeeze(0)).squeeze(-1)

# ── Bidirectional LSTM ────────────────────────────────────────────
class BiLSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb   = nn.Embedding(VOCAB_SZ, EMB_DIM, padding_idx=0)
        self.bilstm = nn.LSTM(EMB_DIM, HID_DIM,
                              batch_first=True, bidirectional=True)
        self.fc = nn.Linear(HID_DIM * 2, 1)  # ×2 for fwd+bwd
    def forward(self, x):
        _, (h, _) = self.bilstm(self.emb(x))
        # concat forward (h[0]) and backward (h[1]) final states
        h_cat = torch.cat([h[0], h[1]], dim=-1)
        return self.fc(h_cat).squeeze(-1)