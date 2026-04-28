# data.py
import torch
from datasets import load_dataset
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset

VOCAB_SIZE = 10_000
MAX_LEN    = 200
BATCH_SIZE = 64

def build_vocab(texts, max_vocab=VOCAB_SIZE):
    counter = Counter(w for t in texts for w in t.split())
    vocab = {"<pad>": 0, "<unk>": 1}
    for w, _ in counter.most_common(max_vocab - 2):
        vocab[w] = len(vocab)
    return vocab

def encode(texts, vocab, max_len=MAX_LEN):
    out = []
    for t in texts:
        ids = [vocab.get(w, 1) for w in t.split()][:max_len]
        ids += [0] * (max_len - len(ids))   # pad to MAX_LEN
        out.append(ids)
    return torch.tensor(out, dtype=torch.long)

def get_loaders():
    ds = load_dataset("imdb")
    train_texts = ds["train"]["text"]
    train_labels = torch.tensor(ds["train"]["label"])
    test_texts  = ds["test"]["text"]
    test_labels = torch.tensor(ds["test"]["label"])

    vocab = build_vocab(train_texts)
    X_train = encode(train_texts, vocab)
    X_test  = encode(test_texts, vocab)

    train_loader = DataLoader(
        TensorDataset(X_train, train_labels),
        batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(X_test, test_labels),
        batch_size=BATCH_SIZE
    )
    return train_loader, test_loader, vocab