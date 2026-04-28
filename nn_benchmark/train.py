# train.py
import time, torch, torch.nn as nn
from torch.optim import Adam

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 5
LR     = 1e-3

def train_model(model, train_loader, test_loader):
    model = model.to(DEVICE)
    opt   = Adam(model.parameters(), lr=LR)
    crit  = nn.BCEWithLogitsLoss()

    history = {"train_loss": [], "val_acc": [], "epoch_time": []}
    total_start = time.time()

    for epoch in range(EPOCHS):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.float().to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running_loss += loss.item()

        epoch_time = time.time() - epoch_start
        val_acc = evaluate(model, test_loader)

        history["train_loss"].append(running_loss / len(train_loader))
        history["val_acc"].append(val_acc)
        history["epoch_time"].append(epoch_time)

        print(f"  epoch {epoch+1}/{EPOCHS}  loss={running_loss/len(train_loader):.4f}"
              f"  val_acc={val_acc:.4f}  time={epoch_time:.1f}s")

    history["total_time"] = time.time() - total_start
    return history

def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = (model(xb) > 0).long()
            correct += (preds == yb).sum().item()
            total   += len(yb)
    return correct / total