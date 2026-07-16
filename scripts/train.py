# train.py

import torch
import torch.nn as nn

from tqdm import tqdm

from torch.utils.data import DataLoader

from stroke_dataset import StrokeDataset
from transformer_model import StrokeTransformer


BATCH_SIZE = 32
EPOCHS = 70
LR = 1e-3

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)

train_dataset = StrokeDataset(
    "X_train.npy",
    "y_train.npy"
)

val_dataset = StrokeDataset(
    "X_val.npy",
    "y_val.npy"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE
)

model = StrokeTransformer()

model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)

best_val_acc = 0

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0

    for X, y in tqdm(train_loader):

        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(X)

        loss = criterion(
            logits,
            y
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for X, y in val_loader:

            X = X.to(device)
            y = y.to(device)

            pred = model(X)

            pred = pred.argmax(1)

            correct += (
                pred == y
            ).sum().item()

            total += y.size(0)

    val_acc = correct / total

    print(
        f"Epoch {epoch+1:02d} | "
        f"Loss {running_loss:.4f} | "
        f"Val Acc {val_acc:.4f}"
    )

    if val_acc > best_val_acc:

        best_val_acc = val_acc

        torch.save(
            model.state_dict(),
            "best_model.pt"
        )

print(
    "\nBest Validation Accuracy:",
    best_val_acc
)