import copy
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from point_sequence_dataset import PointSequenceDataset, load_uji_samples
from point_sequence_transformer import PointSequenceTransformer

DATA_DIR = PROJECT_ROOT / "data"
MODEL_PATH = PROJECT_ROOT / "best_point_model.pt"

SEED = 42
MAX_POINTS = 96
BATCH_SIZE = 64
EPOCHS = 350
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-2
PATIENCE = 45
AUGMENT_REPEATS = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0
    predictions = []
    labels = []

    with torch.no_grad():
        for xb, mb, yb in loader:
            xb = xb.to(DEVICE)
            mb = mb.to(DEVICE)
            yb = yb.to(DEVICE)

            logits = model(xb, mb)
            preds = logits.argmax(1)

            correct += (preds == yb).sum().item()
            total += len(yb)
            predictions.extend(preds.cpu().numpy())
            labels.extend(yb.cpu().numpy())

    return correct / total, labels, predictions


samples = load_uji_samples(DATA_DIR)
classes = np.array(sorted({sample["label"] for sample in samples}))
label_to_index = {label: index for index, label in enumerate(classes)}
labels = np.array([label_to_index[sample["label"]] for sample in samples])
indices = np.arange(len(samples))

train_idx, temp_idx = train_test_split(
    indices,
    test_size=0.30,
    random_state=SEED,
    stratify=labels,
)

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.50,
    random_state=SEED,
    stratify=labels[temp_idx],
)

train_samples = [samples[i] for i in train_idx]
val_samples = [samples[i] for i in val_idx]
test_samples = [samples[i] for i in test_idx]

# With only 22 samples per class, light augmentation helps the model see
# small writing variations without changing the underlying labels.
augmented_train_samples = train_samples * AUGMENT_REPEATS

train_loader = DataLoader(
    PointSequenceDataset(
        augmented_train_samples,
        label_to_index,
        max_points=MAX_POINTS,
        augment=True,
        seed=SEED,
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_loader = DataLoader(
    PointSequenceDataset(val_samples, label_to_index, max_points=MAX_POINTS),
    batch_size=BATCH_SIZE,
)

test_loader = DataLoader(
    PointSequenceDataset(test_samples, label_to_index, max_points=MAX_POINTS),
    batch_size=BATCH_SIZE,
)

print("Dataset Loaded")
print("Raw samples       :", len(samples))
print("Classes           :", len(classes))
print("Train/Val/Test    :", len(train_samples), len(val_samples), len(test_samples))
print("Augmented training:", len(augmented_train_samples))
print("Device            :", DEVICE)

model = PointSequenceTransformer(
    input_dim=5,
    embed_dim=128,
    num_heads=8,
    num_layers=4,
    num_classes=len(classes),
    max_points=MAX_POINTS,
    dropout=0.20,
).to(DEVICE)

criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=10,
)

best_val_acc = 0.0
best_weights = None
wait = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for xb, mb, yb in train_loader:
        xb = xb.to(DEVICE)
        mb = mb.to(DEVICE)
        yb = yb.to(DEVICE)

        optimizer.zero_grad()
        logits = model(xb, mb)
        loss = criterion(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(yb)
        correct += (logits.argmax(1) == yb).sum().item()
        total += len(yb)

    train_acc = correct / total
    train_loss = total_loss / total
    val_acc, _, _ = accuracy(model, val_loader)
    scheduler.step(val_acc)

    print(
        f"Epoch {epoch + 1:03d} | "
        f"Loss {train_loss:.4f} | "
        f"Train {train_acc:.4f} | "
        f"Val {val_acc:.4f}"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_weights = copy.deepcopy(model.state_dict())
        torch.save(best_weights, MODEL_PATH)
        wait = 0
    else:
        wait += 1

    if wait >= PATIENCE:
        print("\nEarly stopping")
        break

if best_weights is not None:
    model.load_state_dict(best_weights)

test_acc, y_true, y_pred = accuracy(model, test_loader)

print("\n==========================")
print("Point Transformer Complete")
print("==========================")
print("Best Validation :", round(best_val_acc * 100, 2), "%")
print("Test Accuracy   :", round(test_acc * 100, 2), "%")
print("Saved Model     :", MODEL_PATH)
print("==========================")

print("\nClassification Report")
print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))

print("Confusion Matrix")
import pandas as pd

true_chars = [classes[i] for i in y_true]
pred_chars = [classes[i] for i in y_pred]

pd.DataFrame({"true_label": true_chars, "pred_label": pred_chars}).to_csv(
    PROJECT_ROOT / "y_true_y_pred.csv", index=False
)
print("Saved y_true_y_pred.csv")

