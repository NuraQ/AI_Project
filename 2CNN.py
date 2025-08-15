import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import json
import os
import numpy as np
import torch
from collections import defaultdict
from sklearn.model_selection import train_test_split

DATA_DIR = './1segment_topomap_5channel_11classes'
VALID_LABELS = {2, 3, 6, 7}
LABEL_MAP = {2: 0, 3: 1, 6: 2, 7: 3}

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    cohen_kappa_score, matthews_corrcoef
)
from sklearn.preprocessing import OneHotEncoder


def evaluate_model(model, loader, device, n_classes, split_name=""):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(X_batch)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())
            all_labels.append(y_batch.cpu())

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()
    y_probs = torch.cat(all_probs).numpy()

    encoder = OneHotEncoder(sparse_output=False)
    y_true_1hot = encoder.fit_transform(y_true.reshape(-1,1))

    try:
        auc = roc_auc_score(y_true_1hot, y_probs, average='macro', multi_class='ovr')
    except:
        auc = 0.0

    results = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'cohen_kappa': cohen_kappa_score(y_true, y_pred),
        'matthews_corrcoef': matthews_corrcoef(y_true, y_pred),
        'auc': auc,
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        'classification_report': classification_report(y_true, y_pred, digits=4)
    }

    print(f"\n📊 {split_name} Evaluation:")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"{k:<25}: {v:.4f}")
    print("\n" + results['classification_report'])

    return results

subject_files = defaultdict(list)
for fname in os.listdir(DATA_DIR):
    if fname.endswith('.npz'):
        parts = fname.split('_')
        subject_id = parts[1]  # e.g., S001
        subject_files[subject_id].append(os.path.join(DATA_DIR, fname))

all_subjects = sorted(subject_files.keys())
train_subj, temp_subj = train_test_split(all_subjects, test_size=0.3, random_state=42)
dev_subj, test_subj = train_test_split(temp_subj, test_size=0.5, random_state=42)


print(f" Subjects  Train: {len(train_subj)}, Dev: {len(dev_subj)}, Test: {len(test_subj)}")

# Step 3: Load chunks and filter by VALID_LABELS
def load_chunks(subject_ids):
    X_list, y_list = [], []
    for sid in subject_ids:
        for file in subject_files[sid]:
            with np.load(file) as data:
                X = data['X']  # shape: (5, 64, 64)
                y = int(data['y'])
                if y in VALID_LABELS:
                    X_list.append(X)
                    y_list.append(LABEL_MAP[y])  # map to 0–3
    return np.array(X_list), np.array(y_list)

X_train, y_train = load_chunks(train_subj)
X_dev, y_dev     = load_chunks(dev_subj)
X_test, y_test   = load_chunks(test_subj)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_dev_tensor   = torch.tensor(X_dev, dtype=torch.float32)
X_test_tensor  = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_dev_tensor   = torch.tensor(y_dev, dtype=torch.long)
y_test_tensor  = torch.tensor(y_test, dtype=torch.long)

from collections import Counter
print("Train:", Counter(y_train.tolist()))
print("Dev:  ", Counter(y_dev.tolist()))
print("Test: ", Counter(y_test.tolist()))
print(f"\n Final Shapes:")
print(f"Train: {X_train_tensor.shape}, Labels: {set(y_train_tensor.tolist())}")
print(f"Dev:   {X_dev_tensor.shape}, Labels: {set(y_dev_tensor.tolist())}")
print(f"Test:  {X_test_tensor.shape}, Labels: {set(y_test_tensor.tolist())}")

class TopomapCNN(nn.Module):
    def __init__(self, in_channels=5, num_classes=4):
        super(TopomapCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 64 → 32
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 32 → 16
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)  # (N, 32, 32, 32)
        x = self.conv2(x)  # (N, 64, 16, 16)
        x = self.conv3(x)  # (N, 128, 1, 1)
        x = x.view(x.size(0), -1)  # flatten
        return self.fc(x)

# --- Training + Evaluation Functions ---
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb)
            preds = out.argmax(dim=1).cpu().numpy()
            y_true.extend(yb.numpy())
            y_pred.extend(preds)
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4)
    return acc, report

# Replace these with the actual tensors from your loader
# X_train_tensor, y_train_tensor, X_dev_tensor, y_dev_tensor, X_test_tensor, y_test_tensor

BATCH_SIZE = 64
EPOCHS = 15
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_ds = TensorDataset(X_train_tensor, y_train_tensor)
dev_ds   = TensorDataset(X_dev_tensor, y_dev_tensor)
test_ds  = TensorDataset(X_test_tensor, y_test_tensor)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
dev_dl   = DataLoader(dev_ds, batch_size=BATCH_SIZE)
test_dl  = DataLoader(test_ds, batch_size=BATCH_SIZE)

model = TopomapCNN(in_channels=5, num_classes=4).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# --- Training Loop ---
for epoch in range(1, EPOCHS + 1):
    loss = train(model, train_dl, optimizer, criterion, DEVICE)
    acc, _ = evaluate(model, dev_dl, DEVICE)
    print(f"Epoch {epoch:02d} | Train Loss: {loss:.4f} | Dev Acc: {acc:.4f}")

# --- Final Evaluation ---
print("\n📊 Evaluation on Test Set:")
test_acc, test_report = evaluate(model, test_dl, DEVICE)

dev_metrics = evaluate_model(model, dev_dl, DEVICE, n_classes=4, split_name="Dev")
train_metrics = evaluate_model(model, train_dl, DEVICE, n_classes=4, split_name="Train")
test_metrics = evaluate_model(model, test_dl, DEVICE, n_classes=4, split_name="Test")

results = {
'dev_metrics':dev_metrics,
'train_metrics':train_metrics,
'test_metrics':test_metrics
}
print(test_report)
respath =f'./2DCNN_1Segment_5Channels_classes_{len(VALID_LABELS)}'
with open(respath, 'w') as f:
    json.dump(results, f, indent=4)
print(f" Saved CNN results to {respath}")
