import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, cohen_kappa_score,
                             matthews_corrcoef, classification_report)
from scipy.sparse import coo_matrix

from torch.nn import Sequential, Linear, ReLU
import json
import itertools

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

DATA_DIR = '../dataset_unzipped'
TRAIN_IDS = list(range(1, 74))
DEV_IDS = list(range(75, 89))
TEST_IDS = list(range(90, 104))
no_feature, segment_length, n_class = 64, 16, 4


def extract(input, n_classes, n_fea, time_window, moving):
    xx, yy = input[:, :n_fea], input[:, n_fea:n_fea + 1]
    new_x, new_y = [], []
    for i in range(int((xx.shape[0] / moving) - 1)):
        ave_y = np.average(yy[int(i * moving):int(i * moving + time_window)])
        new_x.append(xx[int(i * moving):int(i * moving + time_window), :])
        window = yy[int(i * moving):int(i * moving + time_window)]
        label = np.bincount(window.astype(int).flatten()).argmax()
        new_y.append(label)

    new_x = np.array(new_x).reshape([-1, n_fea * time_window])
    new_y = np.array(new_y).reshape([-1, 1])

    return np.vstack((np.hstack((new_x, new_y)), np.hstack((new_x[-1], new_y[-1]))))


def load_subjects(ids):
    X_list, y_list = [], []
    for sid in ids:
        if sid >= 78 and sid <= 81:
            continue
        path = os.path.join(DATA_DIR, f'{sid}.npy')
        if not os.path.exists(path):
            print(f" {path} Missing {sid}")
            continue
        data = np.load(path)

        seg_data = extract(data, n_class, no_feature, segment_length, segment_length / 2)
        X = seg_data[:, :-1]
        y = seg_data[:, -1]

        mask = (y == 2) | (y == 3) | (y == 6) | (y == 7)
        X, y = X[mask], y[mask]

        if len(y) == 0:
            print(f" No valid labels for subject {sid}")
            continue

        label_map = {2: 0, 3: 1, 6: 2, 7: 3}
        y = np.vectorize(label_map.get)(y)
        X_list.append(X)
        y_list.append(y)

    if not X_list:
        raise ValueError("No data loaded after filtering.")

    return np.vstack(X_list), np.hstack(y_list)


X_train_raw, y_train = load_subjects(TRAIN_IDS)
X_dev_raw, y_dev = load_subjects(DEV_IDS)
X_test_raw, y_test = load_subjects(TEST_IDS)
print("Train classes:", np.unique(y_train, return_counts=True))
print("Dev classes:", np.unique(y_dev, return_counts=True))
print("Test classes:", np.unique(y_test, return_counts=True))

all_labels = np.concatenate([y_train, y_dev, y_test])
unique_classes, counts = np.unique(all_labels, return_counts=True)
print("Unique classes after filtering and mapping:")
for cls, cnt in zip(unique_classes, counts):
    print(f"Class {cls}: {cnt} samples")

scaler = StandardScaler().fit(X_train_raw.reshape(-1, no_feature))
reshape_data = lambda X: scaler.transform(X.reshape(-1, no_feature)).reshape(-1, segment_length, no_feature)
X_train, X_dev, X_test = map(reshape_data, [X_train_raw, X_dev_raw, X_test_raw])

X_train_tensor, X_dev_tensor, X_test_tensor = map(lambda x: torch.tensor(x, dtype=torch.float32), [X_train, X_dev, X_test])
y_train_tensor, y_dev_tensor, y_test_tensor = map(lambda y: torch.tensor(y, dtype=torch.long), [y_train, y_dev, y_test])

edge_index = torch.from_numpy(np.vstack(coo_matrix(torch.ones([no_feature, no_feature])).nonzero())).to(torch.long).to(device)
flatten = lambda X: X.view(-1, no_feature)
make_batch = lambda n: torch.arange(n).repeat_interleave(segment_length).to(device)

X_train_flat, train_batch = flatten(X_train_tensor), make_batch(X_train_tensor.shape[0])
X_dev_flat, dev_batch = flatten(X_dev_tensor), make_batch(X_dev_tensor.shape[0])
X_test_flat, test_batch = flatten(X_test_tensor), make_batch(X_test_tensor.shape[0])

print('********* X_train_flat.shape ********** train_batch.shape ***********')
print(X_train_flat.shape, train_batch.shape)

# ==== DATALOADERS (permute to (B, C=64, T=16)) ====
BATCH_SIZE = 512
NUM_CLASSES = n_class 

def to_channels_first(x):  # (N, 16, 64) -> (N, 64, 16)
    return x.permute(0, 2, 1).contiguous()

train_ds = TensorDataset(to_channels_first(X_train_tensor), y_train_tensor)
dev_ds   = TensorDataset(to_channels_first(X_dev_tensor),   y_dev_tensor)
test_ds  = TensorDataset(to_channels_first(X_test_tensor),  y_test_tensor)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
dev_loader   = DataLoader(dev_ds,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# ==== CNN-BiLSTM MODEL ====
class CNNBiLSTM(nn.Module):
    def __init__(self, num_classes=2, cnn_out_channels=32, lstm_hidden=64, dropout=0.25):
        super().__init__()
        # CNN: input (B, 1, 64, 16)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, cnn_out_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(cnn_out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(cnn_out_channels, cnn_out_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(cnn_out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # Output shape: (B, C, H=64, W=16) → permute to (B, 16, C×64)
        self.lstm_input_dim = cnn_out_channels * 64
        self.lstm = nn.LSTM(input_size=self.lstm_input_dim,
                            hidden_size=lstm_hidden,
                            num_layers=1,
                            bidirectional=True,
                            batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(2 * lstm_hidden, num_classes)
        )

    def forward(self, x):  # x: (B, 64, 16)
        x = x.unsqueeze(1)  # → (B, 1, 64, 16)
        x = self.cnn(x)     # → (B, C, 64, 16)
        x = x.permute(0, 3, 1, 2).contiguous()  # → (B, 16, C, 64)
        x = x.view(x.size(0), x.size(1), -1)    # → (B, 16, C*64)
        x, _ = self.lstm(x)                     # → (B, 16, 2*hidden)
        x = x[:, -1, :]                         # last time step
        x = self.classifier(x)                  # → (B, num_classes)
        return x

model = CNNBiLSTM(num_classes=NUM_CLASSES, cnn_out_channels=32, lstm_hidden=64, dropout=0.3).to(device)

# ==== TCN MODEL (with GroupNorm & dilation 8) ====
class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.depthwise = nn.Conv1d(in_ch, in_ch, kernel_size,
                                   padding=pad, dilation=dilation, groups=in_ch, bias=False)
        self.pointwise = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(8, out_ch)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = F.relu(x, inplace=True)
        x = self.dropout(x)
        return x

class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        self.conv = DepthwiseSeparableConv1d(in_ch, out_ch, kernel_size, dilation, dropout)
        self.residual = nn.Identity() if in_ch == out_ch else nn.Conv1d(in_ch, out_ch, 1, bias=False)
        # CHANGED: GroupNorm for post-residual
        self.post_norm = nn.GroupNorm(8, out_ch)

    def forward(self, x):
        out = self.conv(x)
        res = self.residual(x)
        out = F.relu(self.post_norm(out + res), inplace=True)  # CHANGED
        return out

class SimpleTCN(nn.Module):
    def __init__(self, in_ch=64, base_ch=128, num_classes=2, kernel_size=3, dropout=0.1,
                 dilations=(1, 2, 4, 8)):  # CHANGED: added dilation 8
        super().__init__()
        # Make channels list match number of blocks
        chs = [in_ch] + [base_ch] * len(dilations)
        blocks = []
        for i, d in enumerate(dilations):
            blocks.append(
                TCNBlock(chs[i], chs[i+1], kernel_size=kernel_size, dilation=d, dropout=dropout)
            )
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(base_ch, num_classes)
        )

    def forward(self, x):  # x: (B, 64, 16)
        x = self.tcn(x)
        x = self.head(x)
        return x

# # choose either this one or the cnn_LSTM model
# model = SimpleTCN(in_ch=64, base_ch=128, num_classes=NUM_CLASSES, kernel_size=3,
#                   dropout=0.25, dilations=(1, 2, 4, 8)).to(device)

classes, counts = np.unique(y_train, return_counts=True)
weights = counts.sum() / (len(classes) * counts)
class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
# CHANGED: label_smoothing
# criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
#
#
# # ==== OPTIMIZER / SCHEDULER ====
# LR = 3e-4
# # CHANGED: higher weight_decay
# optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

# ==== TRAIN / EVAL UTILS ====
def run_epoch(loader, train: bool):
    model.train(mode=train)
    total_loss, n = 0.0, 0
    all_probs, all_preds, all_labels = [], [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += loss.item() * xb.size(0)
        n += xb.size(0)
        probs = F.softmax(logits, dim=1).detach().cpu().numpy()
        preds = probs.argmax(1)
        all_probs.append(probs)
        all_preds.append(preds)
        all_labels.append(yb.cpu().numpy())
    avg_loss = total_loss / max(n, 1)
    probs = np.concatenate(all_probs)
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    return avg_loss, probs, preds, labels

def compute_metrics(probs, preds, labels, num_classes=2):
    acc = accuracy_score(labels, preds)
    prec_w = precision_score(labels, preds, average='weighted', zero_division=0)
    rec_w  = recall_score(labels, preds, average='weighted', zero_division=0)
    f1_w   = f1_score(labels, preds, average='weighted', zero_division=0)
    prec_m = precision_score(labels, preds, average='macro', zero_division=0)
    rec_m  = recall_score(labels, preds, average='macro', zero_division=0)
    f1_m   = f1_score(labels, preds, average='macro', zero_division=0)
    try:
        if num_classes == 2:
            auc = roc_auc_score(labels, probs[:, 1])
        else:
            auc = roc_auc_score(labels, probs, multi_class='ovr')
    except Exception:
        auc = float('nan')
    mcc = matthews_corrcoef(labels, preds)
    kappa = cohen_kappa_score(labels, preds)
    cm = confusion_matrix(labels, preds)
    cr = classification_report(labels, preds, zero_division=0)
    return {
        "accuracy": acc, "precision_weighted": prec_w, "recall_weighted": rec_w, "f1_weighted": f1_w,
        "precision_macro": prec_m, "recall_macro": rec_m, "f1_macro": f1_m,
        "auc": auc, "matthews_corrcoef": mcc, "cohen_kappa": kappa,
        "confusion_matrix": cm.tolist(), "classification_report": cr
    }

# ==== TRAINING LOOP WITH EARLY STOP (note some tests I disabled it and kept it for others) ====
EPOCHS = 30
PATIENCE = 7
best_dev = -1
best_state = None
t0 = time.time()
patience_left = PATIENCE  # CHANGED: initialize before loop

# ==== EXPERIMENT CONFIG LIST ====
experiments = [
    {"lr": 3e-4, "epochs": 30, "dropout": 0.3, "batch_size": 512, "weight_decay": 1e-3},
    {"lr": 1e-3, "epochs": 40, "dropout": 0.4, "batch_size": 256, "weight_decay": 1e-4},
    {"lr": 5e-4, "epochs": 25, "dropout": 0.5, "batch_size": 512, "weight_decay": 5e-4},
]

results_all = []

def train_and_evaluate(config):
    print(f"\n Starting experiment: {config}")
    model = CNNBiLSTM(
        num_classes=NUM_CLASSES,
        cnn_out_channels=32,
        lstm_hidden=64,
        dropout=config["dropout"]
    ).to(device)

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, drop_last=False)
    dev_loader   = DataLoader(dev_ds,   batch_size=config["batch_size"], shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=config["batch_size"], shuffle=False, drop_last=False)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

    best_dev = -1
    best_state = None
    patience_left = PATIENCE
    t0 = time.time()

    for epoch in range(1, config["epochs"] + 1):
        tr_loss, tr_probs, tr_preds, tr_labels = run_epoch(train_loader, train=True)
        dv_loss, dv_probs, dv_preds, dv_labels = run_epoch(dev_loader, train=False)
        scheduler.step()

        tr_metrics = compute_metrics(tr_probs, tr_preds, tr_labels, NUM_CLASSES)
        dv_metrics = compute_metrics(dv_probs, dv_preds, dv_labels, NUM_CLASSES)

        print(f"Epoch {epoch:02d} | "
              f"train_loss {tr_loss:.4f} acc {tr_metrics['accuracy']:.3f} | "
              f"dev_loss {dv_loss:.4f} acc {dv_metrics['accuracy']:.3f} auc {dv_metrics['auc']:.3f}")

        score = dv_metrics['f1_weighted']
        if score > best_dev:
            best_dev = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(" Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    ts_loss, ts_probs, ts_preds, ts_labels = run_epoch(test_loader, train=False)
    ts_metrics = compute_metrics(ts_probs, ts_preds, ts_labels, NUM_CLASSES)

    elapsed = time.time() - t0
    print(f" Training finished in {elapsed:.1f}s")

    # Save experiment results
    results_all.append({
        "config": config,
        "dev_best_f1": best_dev,
        "test_metrics": ts_metrics
    })
    print(results_all)

# ==== RUN ALL EXPERIMENTS ====
for exp in experiments:
    train_and_evaluate(exp)

# ==== SAVE ALL RESULTS ====
with open("experiment_results.json", "w") as f:
    json.dump(results_all, f, indent=2)

print(" All experiments done. Results saved to experiment_results.json")


# print("\n=== TEST METRICS (TCN) ===")
# for k in ["accuracy","auc","precision_weighted","recall_weighted","f1_weighted",
#           "precision_macro","recall_macro","f1_macro","matthews_corrcoef","cohen_kappa"]:
#     print(f"{k:>20}: {ts_metrics[k]:.4f}" if isinstance(ts_metrics[k], float) else f"{k:>20}: {ts_metrics[k]}")
# print("\nConfusion Matrix:\n", np.array(ts_metrics["confusion_matrix"]))
# print("\nClassification Report:\n", ts_metrics["classification_report"])
