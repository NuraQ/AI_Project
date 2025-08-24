## Dataset: PhysioNet EEG Motor Movement/Imagery (EEGMMIBD)

The PhysioNet **EEG Motor Movement/Imagery** database (EEGMMIBD) was used: 109 volunteers, 64-channel EEG (10–10 system) recorded with BCI2000 at 160 Hz. Each subject performed 14 runs: two 1-minute baselines (eyes open/closed) and three 2-minute runs for each of four task families covering executed and imagined limb movements (e.g., open/close fists or feet, and their imagined counterparts)

Multi-class cassification is testes on EEG signals using both deep-learning and transformer models. In addition to that, Full preprocessing pipelines implemented frpm scratch to evaluate performance on several models.

## What’s here

- **Data loaders** for per-sample `.npz` chunks with subject-aware splitting.
- **Pipelines**
  - ViT on **5 bands** (64×64 → 224×224) with adapted 5-channel patch embed.
  - ViT on **3 bands** (64×64 → 224×224) using the native 3-channel ViT.
  - ResNet-18 on **3 bands** (64×64 → 224×224).
  - ViT on **8×8** topomaps (binary), channel-collapsed to 3 for pretrained backbones.
  - CNN-BiLSTM for temporal stacks (e.g., 60 frames of 64×64).
  - A lightweight 2-CNN baseline.
  - Perceiver
  - EEGNetV4
  - TCN

- **Subject-independent evaluation**
  - Deterministic splits by subject ID derived from filenames.
- **Experiment sweeps**
  - Learning rate, weight decay, batch size, label smoothing, augmentation, weighted sampling, freezing, grad accumulation.
- **Metrics**
  - Accuracy, weighted/macro F1, AUC (OvR), precision/recall (macro & weighted), MCC, Cohen’s κ, confusion matrix, classification report.

---

## Expected dataset format

Per-sample `.npz` files stored in a folder (It wa stored on google collab). Typical contents:

- `X`: EEG topomap array
  - **5-band 64×64**: `(C, H, W)` with `C >= 5` (delta, theta, alpha, beta, gamma) => scripts keep the first 5.
  - **3-band 64×64**: `(C, H, W)` where code selects band indices (commonly theta/alpha/beta) before normalization.
  - **8×8**: `(5, 8, 8)`; some pipelines average across bands to `(1, 8, 8)` then replicate to 3 channels.
- `y`: integer class label per sample.


## Subject-independent splits (deterministic)

Subjects are parsed from file names (e.g., `sample_S001R01_chunk00.npz`  => `S001`). Splits are fixed via list slicing—no randomness:

```python
def get_subject_ids(DATA_DIR):
    ids = set()
    for fname in os.listdir(DATA_DIR):
        if fname.endswith(".npz") and fname.startswith("sample"):
            sid = fname.split("_")[1][:4]  # 'S001'
            ids.add(sid)
    return sorted(ids)

def _slice(lst, a, b):
    a = min(a, len(lst)); b = min(b, len(lst))
    return lst[a:b]

subject_ids = get_subject_ids(DATA_DIR)
train_ids = _slice(subject_ids, 0, 70)
dev_ids   = _slice(subject_ids, 71, 86)
test_ids  = _slice(subject_ids, 87, 103)
