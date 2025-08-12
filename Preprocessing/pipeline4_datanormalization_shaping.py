import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---- Config ----
DATA_DIR = './processed'  # Where your .npz files are saved
N_CHANNELS = 9               # Based on MOTOR_CHANNELS
N_FREQ = 12                  # From tfr_multitaper bins
N_TIME = 32                  # Time steps

# ---- Step 1: Load all preprocessed files ----
file_paths = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR)
             if f.endswith('.npz') and not f.endswith('.event')]


X_list, y_list = [], []
for path in file_paths:
    data = np.load(path)
    X_list.append(data['X'])
    y_list.append(data['y'])


# Stack and reshape for ViT (n_samples, ch×freq, time)
X = np.vstack(X_list)  # Shape: (n_samples, N_CHANNELS*N_FREQ, N_TIME)
y = np.concatenate(y_list)

print(f"X shape: {X.shape}")
print(f"Expected: (n_samples, {N_CHANNELS*N_FREQ}, {N_TIME})")

# ---- Step 2: Train/Dev/Test Split (Subject-Aware) ----
# Assuming files follow S001R01.npz naming
subject_ids = [int(f.split('S')[1].split('R')[0]) for f in os.listdir(DATA_DIR)
              if f.startswith('S')]
unique_subjects = np.unique(subject_ids)

# Stratified split (70/15/15)
train_subj, test_subj = train_test_split(unique_subjects, test_size=0.3, random_state=42)
dev_subj, test_subj = train_test_split(test_subj, test_size=0.5, random_state=42)

# Get indices for each split
train_idx = [i for i, subj in enumerate(subject_ids) if subj in train_subj]
dev_idx = [i for i, subj in enumerate(subject_ids) if subj in dev_subj]
test_idx = [i for i, subj in enumerate(subject_ids) if subj in test_subj]

# ---- Step 3: Normalize per frequency band ----
scaler = StandardScaler()
# Reshape to (samples×time, ch×freq) for scaling
X_scaled = scaler.fit_transform(X.transpose(0,2,1).reshape(-1, N_CHANNELS*N_FREQ))
X_scaled = X_scaled.reshape(X.shape[0], N_TIME, N_CHANNELS*N_FREQ).transpose(0,2,1)

# ---- Step 4: Convert to ViT Input ----
# Add channel dim (ViT expects [batch, ch, h, w] - we use ch=1)
X_vit = torch.tensor(X_scaled[:, None], dtype=torch.float32)  # Shape: [n, 1, ch×freq, time]

# ---- Step 5: Final Split ----
X_train, y_train = X_vit[train_idx], torch.tensor(y[train_idx], dtype=torch.long)
X_dev, y_dev = X_vit[dev_idx], torch.tensor(y[dev_idx], dtype=torch.long)
X_test, y_test = X_vit[test_idx], torch.tensor(y[test_idx], dtype=torch.long)

print("🎯 Final Shapes:")
print(f"Train: {X_train.shape} (targets: {y_train.shape})")
print(f"Dev: {X_dev.shape} (targets: {y_dev.shape})")
print(f"Test: {X_test.shape} (targets: {y_test.shape})")

# ---- Save ----
torch.save({'X': X_train, 'y': y_train}, 'eeg_vit_train.pt')
torch.save({'X': X_dev, 'y': y_dev}, 'eeg_vit_dev.pt')
torch.save({'X': X_test, 'y': y_test}, 'eeg_vit_test.pt')