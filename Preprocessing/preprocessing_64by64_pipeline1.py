import os
import numpy as np
import mne
from sklearn.manifold import TSNE
from sklearn.decomposition import FastICA
from scipy.interpolate import griddata
from tqdm import tqdm

# --- Config ---
DATA_DIR = "./files"  # Directory with subject folders containing .edf files
GRID_SIZE = 64
FS = 160  # Hz
SEGMENT_SECONDS = 6
SAMPLES_PER_SEGMENT = FS * SEGMENT_SECONDS
FRAMES_PER_SEGMENT = 60
SAMPLES_PER_FRAME = SAMPLES_PER_SEGMENT // FRAMES_PER_SEGMENT
NOISE_STD = 1e-4
EXCLUDED_SUBJECTS = {"S038", "S088", "S089", "S092", "S100", "S104"}
TARGET_LABELS = {"T1": 0, "T2": 1, "T0": 2, "T3": 3}  # L, R, O, F

# --- Functions ---
def get_edf_paths(root_dir):
    edf_files = []
    for subj in os.listdir(root_dir):
        if subj in EXCLUDED_SUBJECTS:
            continue
        subj_path = os.path.join(root_dir, subj)
        if os.path.isdir(subj_path):
            for file in os.listdir(subj_path):
                if file.endswith(".edf"):
                    edf_files.append(os.path.join(subj_path, file))
    return edf_files

def extract_eeg_and_labels(edf_file):
    raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
    raw.set_eeg_reference('average', projection=True)
    raw.pick_types(eeg=True)
    annotations = raw.annotations
    data = raw.get_data()
    events = [(int(ann['onset'] * FS), TARGET_LABELS[ann['description']])
              for ann in annotations if ann['description'] in TARGET_LABELS]
    return data, events, raw.info

def compute_tsne_coordinates(info):
    pos_3d = np.array([info['chs'][i]['loc'][:3] for i in range(len(info['chs']))])
    ica = FastICA(n_components=3, random_state=0)
    components = ica.fit_transform(pos_3d)
    tsne = TSNE(n_components=2, perplexity=5, init=components[:, :2], random_state=0)
    coords_2d = tsne.fit_transform(pos_3d)
    coords_2d -= coords_2d.min(axis=0)
    coords_2d /= coords_2d.max(axis=0)
    coords_2d *= (GRID_SIZE - 1)
    return coords_2d

def generate_topomap(frame, coords_2d):
    H, W = GRID_SIZE, GRID_SIZE
    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
    points = coords_2d
    values = frame
    img = griddata(points, values, (grid_x, grid_y), method='nearest', fill_value=0)
    return img

# --- Main Processing ---
X, y = [], []
edf_files = get_edf_paths(DATA_DIR)

for edf_file in tqdm(edf_files, desc="Processing subjects"):
    data, events, info = extract_eeg_and_labels(edf_file)
    coords_2d = compute_tsne_coordinates(info)
    for onset, label in events:
        if onset + SAMPLES_PER_SEGMENT > data.shape[1]:
            continue
        segment = data[:, onset:onset + SAMPLES_PER_SEGMENT]
        segment += np.random.normal(0, NOISE_STD, size=segment.shape)  # Gaussian noise
        frames = []
        for i in range(FRAMES_PER_SEGMENT):
            start = i * SAMPLES_PER_FRAME
            end = (i + 1) * SAMPLES_PER_FRAME
            frame = segment[:, start:end].mean(axis=1)
            topomap = generate_topomap(frame, coords_2d)
            frames.append(topomap[np.newaxis, :, :])  # Add channel dim
        X.append(np.stack(frames))  # shape: (60, 1, 64, 64)
        y.append(label)

X = np.stack(X)
y = np.array(y)

# Save output
np.savez("preprocessed_eeg_64x64.npz", X=X, y=y)
print(" Preprocessing complete. Saved to preprocessed_eeg_64x64.npz")
