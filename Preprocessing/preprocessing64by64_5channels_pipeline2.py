import os
import numpy as np
import mne
from sklearn.manifold import TSNE
from sklearn.decomposition import FastICA
from scipy.interpolate import griddata
from scipy.signal import welch
from tqdm import tqdm

# --- Config ---
DATA_DIR = "./files"  # Directory with subject folders containing .edf files
GRID_SIZE = 64
FS = 160  # Hz
SEGMENT_SECONDS = 6
SAMPLES_PER_SEGMENT = FS * SEGMENT_SECONDS
EXCLUDED_SUBJECTS = {"S038", "S088", "S089", "S092", "S100", "S104"}

# 4-class (actual) mapping: Left, Right, Rest(Open eyes), Feet
TARGET_LABELS = {"T1": 0, "T2": 1, "T0": 2, "T3": 3}
OUTPUT_DIR = "./preprocessed_pipeline2_samples"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# Canonical bands (Hz)
BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta":  (12.0, 30.0),
    "gamma": (30.0, 45.0),
}

# Welch PSD params
N_PER_SEG = FS * 1      # 1-second window
N_OVERLAP = N_PER_SEG // 2
WINDOW = "hann"
DETREND = "constant"

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
    return sorted(edf_files)

def extract_eeg_and_events(edf_file):
    # Load
    raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
    raw.pick_types(eeg=True)

    # Reference + montage
    raw.set_montage('standard_1005')
    raw.set_eeg_reference('average', projection=True)

    # Notch + band-pass
    raw.notch_filter([50, 60], verbose=False)
    raw.filter(1., 45., fir_design='firwin', verbose=False)

    # Optional: ICA artifact attenuation (simple template)
    ica = mne.preprocessing.ICA(n_components=20, random_state=0, max_iter='auto')
    ica.fit(raw)
    raw = ica.apply(raw.copy())

    # Events + mapping
    events, mapping = mne.events_from_annotations(raw, verbose=False)
    # Build reverse map for string labels present in annotations
    inv_map = {v: k for k, v in mapping.items()}

    # Collect only events we care about
    wanted = []
    for onset, _, code in events:
        desc = inv_map.get(code, "")
        if desc in TARGET_LABELS:
            wanted.append((onset, TARGET_LABELS[desc]))

    data = raw.get_data()  # shape: (n_channels, n_samples)
    return data, wanted, raw.info

def compute_tsne_coordinates(info):
    pos_3d = np.array([info['chs'][i]['loc'][:3] for i in range(len(info['chs']))])
    # ICA -> SVD basis (deterministic-ish) for init
    ica = FastICA(n_components=3, random_state=0)
    components = ica.fit_transform(pos_3d)  # (n_ch, 3)
    tsne = TSNE(n_components=2, perplexity=5, init=components[:, :2], random_state=0)
    coords_2d = tsne.fit_transform(pos_3d)
    coords_2d -= coords_2d.min(axis=0)
    coords_2d /= coords_2d.max(axis=0)
    coords_2d *= (GRID_SIZE - 1)
    return coords_2d

def bandpower_welch(x, fs, fmin, fmax):
    # x: (T,) single-channel segment
    f, Pxx = welch(x, fs=fs, nperseg=N_PER_SEG, noverlap=N_OVERLAP,
                   window=WINDOW, detrend=DETREND, scaling='density')
    # Integrate power in [fmin, fmax]
    idx = (f >= fmin) & (f < fmax)
    return np.trapz(Pxx[idx], f[idx]) if np.any(idx) else 0.0

def compute_bandpowers_segment(segment):
    """
    segment: (n_channels, T=FS*6)
    returns: (5, n_channels) in order [delta, theta, alpha, beta, gamma]
    """
    bands_order = ["delta", "theta", "alpha", "beta", "gamma"]
    bp = np.zeros((len(bands_order), segment.shape[0]), dtype=float)
    for ch in range(segment.shape[0]):
        sig = segment[ch]
        for bi, bname in enumerate(bands_order):
            fmin, fmax = BANDS[bname]
            bp[bi, ch] = bandpower_welch(sig, FS, fmin, fmax)
    return bp  # (5, n_channels)

def generate_topomap(values_per_channel, coords_2d):
    """
    values_per_channel: (n_channels,) → map to 64x64 via nearest interpolation
    """
    H, W = GRID_SIZE, GRID_SIZE
    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
    points = coords_2d
    img = griddata(points, values_per_channel, (grid_x, grid_y),
                   method='nearest', fill_value=0.0)
    return img  # (H, W)

# --- Main ---
X, y = [], []
edf_files = get_edf_paths(DATA_DIR)
sample_idx = 0
for edf_file in tqdm(edf_files, desc="Processing subjects (Pipeline 2: 5-band topomaps)"):
    data, events, info = extract_eeg_and_events(edf_file)
    coords_2d = compute_tsne_coordinates(info)
    subj_id = os.path.basename(os.path.dirname(edf_file))

    for onset, label in events:
        # 6-second trial slice
        start = onset
        end = onset + SAMPLES_PER_SEGMENT
        if end > data.shape[1]:
            continue
        segment = data[:, start:end]  # (n_ch, 960)

        # Compute bandpower per channel for the whole 6s segment
        # returns (5, n_ch) ordered [δ, θ, α, β, γ]
        bandpowers = compute_bandpowers_segment(segment)

        # For each band, make one topomap (H, W), then stack → (5, H, W)
        band_maps = []
        for bi in range(bandpowers.shape[0]):
            ch_values = bandpowers[bi]  # (n_ch,)
            topomap = generate_topomap(ch_values, coords_2d)
            band_maps.append(topomap)
        sample = np.stack(band_maps, axis=0)  # (5, 64, 64)
        out_name = f"sample_{subj_id}_{os.path.splitext(os.path.basename(edf_file))[0]}_{sample_idx:05d}.npz"
        np.savez(os.path.join(OUTPUT_DIR, out_name), X=sample, y=label)
        X.append(sample)
        y.append(label)
        sample_idx += 1

X = np.stack(X) if len(X) else np.empty((0, 5, GRID_SIZE, GRID_SIZE))
y = np.array(y, dtype=int)

np.savez("preprocessed_eeg_64x64_5band_pipeline2.npz", X=X, y=y)
print("Saved: preprocessed_eeg_64x64_5band_pipeline2.npz  |  X shape:", X.shape, " y shape:", y.shape)
