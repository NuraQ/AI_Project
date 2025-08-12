import numpy as np
import os
import mne
from babel.util import missing
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA  # <--- Add this import
from scipy.interpolate import griddata
from mne.time_frequency import psd_array_welch
from sklearn.decomposition import FastICA
from tqdm import tqdm

# --- Config ---
DATA_DIR = "./files"  # Directory with subject folders containing .edf files
GRID_SIZE = 64
FS = 160  # Hz
# SEGMENT_SECONDS = 6
# SAMPLES_PER_SEGMENT = FS * SEGMENT_SECONDS
SEGMENT_SECONDS = 1
SAMPLES_PER_SEGMENT = FS * SEGMENT_SECONDS

FRAMES_PER_SEGMENT = 60
SAMPLES_PER_FRAME = SAMPLES_PER_SEGMENT // FRAMES_PER_SEGMENT
NOISE_STD = 1e-4
NORMALIZE = False  # <--- Set this to False to disable normalization
EXCLUDED_SUBJECTS = {"S038", "S088", "S089", "S092", "S100", "S104"}
TARGET_LABELS = {"T1": 0, "T2": 1, "T0": 2, "T3": 3}  # L, R, O, F
SAVE_DIR = "./1_segment"
os.makedirs(SAVE_DIR, exist_ok=True)

def plot_topomap_sequence(sample_array, n_frames=5):
    n_total_frames = sample_array.shape[0]
    n_frames = min(n_frames, n_total_frames)

    fig, axes = plt.subplots(1, n_frames, figsize=(15, 3))
    if n_frames == 1:
        axes = [axes]  # Make it iterable if only 1 frame

    for i in range(n_frames):
        frame_idx = i * (n_total_frames // n_frames)
        axes[i].imshow(sample_array[frame_idx, 0], cmap='viridis')  # first band
        axes[i].set_title(f"Frame {frame_idx}")
        axes[i].axis('off')

    plt.suptitle("Topological Map Sequence (First Band)")
    plt.tight_layout()
    plt.show()



# def plot_topomap_sequence(sample_array, n_frames=5):
#     fig, axes = plt.subplots(1, n_frames, figsize=(15, 3))
#     for i in range(n_frames):
#         frame_idx = i * (FRAMES_PER_SEGMENT // n_frames)  # Evenly sample frames
#         axes[i].imshow(sample_array[frame_idx, 0], cmap='viridis')  # shape: (1, 64, 64)
#         axes[i].set_title(f"Frame {frame_idx}")
#         axes[i].axis('off')
#     plt.suptitle("Topological Map Sequence (Amplitude)")
#     plt.show()

def plot_tsne_projection(coords_2d, ch_names):
    plt.figure(figsize=(8, 6))
    plt.scatter(coords_2d[:, 0], coords_2d[:, 1], alpha=0.5)
    for i, ch in enumerate(ch_names):
        plt.text(coords_2d[i, 0], coords_2d[i, 1], ch, fontsize=8)
    plt.title("t-SNE Electrode Projection")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()

def plot_label_distribution(y):
    labels, counts = np.unique(y, return_counts=True)
    plt.bar(labels, counts, tick_label=["O", "L", "R", "F"])
    plt.title("Class Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.show()

def plot_raw_vs_quantized(segment, sample_array, ch_idx=0):
    plt.figure(figsize=(12, 4))

    # Raw EEG for one channel
    plt.subplot(1, 2, 1)
    plt.plot(segment[ch_idx], alpha=0.7)
    plt.title(f"Raw EEG (Channel {info['ch_names'][ch_idx]})")
    plt.xlabel("Samples")

    # Quantized frames (mean amplitude per frame)
    plt.subplot(1, 2, 2)
    frame_means = [frame[ch_idx].mean() for frame in sample_array[:, 0]]
    plt.plot(frame_means, 'o-')
    plt.title("Quantized Frame Means")
    plt.xlabel("Frame #")

    plt.tight_layout()
    plt.show()

def plot_label_distribution(y):
    labels, counts = np.unique(y, return_counts=True)
    plt.bar(labels, counts, tick_label=["O", "L", "R", "F"])
    plt.title("Class Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.show()

def get_label(run, annot):
    run = int(run)

    # Eyes open/closed
    if run == 1:
        return 0  # open eyes
    elif run == 2:
        return 1  # closed eyes

    if annot == 'T0':
        return 10  # rest

    elif annot == 'T1':
        if run in [3, 4]:
            return 2  # left hand
        elif run in [7, 8]:
            return 4  # imagined left hand
        elif run in [5, 6]:
            return 6  # both fists
        elif run in [9, 10]:
            return 8  # imagined fists
        elif run in [11, 12]:
            return 2  # left hand
        elif run in [13, 14]:
            return 6  # both fists

    elif annot == 'T2':
        if run in [3, 4]:
            return 3  # right hand
        elif run in [7, 8]:
            return 5  # imagined right hand
        elif run in [5, 6]:
            return 7  # both feet
        elif run in [9, 10]:
            return 9  # imagined feet
        elif run in [11, 12]:
            return 3  # right hand
        elif run in [13, 14]:
            return 7  # both feet

    return None


def apply_montage(raw):
    montage = mne.channels.make_standard_montage('standard_1005')
    raw.rename_channels({ch: ch.replace('.', '').upper() for ch in raw.ch_names})
    rename_dict = {
        "CZ": "Cz", "FP1": "Fp1", "FPZ": "Fpz", "FP2": "Fp2",
        "FZ": "Fz", "PZ": "Pz", "OZ": "Oz", "IZ": "Iz",
        "FCZ": "FCz", "CPZ": "CPz", "AFZ": "AFz", "POZ": "POz"
    }
    raw.rename_channels(rename_dict)
    raw.set_montage(montage)
    return raw
# --- Functions ---
def get_edf_paths(root_dir, include_subjects=None):
    edf_files = []
    for subj in os.listdir(root_dir):
        if subj in EXCLUDED_SUBJECTS:
            continue
        if include_subjects is not None and subj not in include_subjects:
            continue
        subj_path = os.path.join(root_dir, subj)
        if os.path.isdir(subj_path):
            for file in os.listdir(subj_path):
                if file.endswith(".edf"):
                    edf_files.append(os.path.join(subj_path, file))
    return edf_files


def extract_eeg_and_labels(edf_file):
    raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
    # Add this right after raw = ...
    print(f"Annotations in {edf_file}:")
    print(set([ann['description'] for ann in raw.annotations]))

    raw = apply_montage(raw)
    print("Channel names after montage:", raw.info['ch_names'])

    raw.set_eeg_reference('average', projection=True)
    raw.apply_proj()
    raw.pick(picks='eeg')  # Modern equivalent
    annotations = raw.annotations
    data = raw.get_data()
    # Extract run number from EDF filename
    filename = os.path.basename(edf_file)
    run_str = filename[-6:-4]
    try:
        run_number = int(run_str)
    except ValueError:
        raise ValueError(f"Could not parse run number from filename: {filename}")

    # Extract labeled segments
    events = []
    print(f"Annotations in {edf_file}:")
    print(set([ann['description'] for ann in raw.annotations]))

    for ann in annotations:
        onset_sample = int(ann['onset'] * 160)
        desc = ann['description']
        label = get_label(run_number, desc)


            # Add this to DEBUG:
        # print(f"🧠 Run {run_number}, Annotation: {desc} → Label: {label}")


        if label is not None:
            events.append((onset_sample, label))

        print(f"\n🔍 Annotations in {edf_file} (Run {run_number}):")
        for ann in annotations:
            print(f" - At {ann['onset']:.2f}s → {ann['description']}")

        # print("\n🧠 Events extracted:")
        for ann in annotations:
            onset_sample = int(ann['onset'] * 160)
            desc = ann['description']
            label = get_label(run_number, desc)
            # print(f"  {desc} → Label: {label}")

            if label is not None:
                events.append((onset_sample, label))

    return data, events, raw.info


def compute_tsne_coordinates(info):
    # Get 3D electrode positions
    pos_3d = np.array([info['chs'][i]['loc'][:3] for i in range(len(info['chs']))])

    # Initialize with PCA (full SVD as in paper)
    pca = PCA(n_components=2, random_state=0)
    components = pca.fit_transform(pos_3d)

    # t-SNE projection (perplexity=5 matches paper)
    tsne = TSNE(n_components=2, perplexity=5, init=components,
                random_state=0)  # Note: init=components (no slicing needed)
    coords_2d = tsne.fit_transform(pos_3d)

    # Scale to 0-(GRID_SIZE-1) range
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
# missing = ['S002', 'S015', 'S026', 'S055', 'S061', 'S081', 'S086', 'S101', 'S106', 'S107', 'S108', 'S109']


edf_files = get_edf_paths(DATA_DIR)
sample_id = 0

for edf_file in tqdm(edf_files, desc="Processing subjects"):
    data, events, info = extract_eeg_and_labels(edf_file)
    coords_2d = compute_tsne_coordinates(info)

    if not hasattr(plot_tsne_projection, 'already_plotted'):
        plot_tsne_projection(coords_2d, info['ch_names'])
        plot_tsne_projection.already_plotted = True

    # Update your config
    BANDS = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 12),
        'beta': (12, 30),
        'gamma': (30, 45)
    }

    # Sort events to get next-onset bounds
    events = sorted(events, key=lambda x: x[0])
    num_events = len(events)

    for idx, (onset, label) in enumerate(events):
        trial_end = events[idx + 1][0] if idx + 1 < num_events else data.shape[1]
        trial_data = data[:, onset:trial_end]
        total_samples = trial_data.shape[1]

        # if total_samples < FS:
        #     print(f"⚠️ Skipping short trial: only {total_samples} samples")
        #     continue

        n_chunks = total_samples // FS  # 1 second = 160 samples

        for chunk_idx in range(n_chunks):
            start = chunk_idx * FS
            end = start + FS
            frame_data = trial_data[:, start:end]

            frame_data += np.random.normal(0, NOISE_STD, size=frame_data.shape)

            if NORMALIZE:
                mean = np.mean(frame_data, axis=1, keepdims=True)
                std = np.std(frame_data, axis=1, keepdims=True) + 1e-6
                frame_data = (frame_data - mean) / std

            # psd, freqs = psd_array_welch(
            #     frame_data,  # shape: (n_channels, 160)
            #     sfreq=FS,
            #     fmin=1, fmax=45,
            #     n_fft=256,
            #     n_per_seg=160,
            #     n_overlap=0,
            #     average='mean',
            #     verbose=False
            # )
            psd, freqs = psd_array_welch(
                frame_data,  # (64, 160)
                sfreq=160,
                fmin=1, fmax=45,
                n_fft=160,  # Match segment length
                n_per_seg=160,
                n_overlap=0,
                average='mean',
                verbose=False
            )
            # print(f"\nPSD Verification for chunk {chunk_idx}:")
            # print(f"Input frame shape: {frame_data.shape} (should be (n_channels, 160))")
            # print(f"PSD output shape: {psd.shape} (should be (n_channels, n_freqs))")
            # print(f"Frequency bins: {freqs.shape} (should be (n_freqs,))")

            band_maps = []
            for band_name, (fmin, fmax) in BANDS.items():
                band_mask = (freqs >= fmin) & (freqs <= fmax)
                # band_power = psd[:, band_mask].mean(axis=1)
                band_power = np.log(psd[:, band_mask].mean(axis=1))  # Natural log scaling

                # print(f"\n{band_name} band ({fmin}-{fmax}Hz):")
                # print(f"Selected {band_mask.sum()} frequency bins")
                # print(f"Band power shape: {band_power.shape} (should be (n_channels,))")
                # print(f"Min power: {band_power.min():.2f}, Max: {band_power.max():.2f}")

                band_map = generate_topomap(band_power, coords_2d)
                band_maps.append(band_map)

                if chunk_idx == 0 and not hasattr(plot_topomap_sequence, f'band_plotted_{band_name}'):
                    plt.figure()
                    plt.imshow(band_map, cmap='viridis')
                    plt.title(f"{band_name} band topomap (first chunk)")
                    plt.colorbar()
                    plt.show()
                    setattr(plot_topomap_sequence, f'band_plotted_{band_name}', True)

            frame_5channel = np.stack(band_maps)  # shape: (5, 64, 64)

            # Optional: visualize one full band
            if sample_id == 0 and not hasattr(plot_topomap_sequence, 'already_plotted'):
                plot_topomap_sequence(np.expand_dims(frame_5channel, axis=0))
                plot_topomap_sequence.already_plotted = True

            X.append(frame_5channel)
            y.append(label)
            sample_name = f"sample_{os.path.basename(edf_file).replace('.edf', '')}_chunk{chunk_idx:02d}.npz"
            np.savez_compressed(os.path.join(SAVE_DIR, sample_name), X=frame_5channel, y=label)
            sample_id += 1

X = np.stack(X)
y = np.array(y)

# Save combined output
np.savez("preprocessed_eeg_64x64.npz", X=X, y=y)
print("✅ Preprocessing complete. Saved to preprocessed_eeg_64x64.npz and individual files in ./preprocessed_per_sample/")


