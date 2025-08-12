import os
import numpy as np
import mne
from mne.time_frequency import tfr_multitaper

# ---- Parameters ----
DATA_ROOT = './files'  # Folder with S001, S002, etc.
OUTPUT_DIR = './processed_Motor_channels'  # Output folder
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_SFREQ = 160
MOTOR_CHANNELS = ['C3', 'Cz', 'C4', 'FC1', 'FC2', 'CP1', 'CP2', 'P3', 'P4']
FREQ_RANGE = (4, 40)
N_FREQ_BINS = 12

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

def get_label(run_num, annotation):
    run_num = int(run_num)
    if run_num == 1: return 0  # Eyes open
    if run_num == 2: return 1  # Eyes closed
    if 3 <= run_num <= 6 or 11 <= run_num <= 14:
        if annotation == 'T1':
            return 2 if run_num in [3, 4, 11, 12] else 4
        elif annotation == 'T2':
            return 3 if run_num in [3, 4, 11, 12] else 5
    elif 7 <= run_num <= 10:
        if annotation == 'T1':
            return 6 if run_num in [7, 8] else 8
        elif annotation == 'T2':
            return 7 if run_num in [7, 8] else 9
    raise ValueError(f"Unknown run_num={run_num}, annotation={annotation}")

def process_edf_file(edf_path):
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True)
        raw = apply_montage(raw)
        raw.pick_channels(MOTOR_CHANNELS)
        raw.set_eeg_reference('average')
        raw.notch_filter([50, 60], fir_design='firwin', pad='reflect')
        raw.filter(4., 40., fir_design='firwin', phase='zero-double')
        raw.resample(TARGET_SFREQ)

        # Define events for Epochs
        event_id = {'T1': 1, 'T2': 2}
        events, _ = mne.events_from_annotations(raw, event_id=event_id)
        if not events.size:
            return None

        # Extract valid epochs only
        epochs = mne.Epochs(raw, events, event_id=event_id,
                            tmin=0.0, tmax=3.0, baseline=None, preload=True)

        if len(epochs) == 0:
            print(f" No valid epochs in {edf_path}")
            return None

        # Compute time-frequency representation
        tfr = tfr_multitaper(
            epochs,
            freqs=np.linspace(*FREQ_RANGE, N_FREQ_BINS),
            n_cycles=4,
            time_bandwidth=4.0,
            return_itc=False
        )
        power = np.log1p(tfr.data)  # Shape: (n_epochs, n_chans, n_freq, n_time)

        print("🔍 Power stats — min:", power.min(), "max:", power.max(), "mean:", power.mean())

        # Label mapping (guaranteed 1:1 with epochs now)
        run_num = int(os.path.basename(edf_path).split('R')[-1].split('.')[0])
        labels = []
        for code in epochs.events[:, 2]:
            label_desc = [k for k, v in event_id.items() if v == code][0]  # 'T1' or 'T2'
            labels.append(get_label(run_num, label_desc))

        # Save
        output_name = os.path.join(OUTPUT_DIR, os.path.basename(edf_path).replace('.edf', '.npz'))
        np.savez(output_name, X=power, y=np.array(labels))
        print(f"✅ Saved {output_name}")
        return output_name

    except Exception as e:
        print(f" Failed {os.path.basename(edf_path)}: {str(e)}")
        return None

# ---- Main Execution ----
if __name__ == "__main__":
    processed_files = []
    count = 0
    for subject in sorted(os.listdir(DATA_ROOT)):
        subject_dir = os.path.join(DATA_ROOT, subject)
        if not os.path.isdir(subject_dir):
            continue
        count += 1
        if count == 2:
            break
        for file in sorted(f for f in os.listdir(subject_dir) if f.endswith('.edf')):
            edf_path = os.path.join(subject_dir, file)
            result = process_edf_file(edf_path)
            if result:
                processed_files.append(result)
                print(f"Processed: {file}")
            else:
                print(f"Skipped (error): {file}")
    print(f"\n Successfully processed {len(processed_files)} files")
