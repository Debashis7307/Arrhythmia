import os
import pandas as pd
import numpy as np
from glob import glob
from scipy.signal import find_peaks
from collections import Counter

# Sampling frequency and window size
FS = 360  # Hz
BEAT_HALF_WINDOW = 20  # 20 samples before and after R peak
BEAT_WINDOW = BEAT_HALF_WINDOW * 2

# Estimate QRS width using derivative thresholding
def estimate_qrs_width(beat):
    deriv = np.diff(beat)
    threshold = np.max(np.abs(deriv)) * 0.5
    above_thresh = np.where(np.abs(deriv) > threshold)[0]
    if len(above_thresh) < 2:
        return 0
    return above_thresh[-1] - above_thresh[0]

# Heuristic beat classification
def classify_beat(qrs_width, rr_interval):
    if qrs_width > 40:          # Very wide QRS
        return "LBB" if np.random.rand() > 0.5 else "RBB"
    elif rr_interval < 0.6 * FS:  # Premature beat
        return "APC"
    elif qrs_width > 30:         # Moderately wide QRS
        return "PVC"
    else:
        return "NOR"

# Pick the best lead available in a file
def pick_lead(df):
    for col in ['MLII', 'V1', 'V2', 'V5']:
        if col in df.columns:
            return col
    return None

# Process a single CSV file and return the most common beat label
def process_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        lead = pick_lead(df)
        if lead is None:
            return None

        signal = df[lead].fillna(0).values
        peaks, _ = find_peaks(signal, distance=int(0.6 * FS))

        beat_labels = []
        for i in range(1, len(peaks) - 1):
            idx = peaks[i]
            rr_interval = idx - peaks[i - 1]

            if idx - BEAT_HALF_WINDOW < 0 or idx + BEAT_HALF_WINDOW >= len(signal):
                continue

            beat = signal[idx - BEAT_HALF_WINDOW:idx + BEAT_HALF_WINDOW]
            if len(beat) != BEAT_WINDOW:
                continue

            qrs_width = estimate_qrs_width(beat)
            label = classify_beat(qrs_width, rr_interval)
            beat_labels.append(label)

        if beat_labels:
            file_label = Counter(beat_labels).most_common(1)[0][0]
            return os.path.basename(file_path), file_label
        else:
            return None

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Main function to process all CSVs in a directory
def approximate_all_labels(csv_dir):
    all_files = sorted(glob(os.path.join(csv_dir, "*.csv")))
    label_data = []

    for file in all_files:
        result = process_csv(file)
        if result:
            label_data.append(result)

    df_labels = pd.DataFrame(label_data, columns=["filename", "label"])
    return df_labels

# Set your local dataset path here
csv_dir = "dataset"  # Replace with actual path

# Run label approximation
label_df = approximate_all_labels(csv_dir)
label_df.to_csv("labels.csv", index=False)
label_df.head()
