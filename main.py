import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


FS = 360
BEAT_HALF_WINDOW = 20
BEAT_WINDOW = 2 * BEAT_HALF_WINDOW
LEADS = ['MLII', 'V1', 'V2', 'V5']
LABELS_CSV_PATH = "labels.csv"
CSV_DIR = "dataset"


def load_labels(path):
    df = pd.read_csv(path)
    return dict(zip(df["filename"], df["label"]))

def pick_lead(df):
    for lead in LEADS:
        if lead in df.columns:
            return lead
    return None

def extract_beats(signal):
    beats = []
    peaks, _ = find_peaks(signal, distance=int(0.6 * FS))
    for i in range(1, len(peaks) - 1):
        idx = peaks[i]
        start = idx - BEAT_HALF_WINDOW
        end = idx + BEAT_HALF_WINDOW
        if start >= 0 and end < len(signal):
            beat = signal[start:end]
            if len(beat) == BEAT_WINDOW:
                beats.append(beat)
    return beats


def load_dataset(data_dir, label_map):
    X, y = [], []
    for file_name in os.listdir(data_dir):
        if not file_name.endswith(".csv") or file_name not in label_map:
            continue
        try:
            df = pd.read_csv(os.path.join(data_dir, file_name))
            lead = pick_lead(df)
            if not lead:
                continue
            signal = df[lead].fillna(0).values
            beats = extract_beats(signal)
            X.extend(beats)
            y.extend([label_map[file_name]] * len(beats))
        except Exception as e:
            print(f"[Error] {file_name}: {e}")
    return np.array(X), np.array(y)


def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


def main():
    label_map = load_labels(LABELS_CSV_PATH)
    X, y = load_dataset(CSV_DIR, label_map)

    if len(X) == 0:
        print("No valid beats found.")
        return

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n📋 Classification Report:\n")
    print(classification_report(y_test, y_pred))

    plot_confusion_matrix(y_test, y_pred, sorted(np.unique(y)))


if __name__ == "__main__":
    main()
