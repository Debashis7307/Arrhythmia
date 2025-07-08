# 🫀 Arrhythmia Detection Using Explainable Machine Learning

> A signal-processing and machine learning pipeline for ECG-based arrhythmia classification using traditional interpretable models.

---

## 🧠 What is Arrhythmia?

**Arrhythmia** refers to an abnormal heart rhythm caused by irregularities in the heart's electrical impulses. These irregularities can lead to slow (bradycardia), fast (tachycardia), or erratic heartbeats, potentially causing dizziness, shortness of breath, stroke, or even sudden cardiac death.

---

## ❗ Why Detect It Early?

Timely and accurate arrhythmia detection:
- Helps prevent life-threatening cardiac events
- Reduces risk of stroke or heart failure
- Allows real-time diagnosis in clinical and wearable health applications

However, manual analysis of ECG signals is time-consuming and error-prone. That’s where machine learning steps in.

---

## ⚙️ How This Project Works

This repository implements a hybrid **ECG beat segmentation and machine learning classification pipeline** to detect arrhythmia types using a dataset of ECG signals (e.g. MIT-BIH). It uses **interpretable models** like K-Nearest Neighbors for explainability and low complexity.

---

## 🧪 Model Architecture

The pipeline follows a structured flow starting from raw ECG signal to final arrhythmia classification. It includes preprocessing, heuristic labeling, beat segmentation, and machine learning.

<p align="center">
  <img src="model.png" alt="Model Architecture Diagram" width="700"/>
</p>


## 🛠 Features

- 🔍 **R-Peak Detection**: Uses `scipy.signal.find_peaks` to detect QRS complexes based on inter-peak distance (approx. 0.6s).
- 📐 **Beat Segmentation**: Extracts 40-sample windows centered around each R-peak.
- 🧠 **Heuristic Beat Labeling**:
  - QRS width estimation via derivative thresholding
  - RR interval calculation
  - Labeled as one of: `NOR`, `PVC`, `APC`, `LBB`, or `RBB`
- 🗃️ **Beat Dataset Generation**: Saves a `labels.csv` mapping filenames to their majority beat class.
- 📊 **Feature Scaling**: StandardScaler normalizes beat data before training.
- 🤖 **Classification Model**: K-Nearest Neighbors (KNN) classifier trained on ECG beats.
- 📈 **Evaluation**:
  - Classification report (precision, recall, F1-score)
  - Confusion matrix heatmap using `seaborn`

---

## 🧪 Example Results

Once the model is trained, you’ll see outputs like:


