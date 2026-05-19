# pengelompokan manusia vs tidak ada manusia
# membuat model .pkl

import numpy as np
import pandas as pd
from scipy import signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# =====================
# 1. LOAD DATA + RELABEL
# =====================
files = {
    "no_human": [
        "csi_empty_room.csv",
        "csi_object_motion_center.csv",
        "csi_object_motion_45deg.csv",
    ],
    "human": [
        "csi_human_idle_center.csv",
        "csi_human_idle_45deg.csv",
        "csi_human_motion_center.csv",
        "csi_human_motion_45deg.csv",
    ],
}

def load_all(files):
    dfs = []
    for label, paths in files.items():
        for path in paths:
            df = pd.read_csv(path)
            df["label"] = label
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

df = load_all(files)
print(f"Total sample: {len(df)}")
print(df["label"].value_counts())

# =====================
# 2. WINDOWING + FEATURE EXTRACTION
# =====================
WINDOW_SIZE = 50
STEP_SIZE   = 25
FS          = 10

def compute_amplitude(vals):
    return np.array([np.sqrt(vals[i]**2 + vals[i+1]**2)
                     for i in range(0, len(vals), 2)])

def extract_window_features(window_df):
    amps = []
    for _, row in window_df.iterrows():
        vals = row[[f"v{i}" for i in range(128)]].values.astype(float)
        amps.append(compute_amplitude(vals))
    amps = np.array(amps)

    features = []
    features += list(np.mean(amps, axis=0))
    features += list(np.std(amps, axis=0))
    features += list(np.var(amps, axis=0))
    features.append(np.mean(amps))
    features.append(np.std(amps))
    features.append(np.max(amps))
    features.append(np.min(amps))

    rssi = window_df["rssi"].values.astype(float)
    features.append(np.mean(rssi))
    features.append(np.std(rssi))
    features.append(np.max(rssi) - np.min(rssi))

    amp_signal = np.mean(amps, axis=1)
    fft_vals = np.abs(np.fft.rfft(amp_signal))
    fft_freqs = np.fft.rfftfreq(WINDOW_SIZE, d=1.0/FS)

    features += list(fft_vals)
    features.append(np.argmax(fft_vals) * FS / WINDOW_SIZE)
    features.append(np.max(fft_vals))

    breath_mask = (fft_freqs >= 0.1) & (fft_freqs <= 0.5)
    features.append(np.sum(fft_vals[breath_mask]))

    motion_mask = (fft_freqs >= 0.5) & (fft_freqs <= 2.0)
    features.append(np.sum(fft_vals[motion_mask]))

    return features

print("Extracting windowed features...")
X_all = []
y_all = []

for label, paths in files.items():
    for path in paths:
        df_label = pd.read_csv(path)
        for start in range(0, len(df_label) - WINDOW_SIZE, STEP_SIZE):
            window = df_label.iloc[start:start+WINDOW_SIZE]
            features = extract_window_features(window)
            X_all.append(features)
            y_all.append(label)

X = np.array(X_all)
y = np.array(y_all)

human_idx = np.where(y == "human")[0][0]
no_human_idx = np.where(y == "no_human")[0][0]
print(f"TRAINING human - mean: {X[human_idx][0]:.2f}, std: {X[human_idx][64]:.2f}, fft_peak: {X[human_idx][195]:.2f}")
print(f"TRAINING no_human - mean: {X[no_human_idx][0]:.2f}, std: {X[no_human_idx][64]:.2f}, fft_peak: {X[no_human_idx][195]:.2f}")

print(f"Total windows: {len(X)}")
print(f"Feature vector size: {X.shape[1]}")
print(pd.Series(y).value_counts())

# =====================
# 3. TRAIN MODEL
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("\nTraining Random Forest...")
clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# =====================
# 4. EVALUASI
# =====================
y_pred = clf.predict(X_test)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=["no_human", "human"])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["no_human", "human"],
            yticklabels=["no_human", "human"])
plt.title("Confusion Matrix - v3 (Binary: Human vs No Human)")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()


joblib.dump(clf, "csi_model.pkl")
print("Model tersimpan: csi_model.pkl")

plt.show()