# proses CSI per window 50 sampel

import numpy as np
import pandas as pd
from scipy import signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# =====================
# 1. LOAD DATA
# =====================
files = {
    "empty_room":           "csi_empty_room.csv",
    "object_motion_center": "csi_object_motion_center.csv",
    "object_motion_45deg":  "csi_object_motion_45deg.csv",
    "human_idle_center":    "csi_human_idle_center.csv",
    "human_idle_45deg":     "csi_human_idle_45deg.csv",
    "human_motion_center":  "csi_human_motion_center.csv",
    "human_motion_45deg":   "csi_human_motion_45deg.csv",
}

def load_all(files):
    dfs = []
    for label, path in files.items():
        df = pd.read_csv(path)
        df["label"] = label
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

df = load_all(files)
print(f"Total sample: {len(df)}")

# =====================
# 2. WINDOWING + FEATURE EXTRACTION
# =====================
WINDOW_SIZE = 50   # 50 sample per window = 5 detik data
STEP_SIZE   = 25   # overlap 50%
FS          = 10   # sampling rate ~10 Hz (ping 100ms)

def compute_amplitude(vals):
    return np.array([np.sqrt(vals[i]**2 + vals[i+1]**2)
                     for i in range(0, len(vals), 2)])

def extract_window_features(window_df):
    # Ambil amplitude semua subcarrier per sample
    amps = []
    for _, row in window_df.iterrows():
        vals = row[[f"v{i}" for i in range(128)]].values.astype(float)
        amps.append(compute_amplitude(vals))
    amps = np.array(amps)  # shape: (WINDOW_SIZE, 64)

    features = []

    # --- Fitur statistik per subcarrier ---
    features += list(np.mean(amps, axis=0))    # mean per subcarrier
    features += list(np.std(amps, axis=0))     # std per subcarrier
    features += list(np.var(amps, axis=0))     # variance per subcarrier

    # --- Fitur statistik global ---
    mean_amp = np.mean(amps)
    features.append(mean_amp)
    features.append(np.std(amps))
    features.append(np.max(amps))
    features.append(np.min(amps))

    # --- Fitur RSSI ---
    rssi = window_df["rssi"].values.astype(float)
    features.append(np.mean(rssi))
    features.append(np.std(rssi))
    features.append(np.max(rssi) - np.min(rssi))  # RSSI range

    # --- Fitur FFT (deteksi periodiitas gerakan & pernapasan) ---
    # Gunakan mean amplitude across subcarrier sebagai sinyal 1D
    amp_signal = np.mean(amps, axis=1)  # shape: (WINDOW_SIZE,)
    fft_vals = np.abs(np.fft.rfft(amp_signal))
    fft_freqs = np.fft.rfftfreq(WINDOW_SIZE, d=1.0/FS)

    features += list(fft_vals)          # FFT magnitude
    features.append(np.argmax(fft_vals) * FS / WINDOW_SIZE)  # dominant frequency
    features.append(np.max(fft_vals))   # peak FFT magnitude

    # Energy di band pernapasan (0.1 - 0.5 Hz)
    breath_mask = (fft_freqs >= 0.1) & (fft_freqs <= 0.5)
    features.append(np.sum(fft_vals[breath_mask]))

    # Energy di band gerakan (0.5 - 2.0 Hz)
    motion_mask = (fft_freqs >= 0.5) & (fft_freqs <= 2.0)
    features.append(np.sum(fft_vals[motion_mask]))

    return features

# Proses windowing per label
print("Extracting windowed features...")
X_all = []
y_all = []

for label, path in files.items():
    df_label = pd.read_csv(path)
    df_label["label"] = label

    # Sliding window
    for start in range(0, len(df_label) - WINDOW_SIZE, STEP_SIZE):
        window = df_label.iloc[start:start+WINDOW_SIZE]
        features = extract_window_features(window)
        X_all.append(features)
        y_all.append(label)

X = np.array(X_all)
y = np.array(y_all)

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

# Confusion matrix
labels = list(files.keys())
cm = confusion_matrix(y_test, y_pred, labels=labels)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix - v2 (Windowed + FFT)")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()