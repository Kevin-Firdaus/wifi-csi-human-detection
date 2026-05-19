#ML relatif

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# =====================
# 1. LOAD DATA
# =====================
files = {
    "no_human": [
        "csi_rel_empty_room.csv",
        "csi_rel_object_motion.csv",
    ],
    "human": [
        "csi_rel_human_motion.csv",
        "csi_rel_human_idle.csv",
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

# Shuffle untuk hindari data leakage
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Total sample: {len(df)}")
print(df["label"].value_counts())

# =====================
# 2. WINDOWING + FEATURE EXTRACTION
# =====================
WINDOW_SIZE = 50
STEP_SIZE   = 25
FS          = 10

def extract_window_features(window_df):
    # Ambil kolom delta (relatif terhadap baseline)
    delta     = window_df[[f"delta_{i}" for i in range(64)]].values
    norm_delta = window_df[[f"norm_delta_{i}" for i in range(64)]].values
    amp       = window_df[[f"amp_{i}" for i in range(64)]].values

    features = []

    # Statistik delta per subcarrier
    features += list(np.mean(delta, axis=0))
    features += list(np.std(delta, axis=0))
    features += list(np.mean(norm_delta, axis=0))
    features += list(np.std(norm_delta, axis=0))

    # Statistik global
    features.append(np.mean(np.abs(delta)))
    features.append(np.std(delta))
    features.append(np.max(np.abs(delta)))
    features.append(np.mean(norm_delta))
    features.append(np.std(norm_delta))

    # RSSI relatif
    rssi_delta = window_df["rssi_delta"].values.astype(float)
    features.append(np.mean(rssi_delta))
    features.append(np.std(rssi_delta))
    features.append(np.max(rssi_delta) - np.min(rssi_delta))

    # FFT dari mean delta signal
    delta_signal = np.mean(np.abs(delta), axis=1)
    fft_vals  = np.abs(np.fft.rfft(delta_signal))
    fft_freqs = np.fft.rfftfreq(WINDOW_SIZE, d=1.0/FS)

    features += list(fft_vals)
    features.append(np.argmax(fft_vals) * FS / WINDOW_SIZE)
    features.append(np.max(fft_vals))

    # Energy band pernapasan
    breath_mask = (fft_freqs >= 0.1) & (fft_freqs <= 0.5)
    features.append(np.sum(fft_vals[breath_mask]))

    # Energy band gerakan
    motion_mask = (fft_freqs >= 0.5) & (fft_freqs <= 2.0)
    features.append(np.sum(fft_vals[motion_mask]))

    return features

print("Extracting windowed features...")
X_all = []
y_all = []

for label, paths in files.items():
    for path in paths:
        df_label = pd.read_csv(path)
        # Shuffle per file juga
        df_label = df_label.sample(frac=1, random_state=42).reset_index(drop=True)
        for start in range(0, len(df_label) - WINDOW_SIZE, STEP_SIZE):
            window = df_label.iloc[start:start+WINDOW_SIZE]
            features = extract_window_features(window)
            X_all.append(features)
            y_all.append(label)

X = np.array(X_all)
y = np.array(y_all)

# Shuffle final
idx = np.random.RandomState(42).permutation(len(X))
X, y = X[idx], y[idx]

print(f"Total windows: {len(X)}")
print(f"Feature vector size: {X.shape[1]}")
print(pd.Series(y).value_counts())

# =====================
# 3. TRAIN + CROSS VALIDATION
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("\nTraining Random Forest...")
clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# Cross validation untuk deteksi overfitting
cv_scores = cross_val_score(clf, X, y, cv=5)
print(f"\nCross-validation scores: {cv_scores}")
print(f"CV Mean: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

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
plt.title("Confusion Matrix - v4 (Relative Features)")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.show()

# Simpan model
joblib.dump(clf, "csi_model_v2.pkl")
print("Model tersimpan: csi_model_v2.pkl")