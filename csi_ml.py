import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# =====================
# 1. LOAD DATA
# =====================
files = {
    "empty_room": "csi_empty_room.csv",
    "object_motion_center": "csi_object_motion_center.csv",
    "object_motion_45deg": "csi_object_motion_45deg.csv",
    "human_idle_center": "csi_human_idle_center.csv",
    "human_idle_45deg": "csi_human_idle_45deg.csv",
    "human_motion_center": "csi_human_motion_center.csv",
    "human_motion_45deg": "csi_human_motion_45deg.csv",
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
print(df["label"].value_counts())


# =====================
# 2. FEATURE EXTRACTION
# =====================
def extract_features(row):
    vals = row[[f"v{i}" for i in range(128)]].values.astype(float)

    # Hitung amplitude per subcarrier
    amps = np.array([np.sqrt(vals[i] ** 2 + vals[i + 1] ** 2)
                     for i in range(0, 128, 2)])

    # Hitung phase per subcarrier
    phases = np.array([np.arctan2(vals[i + 1], vals[i])
                       for i in range(0, 128, 2)])

    features = [
        np.mean(amps),  # mean amplitude
        np.std(amps),  # std amplitude
        np.var(amps),  # variance amplitude
        np.max(amps),  # max amplitude
        np.min(amps),  # min amplitude
        np.mean(phases),  # mean phase
        np.std(phases),  # std phase
        row["rssi"],  # RSSI
        row["noise"],  # noise floor
    ]

    # Tambah amplitude per subcarrier sebagai fitur
    features += list(amps)

    return features


print("Extracting features...")
X = np.array([extract_features(row) for _, row in df.iterrows()])
y = df["label"].values
print(f"Feature vector size: {X.shape[1]}")

# =====================
# 3. TRAIN MODEL
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("\nTraining Random Forest...")
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# =====================
# 4. EVALUASI
# =====================
y_pred = clf.predict(X_test)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=list(files.keys()))
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=list(files.keys()),
            yticklabels=list(files.keys()))
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()