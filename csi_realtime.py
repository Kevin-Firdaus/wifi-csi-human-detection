#Rekam realtime CSI relatif

import serial
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

PORT        = "COM4"
BAUD        = 115200
WINDOW_SIZE = 50
FS          = 10

# Load model dan baseline
clf           = joblib.load("csi_model_v2.pkl")
baseline_amp  = np.load("baseline_amp.npy")
baseline_std  = np.load("baseline_std.npy")
baseline_rssi = np.load("baseline_rssi.npy")[0]
print(f"Model & baseline loaded!")
print(f"Baseline RSSI: {baseline_rssi:.2f} dBm")

buffer           = deque(maxlen=WINDOW_SIZE)
rssi_buf         = deque([0]*100, maxlen=100)
prediction_history = deque(["no_human"]*30, maxlen=30)

ser = serial.Serial(PORT, BAUD, timeout=1)

def compute_amplitude(vals):
    return np.array([np.sqrt(vals[i]**2 + vals[i+1]**2)
                     for i in range(0, len(vals), 2)])

def parse_line(line):
    try:
        if not line.startswith("CSI"):
            return None
        parts  = line.split("|")
        meta   = parts[0].strip().split()
        values = list(map(int, parts[1].strip().split()))
        rssi   = int(meta[2].split("=")[1])
        noise  = int(meta[3].split("=")[1])
        return rssi, noise, values[:128]
    except:
        return None

def extract_features(window):
    amps  = np.array([compute_amplitude(np.array(w[2], dtype=float))
                      for w in window])
    rssis = np.array([w[0] for w in window], dtype=float)

    delta      = amps - baseline_amp
    norm_delta = delta / (baseline_std + 1e-6)
    rssi_delta = rssis - baseline_rssi

    features = []
    features += list(np.mean(delta, axis=0))
    features += list(np.std(delta, axis=0))
    features += list(np.mean(norm_delta, axis=0))
    features += list(np.std(norm_delta, axis=0))
    features.append(np.mean(np.abs(delta)))
    features.append(np.std(delta))
    features.append(np.max(np.abs(delta)))
    features.append(np.mean(norm_delta))
    features.append(np.std(norm_delta))
    features.append(np.mean(rssi_delta))
    features.append(np.std(rssi_delta))
    features.append(np.max(rssi_delta) - np.min(rssi_delta))

    delta_signal = np.mean(np.abs(delta), axis=1)
    fft_vals     = np.abs(np.fft.rfft(delta_signal))
    fft_freqs    = np.fft.rfftfreq(WINDOW_SIZE, d=1.0/FS)
    features    += list(fft_vals)
    features.append(np.argmax(fft_vals) * FS / WINDOW_SIZE)
    features.append(np.max(fft_vals))

    breath_mask = (fft_freqs >= 0.1) & (fft_freqs <= 0.5)
    features.append(np.sum(fft_vals[breath_mask]))
    motion_mask = (fft_freqs >= 0.5) & (fft_freqs <= 2.0)
    features.append(np.sum(fft_vals[motion_mask]))

    return np.array(features).reshape(1, -1)

# Setup plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
fig.patch.set_facecolor("#1e1e1e")
for ax in [ax1, ax2]:
    ax.set_facecolor("#2d2d2d")

line_rssi, = ax1.plot([], [], color="royalblue")
ax1.set_xlim(0, 100)
ax1.set_ylim(-100, -30)
ax1.set_ylabel("RSSI (dBm)", color="white")
ax1.set_title("RSSI", color="white")
ax1.tick_params(colors="white")
ax1.grid(True, alpha=0.2)

detection_text = ax2.text(0.5, 0.5, "Initializing...",
    transform=ax2.transAxes,
    fontsize=48, fontweight="bold",
    ha="center", va="center", color="white")
confidence_text = ax2.text(0.5, 0.25, "",
    transform=ax2.transAxes,
    fontsize=16, ha="center", va="center", color="gray")
ax2.set_title("Detection", color="white")
ax2.set_xticks([])
ax2.set_yticks([])

def update(frame):
    try:
        raw    = ser.readline().decode("utf-8", errors="ignore").strip()
        parsed = parse_line(raw)

        if parsed:
            rssi, noise, values = parsed
            buffer.append((rssi, noise, values))
            rssi_buf.append(rssi)
            line_rssi.set_data(range(100), list(rssi_buf))

            if len(buffer) == WINDOW_SIZE:
                features = extract_features(list(buffer))
                pred     = clf.predict(features)[0]
                proba    = clf.predict_proba(features)[0]
                confidence = max(proba) * 100

                prediction_history.append(pred)
                human_count = list(prediction_history).count("human")
                final_pred  = "human" if human_count >= 15 else "no_human"

                print(f"Pred: {pred} | Conf: {confidence:.1f}% | Human votes: {human_count}/30")

                if final_pred == "human":
                    detection_text.set_text("HUMAN DETECTED")
                    detection_text.set_color("#ff4444")
                    ax2.set_facecolor("#3d1515")
                else:
                    detection_text.set_text("NO HUMAN")
                    detection_text.set_color("#44ff44")
                    ax2.set_facecolor("#153d15")

                confidence_text.set_text(f"Confidence: {confidence:.1f}% | Votes: {human_count}/30")
                fig.canvas.draw_idle()

    except Exception:
        pass

    return line_rssi, detection_text, confidence_text

ani = animation.FuncAnimation(fig, update, interval=100, blit=False)
plt.tight_layout()
plt.show()