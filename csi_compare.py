import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

IDLE_FILE   = "csi_data_2 Mei_no movement.csv"
MOTION_FILE = "csi_data_2 Mei_with movement.csv"

def load_amplitude(filepath):
    df = pd.read_csv(filepath)
    # Ambil kolom v0 sampai v127 (pasangan I/Q)
    vals = df[[f"v{i}" for i in range(128)]].values
    # Hitung amplitude per subcarrier (sqrt(I^2 + Q^2))
    amps = []
    for row in vals:
        amp = [np.sqrt(row[i]**2 + row[i+1]**2) for i in range(0, 128, 2)]
        amps.append(amp)
    return np.array(amps), df["rssi"].values

idle_amp,   idle_rssi   = load_amplitude(IDLE_FILE)
motion_amp, motion_rssi = load_amplitude(MOTION_FILE)

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle("CSI Comparison: Idle vs Motion")

# RSSI idle
axes[0,0].plot(idle_rssi, color="royalblue")
axes[0,0].set_title("RSSI - Idle")
axes[0,0].set_ylabel("RSSI (dBm)")
axes[0,0].set_ylim(-100, -30)
axes[0,0].grid(True, alpha=0.3)

# RSSI motion
axes[0,1].plot(motion_rssi, color="tomato")
axes[0,1].set_title("RSSI - Motion")
axes[0,1].set_ylim(-100, -30)
axes[0,1].grid(True, alpha=0.3)

# Heatmap idle
im1 = axes[1,0].imshow(idle_amp.T, aspect="auto", origin="lower",
                        cmap="jet", vmin=0, vmax=50)
axes[1,0].set_title("CSI Amplitude Heatmap - Idle")
axes[1,0].set_ylabel("Subcarrier index")
axes[1,0].set_xlabel("Sample")
plt.colorbar(im1, ax=axes[1,0])

# Heatmap motion
im2 = axes[1,1].imshow(motion_amp.T, aspect="auto", origin="lower",
                        cmap="jet", vmin=0, vmax=50)
axes[1,1].set_title("CSI Amplitude Heatmap - Motion")
axes[1,1].set_xlabel("Sample")
plt.colorbar(im2, ax=axes[1,1])

plt.tight_layout()
plt.show()