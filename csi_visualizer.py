import serial
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

PORT = "COM4"
BAUD = 115200
N_SUBCARRIER = 64  # 128 bytes = 64 pasang I/Q
WINDOW = 100

# Buffer RSSI dan amplitude per subcarrier
rssi_buf = deque([0]*WINDOW, maxlen=WINDOW)
amp_buf  = deque([np.zeros(N_SUBCARRIER)]*WINDOW, maxlen=WINDOW)

ser = serial.Serial(PORT, BAUD, timeout=1)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))
fig.suptitle("Wi-Fi CSI Realtime Visualizer")

# Plot RSSI
line_rssi, = ax1.plot([], [], color="royalblue")
ax1.set_xlim(0, WINDOW)
ax1.set_ylim(-100, -30)
ax1.set_ylabel("RSSI (dBm)")
ax1.set_title("RSSI")
ax1.grid(True, alpha=0.3)

# Heatmap CSI amplitude
heatmap_data = np.zeros((N_SUBCARRIER, WINDOW))
im = ax2.imshow(heatmap_data, aspect="auto", origin="lower",
                cmap="jet", vmin=0, vmax=50,
                extent=[0, WINDOW, 0, N_SUBCARRIER])
plt.colorbar(im, ax=ax2, label="Amplitude")
ax2.set_ylabel("Subcarrier index")
ax2.set_xlabel("Sample")
ax2.set_title("CSI Amplitude Heatmap (semua subcarrier)")

def parse_line(line):
    try:
        if not line.startswith("CSI"):
            return None
        parts = line.split("|")
        meta   = parts[0].strip().split()
        values = list(map(int, parts[1].strip().split()))
        rssi   = int(meta[2].split("=")[1])

        # Hitung amplitude: sqrt(I^2 + Q^2) per pasang
        amps = []
        for i in range(0, len(values)-1, 2):
            amp = np.sqrt(values[i]**2 + values[i+1]**2)
            amps.append(amp)
        return rssi, np.array(amps[:N_SUBCARRIER])
    except:
        return None

def update(frame):
    try:
        raw = ser.readline().decode("utf-8", errors="ignore").strip()
        parsed = parse_line(raw)
        if parsed:
            rssi, amps = parsed
            rssi_buf.append(rssi)
            amp_buf.append(amps)

            # Update RSSI
            line_rssi.set_data(range(WINDOW), list(rssi_buf))

            # Update heatmap
            heatmap = np.array(list(amp_buf)).T
            im.set_data(heatmap)

    except:
        pass
    return line_rssi, im

ani = animation.FuncAnimation(fig, update, interval=50, blit=True)
plt.tight_layout()
plt.show()