#Rekam CSI relatif

import serial
import csv
import numpy as np
import time
from datetime import datetime

PORT        = "COM4"
BAUD        = 115200
MAX_SAMPLES = 1000
COUNTDOWN   = 10

# Load baseline
baseline_amp  = np.load("baseline_amp.npy")
baseline_std  = np.load("baseline_std.npy")
baseline_rssi = np.load("baseline_rssi.npy")[0]

print(f"Baseline loaded! Mean amp: {np.mean(baseline_amp):.2f}, RSSI: {baseline_rssi:.2f} dBm")

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

def get_filename():
    print("\n=== CSI Data Logger v2 (Relative) ===")
    print("Kategori:")
    print("  1 - empty_room")
    print("  2 - object_motion")
    print("  3 - human_motion")
    print("  4 - human_idle")
    label = input("\nMasukkan label sesi ini: ").strip()
    return f"csi_rel_{label}.csv", label

def countdown(seconds):
    print(f"\nDimulai dalam:")
    for i in range(seconds, 0, -1):
        print(f"  {i}...", end="\r")
        time.sleep(1)
    print("  MULAI!          ")

def main():
    filename, label = get_filename()

    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)

    countdown(COUNTDOWN)

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        # Header: delta per subcarrier + fitur relatif tambahan
        header = ["timestamp", "rssi", "rssi_delta", "noise"]
        header += [f"amp_{i}" for i in range(64)]
        header += [f"delta_{i}" for i in range(64)]
        header += [f"norm_delta_{i}" for i in range(64)]
        writer.writerow(header)

        count = 0
        print(f"Merekam {MAX_SAMPLES} sample...\n")

        while count < MAX_SAMPLES:
            raw = ser.readline().decode("utf-8", errors="ignore").strip()
            parsed = parse_line(raw)
            if parsed:
                rssi, noise, values = parsed
                amp = compute_amplitude(np.array(values, dtype=float))

                # Hitung fitur relatif
                delta      = amp - baseline_amp
                norm_delta = delta / (baseline_std + 1e-6)
                rssi_delta = rssi - baseline_rssi

                row = (
                    [datetime.now().isoformat(), rssi, rssi_delta, noise]
                    + list(amp)
                    + list(delta)
                    + list(norm_delta)
                )
                writer.writerow(row)
                f.flush()
                count += 1

                bar = int(count / MAX_SAMPLES * 30)
                print(f"[{'█'*bar}{'░'*(30-bar)}] {count}/{MAX_SAMPLES}", end="\r")

    ser.close()
    print(f"\n\nSelesai! Tersimpan: {filename}")

if __name__ == "__main__":
    main()