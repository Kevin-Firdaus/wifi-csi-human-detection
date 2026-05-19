import serial
import numpy as np
import time

PORT       = "COM4"
BAUD       = 115200
CAL_SAMPLES = 300  # 30 detik kalibrasi

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
        return rssi, values[:128]
    except:
        return None

def calibrate():
    print("=== KALIBRASI BASELINE ===")
    print("Pastikan ruangan KOSONG tanpa ada orang.")
    print("Kalibrasi dimulai dalam:")
    for i in range(10, 0, -1):
        print(f"  {i}...", end="\r")
        time.sleep(1)
    print("  MULAI!          ")

    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)

    amp_buffer = []
    rssi_buffer = []
    count = 0

    print(f"Mengumpulkan {CAL_SAMPLES} sample baseline...\n")
    while count < CAL_SAMPLES:
        raw = ser.readline().decode("utf-8", errors="ignore").strip()
        parsed = parse_line(raw)
        if parsed:
            rssi, values = parsed
            amp = compute_amplitude(np.array(values, dtype=float))
            amp_buffer.append(amp)
            rssi_buffer.append(rssi)
            count += 1
            bar = int(count / CAL_SAMPLES * 30)
            print(f"[{'█'*bar}{'░'*(30-bar)}] {count}/{CAL_SAMPLES}", end="\r")

    ser.close()

    # Hitung baseline
    baseline_amp  = np.mean(amp_buffer, axis=0)   # shape: (64,)
    baseline_std  = np.std(amp_buffer, axis=0)    # shape: (64,)
    baseline_rssi = np.mean(rssi_buffer)

    # Simpan baseline
    np.save("baseline_amp.npy", baseline_amp)
    np.save("baseline_std.npy", baseline_std)
    np.save("baseline_rssi.npy", np.array([baseline_rssi]))

    print(f"\n\nBaseline tersimpan!")
    print(f"Mean amplitude global: {np.mean(baseline_amp):.2f}")
    print(f"Mean RSSI baseline: {baseline_rssi:.2f} dBm")

if __name__ == "__main__":
    calibrate()