import serial
import csv
import time
from datetime import datetime

PORT = "COM4"
BAUD = 115200
MAX_SAMPLES = 1000
COUNTDOWN_SECONDS = 10  # waktu untuk keluar ruangan

def parse_csi_line(line):
    try:
        if not line.startswith("CSI"):
            return None
        parts = line.split("|")
        meta   = parts[0].strip().split()
        values = list(map(int, parts[1].strip().split()))
        length = int(meta[1].split("=")[1])
        rssi   = int(meta[2].split("=")[1])
        noise  = int(meta[3].split("=")[1])
        return {
            "timestamp": datetime.now().isoformat(),
            "len": length,
            "rssi": rssi,
            "noise": noise,
            "values": values[:128]
        }
    except:
        return None

def get_filename():
    print("\n=== CSI Data Logger ===")
    print("Kategori tersedia:")
    print("  1  - empty_room")
    print("  2  - object_motion_center")
    print("  2.1- object_motion_45deg")
    print("  3  - human_idle_center")
    print("  3.1- human_idle_45deg")
    print("  4  - human_motion_center")
    print("  4.1- human_motion_45deg")
    label = input("\nMasukkan label sesi ini: ").strip()
    filename = f"csi_{label}.csv"
    print(f"Output file: empty_room")
    return filename

def countdown(seconds):
    print(f"\nPengambilan data dimulai dalam:")
    for i in range(seconds, 0, -1):
        print(f"  {i}...", end="\r")
        time.sleep(1)
    print("  MULAI!          ")

def main():
    filename = get_filename()

    print(f"\nMembuka port {PORT}...")
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)

    countdown(COUNTDOWN_SECONDS)

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "len", "rssi", "noise"] +
                        [f"v{i}" for i in range(128)])

        count = 0
        print(f"Merekam {MAX_SAMPLES} sample...\n")

        while count < MAX_SAMPLES:
            raw = ser.readline().decode("utf-8", errors="ignore").strip()
            parsed = parse_csi_line(raw)
            if parsed:
                row = [
                    parsed["timestamp"],
                    parsed["len"],
                    parsed["rssi"],
                    parsed["noise"]
                ] + parsed["values"]
                writer.writerow(row)
                f.flush()
                count += 1

                # Progress bar sederhana
                bar = int(count / MAX_SAMPLES * 30)
                print(f"[{'█'*bar}{'░'*(30-bar)}] {count}/{MAX_SAMPLES}", end="\r")

    ser.close()
    print(f"\n\nSelesai! Data tersimpan di: {filename}")

if __name__ == "__main__":
    main()