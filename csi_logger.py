import serial
import csv
import time
from datetime import datetime

PORT = "COM4"
BAUD = 115200
OUTPUT_FILE = "csi_data.csv"
N_VALUES = 128

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
            "values": values[:N_VALUES]
        }
    except:
        return None

def main():
    print(f"Opening {PORT} at {BAUD} baud...")
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "len", "rssi", "noise"] +
                        [f"v{i}" for i in range(N_VALUES)])

        print(f"Logging to {OUTPUT_FILE}... Tekan Ctrl+C untuk stop.")
        count = 0
        while True:
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
                print(f"[{count}] RSSI={parsed['rssi']}")

    ser.close()

if __name__ == "__main__":
    main()