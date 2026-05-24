# Wi-Fi CSI Human Presence Detection

> Passive RF sensing system for human presence detection using Wi-Fi Channel State Information (CSI) — built on ESP32-C6 hardware.

## System Overview

This system uses two ESP32-C6 units (TX and RX) to continuously transmit and receive Wi-Fi signals. When a human enters the sensing area, the multipath propagation of the signal changes, which is captured as variations in the CSI amplitude. A machine learning classifier then determines presence or absence in real time.


## Hardware Stack

| Component | Role |
|-----------|------|
| ESP32-C6 (TX) | CSI transmitter — Station mode |
| ESP32-C6 (RX) | CSI receiver — SoftAP mode |
| USB-Serial | Data streaming to host PC |

## Software Pipeline

| Script | Function |
|--------|----------|
| `csi_logger.py` | Log raw CSI data from RX unit via serial |
| `csi_calibrate.py` | Baseline calibration for empty-room reference |
| `csi_train.py` | Train ML classifier on collected CSI dataset |
| `csi_realtime.py` | Real-time inference and presence detection |
| `csi_visualizer.py` | CSI amplitude visualization and analysis |

## Procedure

**1. Flash firmware**

Using Arduino IDE:
- Open `Firmware/TX/TX.ino` → select ESP32-C6 board → upload to transmitter unit
- Open `Firmware/RX/RX.ino` → select ESP32-C6 board → upload to receiver unit

**2. Installing dependencies**
```bash
pip install -r requirements.txt
```

**3. Calibration**
```bash
python csi_calibrate.py
```

**4. Model training and data collection**
```bash
python csi_logger.py
python csi_train.py
```

**5. Run real-time detection**
```bash
python csi_realtime.py
```

## Realtime Serial Monitor



https://github.com/user-attachments/assets/d93277cf-716b-4350-8b9f-c59040b73f4f



## Confusion Matrix

<img width="933" height="864" alt="Screenshot 2026-05-17 102632" src="https://github.com/user-attachments/assets/fcd48796-21b0-4753-a820-7e0dd73698d0" />

<img width="1276" height="789" alt="Screenshot 2026-05-17 102648" src="https://github.com/user-attachments/assets/9e6f02f5-90af-4155-8a08-48de7dbfdcba" />



## Dataset

Sample CSI data is provided in `data/sample/csi_sample.csv`.
Full dataset available on request.

## Project Status

- [x] CSI data logging via pyserial
- [x] Baseline calibration
- [x] ML classifier training (Random Forest)
- [x] Real-time detection (controlled environment)
- [ ] Multi-RX AoA estimation
- [ ] 2D position mapping

## Known Limitations

- System tested in a 3×3 meter room under controlled conditions
- Detection consistency varies with environmental changes (furniture repositioning, temperature, interference)
- Classifier requires recalibration when deployed in a new environment
- Current architecture uses single TX-RX pair; positional accuracy not yet implemented

## License

MIT License — see [LICENSE](LICENSE) for details.
