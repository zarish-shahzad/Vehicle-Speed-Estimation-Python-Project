# 🚗 Vehicle Speed Estimation using YOLO 11 + OpenCV

Real-time vehicle speed detection from video using YOLO 11's built-in object tracker and simple line-crossing geometry — no radar hardware required.

---

## 📌 Overview

This project detects vehicles in a video, tracks each one with a persistent ID across frames, and calculates its real-world speed (in km/h) by timing how long it takes to travel between two reference lines drawn on the road.

---

## ✨ Features

- 🚘 Detects cars, motorcycles, buses, and trucks
- 🆔 Assigns a stable ID to each vehicle using YOLO's built-in tracker
- 📏 Calculates real speed (km/h) using two calibrated reference lines
- 🎥 Draws bounding boxes, IDs, and speed labels live on the video
- 💾 Saves the final annotated video as `output.mp4`

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Core language |
| [Ultralytics YOLO 11](https://github.com/ultralytics/ultralytics) | Vehicle detection + tracking |
| OpenCV | Video reading, drawing, and export |
| NumPy | Array operations |

---

## 📂 Project Structure

```
├── main.py          # Main script (detection, tracking, speed calc)
├── yolo11x.pt        # Pretrained YOLO 11 weights
├── 01.mp4             # Input video
└── output.mp4         # Output video with speed overlay
```

---

## ⚙️ Installation

```bash
git clone https://github.com/<your-username>/vehicle-speed-estimation.git
cd vehicle-speed-estimation

pip install ultralytics opencv-python numpy
```

---

## ▶️ Usage

1. Place your input video in the project folder and update `VIDEO_PATH` in `main.py`.
2. Adjust the two reference lines (`first_line`, `second_line`) to match your video's road layout.
3. Set `REAL_DISTANCE_M` to the actual real-world distance (in meters) between the two lines.
4. Run the script:

```bash
python main.py
```

5. Press `q` to stop early. The annotated video is saved as `output.mp4`.

---

## 🧠 How It Works

1. Read the video frame by frame.
2. Run YOLO 11 detection + tracking (`model.track(persist=True)`) to get bounding boxes and IDs.
3. Keep only vehicle classes (car, motorcycle, bus, truck).
4. Track each vehicle's center point.
5. When a vehicle crosses **Line A**, record the frame number.
6. When it crosses **Line B**, record the frame number again.
7. Speed = (distance between lines ÷ time between crossings) × 3.6 → km/h
8. Draw the bounding box, ID, and speed on the frame.

---

## 🔧 Configuration

| Variable | Description |
|----------|--------------|
| `VIDEO_PATH` | Path to input video |
| `first_line`, `second_line` | Coordinates of the two reference lines |
| `REAL_DISTANCE_M` | Real-world distance between the lines (meters) |
| `MARGIN` | Pixel tolerance for detecting a line crossing |

---

## 🚀 Future Improvements

- Multi-lane support with automatic calibration
- Automatic violation alerts (SMS / dashboard)
- Deployment on edge devices (Jetson / Raspberry Pi)



## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
