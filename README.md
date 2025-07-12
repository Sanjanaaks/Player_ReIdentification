# Player_ReIdentification
## 📁 Project Structure

```
.
├── tracking.py                  # Main player tracking script
├── best.pt                      # Trained YOLOv11 model
├── 15sec_input_720p.mp4         # Input video (15 seconds)
├── 15sec_tracked_output.mp4     # Output video with tracked player IDs
├── README.md                    # Setup and run instructions
└── REPORT.md                    # Approach, techniques, and challenges
```

---

## ⚙️ Setup Instructions

### 1. Install Dependencies

Ensure Python 3.8+ is installed. Run the following:

```bash
pip install ultralytics deep_sort_realtime opencv-python
```

Also ensure `torch` is installed:

```bash
pip install torch torchvision
```

### 2. Run the Tracker

```bash
python tracking.py
```

This will read `15sec_input_720p.mp4`, run detection + tracking, and write to `15sec_tracked_output.mp4`.

---

## 🧾 Notes

- Ensure `best.pt` (YOLOv11 model) and video files are in the same folder as `tracking.py`.
- Output is saved automatically and includes bounding boxes and consistent IDs for re-identification.
