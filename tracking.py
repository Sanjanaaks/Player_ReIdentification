# track_players.py

import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Paths
MODEL_PATH = 'best.pt'
VIDEO_PATH = '15sec_input_720p.mp4'
OUTPUT_PATH = '15sec_tracked_output.mp4'

# Load YOLOv11 model
model = YOLO(MODEL_PATH)

# Initialize Deep SORT Tracker
tracker = DeepSort(max_age=30, max_iou_distance=0.7, n_init=3)

# Open video capture
cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Output video writer
out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv11 inference
    results = model(frame, verbose=False)[0]

    detections = []
    for det in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = det
        if int(cls) == 0:  # Assuming class 0 is 'player'
            bbox = [x1, y1, x2 - x1, y2 - y1]  # x, y, w, h
            detections.append((bbox, conf, 'player'))

    # Update tracker
    tracks = tracker.update_tracks(detections, frame=frame)
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        l, t, r, b = track.to_ltrb()
        
        cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (int(l), int(t - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    out.write(frame)
    cv2.imshow("Player ReID", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Tracking complete. Output saved to:", OUTPUT_PATH)
