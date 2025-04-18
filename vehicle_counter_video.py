import cv2
import requests
import time
import csv
from datetime import datetime
import os
from ultralytics import YOLO

print("🚀 Starting vehicle detection script...")

# Load the YOLO model
try:
    print("📦 Loading YOLO model...")
    model = YOLO("yolov8n.pt")
except Exception as e:
    print("❌ Failed to load YOLO model:", e)
    exit()

# Define vehicle classes to detect
vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']

# Load the video
#video_path = "dataset/sample2.mp4"
video_path = "dataset/videoplayback.mp4"
print(f"📹 Attempting to load video from: {video_path}")
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Failed to open video. Please check the path or video format.")
    exit()
else:
    print("✅ Video loaded successfully.")

# Setup CSV file for logging
log_file = "vehicle_log.csv"
if not os.path.exists(log_file):
    with open(log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Vehicle_Count"])
    print(f"📄 Created CSV log file: {log_file}")
else:
    print(f"📄 Logging to existing file: {log_file}")

# Function to send data to backend
def send_vehicle_count_to_backend(count):
    try:
        response = requests.post("http://localhost:5000/vehicle_count", json={"count": count})
        if response.status_code == 200:
            print(f"✅ Count sent to backend: {count}")
        else:
            print(f"❌ Backend error - Status code: {response.status_code}")
    except Exception as e:
        print("⚠️ Error while sending to backend:", e)

# Function to log count to CSV
def log_vehicle_count_to_csv(count):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, count])
    print(f"📝 Logged to CSV: {timestamp}, {count}")

# Process video
frame_count = 0
last_sent_time = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("⚠️ Could not read frame. Exiting loop.")
        break

    print("🔁 Processing frame...")
    frame_count += 1
    if frame_count % 5 != 0:
        continue

    results = model(frame)
    vehicle_count = 0

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        if class_name in vehicle_classes:
            vehicle_count += 1

    # Annotate and display frame
    annotated_frame = results[0].plot()
    cv2.putText(annotated_frame, f"Vehicles: {vehicle_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.imshow("🚗 Vehicle Detection - Video", annotated_frame)

    # Send and log every second
    if time.time() - last_sent_time >= 1:
        send_vehicle_count_to_backend(vehicle_count)
        log_vehicle_count_to_csv(vehicle_count)
        last_sent_time = time.time()

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Quitting vehicle detection.")
        break

cap.release()
cv2.destroyAllWindows()
