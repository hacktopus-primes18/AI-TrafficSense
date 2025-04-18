# vehicle_counter.py

from ultralytics import YOLO
import cv2
import requests


def count_vehicles(image_path):
    # Load the YOLO model
    model = YOLO("yolov8n.pt")  # You can use yolov8s.pt or yolov8m.pt for better accuracy

    # Read the image
    image = cv2.imread(image_path)

    # Run detection
    results = model(image)

    # Define vehicle classes
    vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']
    vehicle_count = 0
    
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            if class_name in vehicle_classes:
                vehicle_count += 1

    return vehicle_count, results[0]

def send_vehicle_count_to_backend(count):
    try:
        response = requests.post("http://localhost:5000/vehicle_count", json={"count": count})
        if response.status_code == 200:
            print("✅ Count sent successfully:", count)
        else:
            print("❌ Failed to send count. Status code:", response.status_code)
    except Exception as e:
        print("⚠️ Error sending count:", e)



# Run directly
if __name__ == "__main__":
    image_path = "your_image.jpg"  # Replace this with your input image
    count, result = count_vehicles(image_path)
  
    send_vehicle_count_to_backend(count)

    #print(f"Vehicles Detected: {count}")
    #result.show()  # Show bounding boxes

