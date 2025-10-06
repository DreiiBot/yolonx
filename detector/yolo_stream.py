import cv2, time, threading, requests
from ultralytics import YOLO

# Update with your IP Webcam / RTSP source
CAMERA_SRC = "http://192.168.31.120:8080/video"
MODEL_PATH = "yolov8n.pt"
IMG_SIZE = 640

# Replace this with your Nx Meta server address and port
NX_META_SERVER = "http://192.168.31.234:7001"  # or 7001/7002 depending on config

latest_frame = None
detections = []
frame_lock = threading.Lock()

def send_nx_event(label, confidence):
    """Send YOLO detection event to Nx Meta server."""
    try:
        requests.get(
            f"{NX_META_SERVER}/api/createEvent",
            params={
                "source": "YOLO",
                "caption": f"{label} Detected",
                "description": f"Confidence: {confidence:.2f}",
                "metadata": f"confidence={confidence}"
            },
            timeout=1
        )
    except Exception as e:
        print(f"[NxMeta] Failed to send event: {e}")

def capture_and_detect():
    global latest_frame, detections
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(CAMERA_SRC)

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.5)
            cap = cv2.VideoCapture(CAMERA_SRC)
            continue

        results = model(frame, imgsz=IMG_SIZE)[0]
        annotated = results.plot()

        success, jpeg = cv2.imencode(".jpg", annotated)
        if not success:
            continue

        with frame_lock:
            latest_frame = jpeg.tobytes()
            detections = []

            if results.boxes is not None:
                for box in results.boxes:
                    label = model.names[int(box.cls)]
                    confidence = float(box.conf)
                    xyxy = box.xyxy[0].tolist()

                    detections.append({
                        "label": label,
                        "confidence": confidence,
                        "xyxy": xyxy
                    })

                    # 🔹 Send event to Nx Meta for each detection
                    send_nx_event(label, confidence)

def mjpeg_generator():
    global latest_frame
    while True:
        with frame_lock:
            frame = latest_frame
        if frame:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.05)

# Start background capture thread
threading.Thread(target=capture_and_detect, daemon=True).start()
