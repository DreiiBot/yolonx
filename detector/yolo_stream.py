# detector/yolo_stream.py
import cv2, time, threading
from ultralytics import YOLO

CAMERA_SRC = "rtsp://192.168.31.120:8080/h264_ulaw.sdp"  # update with your phone/cam URL
MODEL_PATH = "best.pt"
IMG_SIZE = 640

latest_frame = None
detections = []
frame_lock = threading.Lock()

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
                    detections.append({
                        "label": model.names[int(box.cls)],
                        "confidence": float(box.conf),
                        "xyxy": box.xyxy[0].tolist()
                    })

def mjpeg_generator():
    global latest_frame
    while True:
        with frame_lock:
            frame = latest_frame
        if frame:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.05)

# Start background thread
threading.Thread(target=capture_and_detect, daemon=True).start()
