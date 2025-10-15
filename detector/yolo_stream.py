import cv2
import time
import threading
from ultralytics import YOLO

# --- Configuration ---
CAMERA_SRC = "rtsp://192.168.31.120:8080/h264_ulaw.sdp"  # update with your phone/cam URL
MODEL_PATH = "best.pt"  # supports both detection and segmentation models (e.g., best-seg.pt)
IMG_SIZE = 640

# --- Global Variables ---
latest_frame = None
detections = []
frame_lock = threading.Lock()


def capture_and_detect():
    """
    Continuously captures frames from the camera and runs YOLO detection/segmentation.
    """
    global latest_frame, detections

    # Load YOLO model (automatically detects if segmentation)
    model = YOLO(MODEL_PATH)
    is_segmentation = model.task == "segment"

    cap = cv2.VideoCapture(CAMERA_SRC)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera source: {CAMERA_SRC}")

    print(f"[INFO] Loaded model '{MODEL_PATH}' (Task: {'Segmentation' if is_segmentation else 'Detection'})")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame grab failed. Reconnecting...")
            time.sleep(0.5)
            cap = cv2.VideoCapture(CAMERA_SRC)
            continue

        # Run inference
        results = model(frame, imgsz=IMG_SIZE, verbose=False)[0]

        # Plot annotated frame (works for both detection + segmentation)
        annotated = results.plot()

        # Encode as JPEG for streaming
        success, jpeg = cv2.imencode(".jpg", annotated)
        if not success:
            continue

        with frame_lock:
            latest_frame = jpeg.tobytes()
            detections = []

            # Handle detections or segmentation masks
            if is_segmentation and results.masks is not None:
                for idx, mask in enumerate(results.masks.data):
                    label = model.names[int(results.boxes.cls[idx])] if results.boxes is not None else "object"
                    confidence = float(results.boxes.conf[idx]) if results.boxes is not None else 0.0
                    xyxy = results.boxes.xyxy[idx].tolist() if results.boxes is not None else []
                    detections.append({
                        "label": label,
                        "confidence": confidence,
                        "xyxy": xyxy,
                        "mask_points": results.masks.xy[idx].tolist()  # list of polygon points
                    })

            elif results.boxes is not None:  # regular detection
                for box in results.boxes:
                    detections.append({
                        "label": model.names[int(box.cls)],
                        "confidence": float(box.conf),
                        "xyxy": box.xyxy[0].tolist()
                    })


def mjpeg_generator():
    """
    Generator for MJPEG streaming endpoint.
    """
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
