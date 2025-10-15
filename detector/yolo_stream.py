import cv2
import time
import threading
from ultralytics import YOLO

# --- Configuration ---
CAMERA_SRC = "rtsp://192.168.31.120:8080/h264_ulaw.sdp"
MODEL_PATH = "best.pt"  # can be detection or segmentation model (e.g., yolov8n.pt or yolov8n-seg.pt)
IMG_SIZE = 640

# --- Global Variables ---
latest_frame = None
detections = []
frame_lock = threading.Lock()


def capture_and_detect():
    global latest_frame, detections

    model = YOLO(MODEL_PATH)

    # detect if this is a segmentation model
    is_segmentation = any("seg" in k for k in model.names.keys()) if hasattr(model, "names") else ("seg" in MODEL_PATH)
    # or more reliably:
    is_segmentation = getattr(model, "task", "") == "segment"

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
        results_list = model(frame, imgsz=IMG_SIZE, verbose=False)
        if not results_list:
            continue

        results = results_list[0]

        # Ensure visualization works for all model types
        if hasattr(results, "plot"):
            annotated = results.plot()  # draws boxes or masks automatically
        else:
            annotated = frame

        if annotated is None or annotated.size == 0:
            annotated = frame  # fallback to original frame

        # Encode frame to JPEG
        success, jpeg = cv2.imencode(".jpg", annotated)
        if not success:
            continue

        with frame_lock:
            latest_frame = jpeg.tobytes()
            detections = []

            # --- Collect detections ---
            if hasattr(results, "boxes") and results.boxes is not None:
                for box in results.boxes:
                    detections.append({
                        "label": model.names[int(box.cls)],
                        "confidence": float(box.conf),
                        "xyxy": box.xyxy[0].tolist()
                    })

            # --- Collect segmentation masks if available ---
            if hasattr(results, "masks") and results.masks is not None:
                for idx, mask in enumerate(results.masks.data):
                    label = (
                        model.names[int(results.boxes.cls[idx])]
                        if hasattr(results, "boxes") and results.boxes is not None
                        else "object"
                    )
                    confidence = (
                        float(results.boxes.conf[idx])
                        if hasattr(results, "boxes") and results.boxes is not None
                        else 0.0
                    )
                    xyxy = (
                        results.boxes.xyxy[idx].tolist()
                        if hasattr(results, "boxes") and results.boxes is not None
                        else []
                    )
                    detections.append({
                        "label": label,
                        "confidence": confidence,
                        "xyxy": xyxy,
                        "mask_points": results.masks.xy[idx].tolist()
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
