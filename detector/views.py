import os
import random
import zipfile
import shutil
import datetime
import yaml
import torch
import uuid
import time
from ultralytics import YOLO
from rest_framework.views import APIView
from rest_framework import status
from rest_framework.response import Response
from django.conf import settings
from django.http import StreamingHttpResponse, JsonResponse
from .mongodb import mongo_manager
from .yolo_stream import camera_manager

BASE_STORAGE = os.path.join(settings.BASE_DIR, "yolo_storage")
DATASETS_DIR = os.path.join(BASE_STORAGE, "datasets")
CONFIGS_DIR = os.path.join(BASE_STORAGE, "configs")
RUNS_DIR = os.path.join(BASE_STORAGE, "runs")

for d in [DATASETS_DIR, CONFIGS_DIR, RUNS_DIR]:
    os.makedirs(d, exist_ok=True)


class CreateYAMLFromZipView(APIView):
    """
    Accepts a local ZIP path, extracts it in the same directory,
    expects inside the ZIP:
        - images/
        - labels/
        - classes.txt
    Generates data.yaml automatically.
    """

    def post(self, request):
        try:
            zip_path = request.data.get("dataset_path")

            if not zip_path:
                return Response({"error": "Missing dataset_path"}, status=status.HTTP_400_BAD_REQUEST)
            if not os.path.exists(zip_path):
                return Response({"error": f"ZIP file not found: {zip_path}"}, status=status.HTTP_400_BAD_REQUEST)
            if not zip_path.lower().endswith(".zip"):
                return Response({"error": "Provided path must be a .zip file"}, status=status.HTTP_400_BAD_REQUEST)

            # Determine extract folder (same name as zip)
            extract_dir = os.path.splitext(zip_path)[0]
            os.makedirs(extract_dir, exist_ok=True)

            # Extract the ZIP
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)

            # Verify required contents
            images_dir = os.path.join(extract_dir, "images")
            labels_dir = os.path.join(extract_dir, "labels")
            classes_txt = os.path.join(extract_dir, "classes.txt")

            if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
                return Response({"error": "Dataset must contain 'images/' and 'labels/' folders"}, status=status.HTTP_400_BAD_REQUEST)
            if not os.path.isfile(classes_txt):
                return Response({"error": "Missing 'classes.txt' file"}, status=status.HTTP_400_BAD_REQUEST)

            # Read classes
            with open(classes_txt, "r", encoding="utf-8") as f:
                classes = [line.strip() for line in f if line.strip()]
            nc = len(classes)

            # Collect and split images
            all_images = [fn for fn in os.listdir(images_dir) if fn.lower().endswith((".jpg", ".jpeg", ".png"))]
            if not all_images:
                return Response({"error": "No images found in images/ folder"}, status=status.HTTP_400_BAD_REQUEST)

            random.shuffle(all_images)
            split_idx = int(len(all_images) * 0.8)
            train_imgs = all_images[:split_idx]
            val_imgs = all_images[split_idx:]

            # Create new dataset split structure
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            data_root = os.path.join(extract_dir, f"data_{ts}")
            for sub in ["train/images", "train/labels", "validation/images", "validation/labels"]:
                os.makedirs(os.path.join(data_root, sub), exist_ok=True)

            # Copy images + labels into train/val structure
            def copy_split(images, split_name):
                for img_name in images:
                    base = os.path.splitext(img_name)[0]
                    src_img = os.path.join(images_dir, img_name)
                    src_lbl = os.path.join(labels_dir, base + ".txt")
                    shutil.copy2(src_img, os.path.join(data_root, f"{split_name}/images", img_name))
                    if os.path.exists(src_lbl):
                        shutil.copy2(src_lbl, os.path.join(data_root, f"{split_name}/labels", base + ".txt"))

            copy_split(train_imgs, "train")
            copy_split(val_imgs, "validation")

            # Create YAML file (Colab style)
            yaml_data = {
                "path": data_root,
                "train": "train/images",
                "val": "validation/images",
                "nc": nc,
                "names": classes
            }

            yaml_name = f"data_{ts}.yaml"
            yaml_path = os.path.join(CONFIGS_DIR, yaml_name)
            with open(yaml_path, "w", encoding="utf-8") as f:
                yaml.dump(yaml_data, f, sort_keys=False)

            return Response({
                "message": "data.yaml created successfully",
                "yaml_path": yaml_path,
                "yaml_content": yaml_data,
                "extracted_dir": extract_dir
            })

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class TrainModelView(APIView):
    """
    Train YOLO model using the generated YAML.
    """
    def post(self, request):
        yaml_path = request.data.get("yaml_path")
        if not yaml_path or not os.path.exists(yaml_path):
            return Response({"error": "yaml_path invalid or missing"}, status=status.HTTP_400_BAD_REQUEST)

        epochs = int(request.data.get("epochs", 50))
        imgsz = int(request.data.get("imgsz", 640))
        batch = int(request.data.get("batch", 16))

        # auto-select device
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        run_name = os.path.splitext(os.path.basename(yaml_path))[0]
        run_dir = os.path.join(RUNS_DIR, run_name)

        model = YOLO("yolov8n-seg.pt")
        model.train(data=yaml_path, epochs=epochs, imgsz=imgsz, batch=batch, device=device, project=run_dir)

        return Response({
            "message": "Training completed",
            "device": device,
            "run_dir": run_dir
        })


class AddCameraView(APIView):
    """
    Add a new camera for YOLO detection/segmentation.
    """
    
    def post(self, request):
        try:
            name = request.data.get("name")
            url = request.data.get("url")
            camera_type = request.data.get("camera_type", "rtsp")
            model_path = request.data.get("model_path")
            img_size = int(request.data.get("img_size", 640))

            if not all([name, url, model_path]):
                return Response(
                    {"error": "Missing required fields: name, url, model_path"}, 
                    status=status.HTTP_400_BAD_REQUEST
                )

            if not os.path.exists(model_path):
                return Response(
                    {"error": f"Model file not found: {model_path}"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Create camera config in MongoDB
            camera_id = mongo_manager.create_camera_config(
                name=name,
                url=url,
                camera_type=camera_type,
                model_path=model_path,
                img_size=img_size
            )

            # Add camera to manager
            camera_manager.add_camera(
                camera_id=camera_id,
                name=name,
                url=url,
                camera_type=camera_type,
                model_path=model_path,
                img_size=img_size
            )

            return Response({
                "camera_id": camera_id,
                "status": "started",
                "message": f"Camera '{name}' started successfully"
            }, status=status.HTTP_201_CREATED)

        except Exception as e:
            return Response(
                {"error": f"Failed to start camera: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class ListCamerasView(APIView):
    """
    List all active cameras from MongoDB.
    """
    
    def get(self, request):
        try:
            # Get cameras from MongoDB
            db_cameras = mongo_manager.get_all_cameras()
            
            # Get runtime status from camera manager
            runtime_cameras = camera_manager.list_cameras()
            runtime_status = {cam['camera_id']: cam for cam in runtime_cameras}
            
            cameras_data = []
            for db_cam in db_cameras:
                runtime_info = runtime_status.get(db_cam['camera_id'], {})
                cameras_data.append({
                    "camera_id": db_cam['camera_id'],
                    "name": db_cam['name'],
                    "url": db_cam['url'],
                    "camera_type": db_cam['camera_type'],
                    "model_path": db_cam['model_path'],
                    "img_size": db_cam['img_size'],
                    "is_active": db_cam['is_active'],
                    "created_at": db_cam['created_at'],
                    "runtime_active": runtime_info.get('active', False),
                    "has_frame": runtime_info.get('has_frame', False),
                    "stats": runtime_info.get('stats', {})
                })
            
            return Response(cameras_data)
        except Exception as e:
            return Response(
                {"error": f"Failed to list cameras: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class StopCameraView(APIView):
    """
    Stop and remove a camera.
    """
    
    def post(self, request, camera_id):
        try:
            # Update MongoDB
            success = mongo_manager.update_camera_status(camera_id, False)
            if not success:
                return Response(
                    {"error": "Camera not found in database"}, 
                    status=status.HTTP_404_NOT_FOUND
                )
            
            # Stop in camera manager
            success = camera_manager.stop_camera(camera_id)
            if not success:
                return Response(
                    {"error": "Camera not found in runtime manager"}, 
                    status=status.HTTP_404_NOT_FOUND
                )

            return Response({
                "camera_id": camera_id,
                "status": "stopped",
                "message": "Camera stopped successfully"
            })

        except Exception as e:
            return Response(
                {"error": f"Failed to stop camera: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class StreamCameraView(APIView):
    """
    Stream camera feed with YOLO detections.
    """
    
    def get(self, request, camera_id):
        def generate_frames():
            while True:
                frame = camera_manager.get_camera_frame(camera_id)
                if frame:
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
                else:
                    # Send a blank frame or wait
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" + b"\r\n")
                time.sleep(0.033)  # ~30 FPS
        
        return StreamingHttpResponse(
            generate_frames(),
            content_type="multipart/x-mixed-replace; boundary=frame"
        )


class GetDetectionsView(APIView):
    """
    Get latest detections from camera and log to MongoDB.
    """
    
    def get(self, request, camera_id):
        try:
            detections = camera_manager.get_camera_detections(camera_id)
            frame_available = camera_manager.get_camera_frame(camera_id) is not None
            
            # Log detections to MongoDB if there are any
            if detections:
                mongo_manager.log_detections(camera_id, detections, time.time())
                mongo_manager.update_camera_stats(camera_id, detection_count=len(detections))
            
            return Response({
                "camera_id": camera_id,
                "detections": detections,
                "timestamp": time.time(),
                "frame_available": frame_available
            })
            
        except Exception as e:
            return Response(
                {"error": f"Failed to get detections: {str(e)}"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class GetCameraStatsView(APIView):
    """
    Get camera statistics from MongoDB and runtime.
    """
    
    def get(self, request, camera_id):
        try:
            # Get runtime stats
            runtime_stats = camera_manager.get_camera_stats(camera_id)
            if not runtime_stats:
                return Response(
                    {"error": "Camera not found in runtime"}, 
                    status=status.HTTP_404_NOT_FOUND
                )
            
            # Get MongoDB stats
            db_stats = mongo_manager.get_camera_stats(camera_id)
            camera_config = mongo_manager.get_camera_config(camera_id)
            
            if not camera_config:
                return Response(
                    {"error": "Camera not found in database"}, 
                    status=status.HTTP_404_NOT_FOUND
                )
            
            stats_data = {
                "camera_id": camera_id,
                "name": camera_config['name'],
                "runtime_stats": runtime_stats,
                "database_stats": db_stats if db_stats else {}
            }
            
            return Response(stats_data)
            
        except Exception as e:
            return Response(
                {"error": f"Failed to get camera stats: {str(e)}"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class GetDetectionHistoryView(APIView):
    """
    Get detection history for a camera from MongoDB.
    """
    
    def get(self, request, camera_id):
        try:
            hours = int(request.GET.get('hours', 24))  # Default to last 24 hours
            limit = int(request.GET.get('limit', 100))  # Default to 100 records
            
            detection_logs = mongo_manager.get_detection_history(camera_id, hours, limit)
            
            # Convert ObjectId to string for JSON serialization
            logs_data = []
            for log in detection_logs:
                log_data = {
                    "timestamp": log['timestamp'],
                    "frame_timestamp": log['frame_timestamp'],
                    "detections": log['detections'],
                    "detection_count": len(log['detections'])
                }
                logs_data.append(log_data)
            
            return Response({
                "camera_id": camera_id,
                "time_period_hours": hours,
                "total_detections": len(logs_data),
                "detection_history": logs_data
            })
            
        except Exception as e:
            return Response(
                {"error": f"Failed to get detection history: {str(e)}"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class HealthCheckView(APIView):
    """
    Health check endpoint.
    """
    
    def get(self, request):
        db_cameras = mongo_manager.get_all_cameras()
        runtime_cameras = camera_manager.list_cameras()
        mongo_healthy = mongo_manager.health_check()
        
        return Response({
            "status": "healthy",
            "timestamp": time.time(),
            "database_cameras": len(db_cameras),
            "runtime_cameras": len(runtime_cameras),
            "mongodb_connected": mongo_healthy
        })
    

def video_feed(request):
    return StreamingHttpResponse(mjpeg_generator(),
        content_type="multipart/x-mixed-replace; boundary=frame")


def detections_json(request):
    return JsonResponse({"detections": detections})


# Legacy functions for backward compatibility
def mjpeg_generator(camera_id):
    """Legacy generator function for backward compatibility"""
    while True:
        frame = camera_manager.get_camera_frame(camera_id)
        if frame:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.05)


def detections():
    """Legacy detections function for backward compatibility"""
    cameras = camera_manager.list_cameras()
    if cameras:
        first_camera_id = cameras[0]['camera_id']
        return camera_manager.get_camera_detections(first_camera_id)
    return []