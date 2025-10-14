import os
import random
import zipfile
import shutil
import datetime
import yaml
import torch
from ultralytics import YOLO
from rest_framework.views import APIView
from rest_framework import status
from rest_framework.response import Response
from django.conf import settings
from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse, HttpResponse
from .yolo_stream import mjpeg_generator, detections

BASE_STORAGE = os.path.join(settings.BASE_DIR, "yolo_storage")
DATASETS_DIR = os.path.join(BASE_STORAGE, "datasets")    # optional place if you want to keep copies
CONFIGS_DIR = os.path.join(BASE_STORAGE, "configs")      # YAMLs will be stored here (inside project)
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
    
    Example JSON:
    {
        "dataset_path": "C:\\Users\\ASUS\\Downloads\\project.zip"
    }
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

        model = YOLO("yolov8n.pt")
        model.train(data=yaml_path, epochs=epochs, imgsz=imgsz, batch=batch, device=device, project=run_dir)

        return Response({
            "message": "Training completed",
            "device": device,
            "run_dir": run_dir
        })


def video_feed(request):
    return StreamingHttpResponse(mjpeg_generator(),
        content_type="multipart/x-mixed-replace; boundary=frame")


def detections_json(request):
    return JsonResponse({"detections": detections})
