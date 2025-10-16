import cv2
import time
import threading
import uuid
from ultralytics import YOLO
from collections import deque
import logging

logger = logging.getLogger(__name__)

class CameraManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.cameras = {}
        self.camera_lock = threading.Lock()
        self._initialized = True
        logger.info("Camera Manager initialized")
    
    def build_camera_url(self, camera_type, url):
        """Build proper camera URL based on type"""
        if camera_type == 'webcam':
            return int(url) if url.isdigit() else url
        return url
    
    def add_camera(self, camera_id, name, url, camera_type, model_path, img_size=640):
        """Add a new camera instance"""
        with self.camera_lock:
            if camera_id in self.cameras:
                raise ValueError(f"Camera {camera_id} already exists")
            
            try:
                model = YOLO(model_path)
                is_segmentation = getattr(model, "task", "") == "segment"
                
                camera_instance = {
                    'camera_id': camera_id,
                    'name': name,
                    'url': url,
                    'camera_type': camera_type,
                    'model_path': model_path,
                    'img_size': img_size,
                    'model': model,
                    'is_segmentation': is_segmentation,
                    'latest_frame': None,
                    'detections': [],
                    'frame_lock': threading.Lock(),
                    'active': True,
                    'cap': None,
                    'stats': {
                        'frames_processed': 0,
                        'last_update': time.time()
                    }
                }
                
                self.cameras[camera_id] = camera_instance
                
                # Start worker thread
                thread = threading.Thread(
                    target=self._camera_worker,
                    args=(camera_instance,),
                    daemon=True,
                    name=f"CameraWorker-{camera_id}"
                )
                thread.start()
                
                logger.info(f"Camera {camera_id} ({name}) started with model {model_path}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to add camera {camera_id}: {str(e)}")
                raise
    
    def _camera_worker(self, camera_instance):
        """Worker function for each camera"""
        camera_id = camera_instance['camera_id']
        camera_url = self.build_camera_url(camera_instance['camera_type'], camera_instance['url'])
        model = camera_instance['model']
        is_segmentation = camera_instance['is_segmentation']
        
        cap = None
        reconnect_attempts = 0
        max_reconnect_attempts = 5
        
        while camera_instance['active']:
            try:
                if cap is None or not cap.isOpened():
                    if camera_instance['camera_type'] == 'rtsp':
                        cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
                    elif camera_instance['camera_type'] in ['udp', 'tcp']:
                        cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
                    else:
                        cap = cv2.VideoCapture(camera_url)
                    
                    if not cap.isOpened():
                        logger.warning(f"Camera {camera_id}: Cannot open, attempt {reconnect_attempts + 1}")
                        reconnect_attempts += 1
                        if reconnect_attempts >= max_reconnect_attempts:
                            logger.error(f"Camera {camera_id}: Max reconnection attempts reached")
                            break
                        time.sleep(2)
                        continue
                    
                    reconnect_attempts = 0
                    logger.info(f"Camera {camera_id}: Connected successfully")
                
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Camera {camera_id}: Frame grab failed")
                    cap.release()
                    cap = None
                    time.sleep(1)
                    continue
                
                # Run inference
                results_list = model(frame, imgsz=camera_instance['img_size'], verbose=False)
                
                if results_list:
                    results = results_list[0]
                    
                    # Visualize results
                    annotated = results.plot() if hasattr(results, "plot") else frame
                    
                    if annotated is None or annotated.size == 0:
                        annotated = frame
                    
                    # Encode frame to JPEG
                    success, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    
                    if success:
                        current_detections = self._extract_detections(results, model, is_segmentation)
                        
                        with camera_instance['frame_lock']:
                            camera_instance['latest_frame'] = jpeg.tobytes()
                            camera_instance['detections'] = current_detections
                            camera_instance['stats']['frames_processed'] += 1
                            camera_instance['stats']['last_update'] = time.time()
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Camera {camera_id} worker error: {str(e)}")
                if cap:
                    cap.release()
                    cap = None
                time.sleep(2)
        
        # Cleanup
        if cap:
            cap.release()
        logger.info(f"Camera {camera_id}: Worker stopped")
    
    def _extract_detections(self, results, model, is_segmentation):
        """Extract detections and masks from results"""
        detections = []
        
        # Extract bounding box detections
        if hasattr(results, "boxes") and results.boxes is not None:
            for box in results.boxes:
                detection_data = {
                    "type": "bbox",
                    "label": model.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "bbox": box.xyxy[0].tolist()
                }
                detections.append(detection_data)
        
        # Extract segmentation masks if available
        if is_segmentation and hasattr(results, "masks") and results.masks is not None:
            for idx, mask in enumerate(results.masks.data):
                label = (
                    model.names[int(results.boxes.cls[idx])]
                    if hasattr(results, "boxes") and results.boxes is not None and idx < len(results.boxes.cls)
                    else "object"
                )
                confidence = (
                    float(results.boxes.conf[idx])
                    if hasattr(results, "boxes") and results.boxes is not None and idx < len(results.boxes.conf)
                    else 0.0
                )
                bbox = (
                    results.boxes.xyxy[idx].tolist()
                    if hasattr(results, "boxes") and results.boxes is not None and idx < len(results.boxes.xyxy)
                    else []
                )
                
                detection_data = {
                    "type": "segmentation",
                    "label": label,
                    "confidence": confidence,
                    "bbox": bbox,
                    "mask_points": results.masks.xy[idx].tolist()
                }
                detections.append(detection_data)
        
        return detections
    
    def get_camera_frame(self, camera_id):
        """Get latest frame from camera"""
        with self.camera_lock:
            camera = self.cameras.get(camera_id)
            
        if not camera or not camera['active']:
            return None
        
        with camera['frame_lock']:
            return camera['latest_frame']
    
    def get_camera_detections(self, camera_id):
        """Get latest detections from camera"""
        with self.camera_lock:
            camera = self.cameras.get(camera_id)
            
        if not camera or not camera['active']:
            return []
        
        with camera['frame_lock']:
            return camera['detections'].copy()
    
    def get_camera_stats(self, camera_id):
        """Get camera statistics"""
        with self.camera_lock:
            camera = self.cameras.get(camera_id)
            
        if not camera:
            return None
        
        with camera['frame_lock']:
            return camera['stats'].copy()
    
    def stop_camera(self, camera_id):
        """Stop a camera instance"""
        with self.camera_lock:
            camera = self.cameras.get(camera_id)
            
        if camera:
            camera['active'] = False
            # Remove from cameras dict after a short delay
            threading.Timer(2.0, self._remove_camera, args=[camera_id]).start()
            return True
        return False
    
    def _remove_camera(self, camera_id):
        """Remove camera from dictionary"""
        with self.camera_lock:
            if camera_id in self.cameras:
                del self.cameras[camera_id]
                logger.info(f"Camera {camera_id} removed")
    
    def list_cameras(self):
        """List all active cameras"""
        with self.camera_lock:
            cameras_info = []
            for cam_id, cam in self.cameras.items():
                cameras_info.append({
                    'camera_id': cam_id,
                    'name': cam['name'],
                    'type': cam['camera_type'],
                    'url': cam['url'],
                    'model_path': cam['model_path'],
                    'active': cam['active'],
                    'has_frame': cam['latest_frame'] is not None,
                    'stats': cam['stats']
                })
            return cameras_info

# Global camera manager instance
camera_manager = CameraManager()