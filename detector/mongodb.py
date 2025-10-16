import pymongo
from django.conf import settings
import uuid
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class MongoDBManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        try:
            # MongoDB connection
            self.client = pymongo.MongoClient(
                "mongodb://localhost:27017/",
                serverSelectionTimeoutMS=5000
            )
            self.db = self.client["yolonx"]
            
            # Collections
            self.camera_configs = self.db["camera_configs"]
            self.detection_logs = self.db["detection_logs"]
            self.camera_stats = self.db["camera_stats"]
            
            # Create indexes
            self.camera_configs.create_index("camera_id", unique=True)
            self.detection_logs.create_index([("camera_id", 1), ("timestamp", -1)])
            self.detection_logs.create_index([("timestamp", -1)])
            
            logger.info("MongoDB connection established successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    def create_camera_config(self, name, url, camera_type, model_path, img_size=640):
        """Create a new camera configuration"""
        camera_id = str(uuid.uuid4())[:12]
        camera_data = {
            "camera_id": camera_id,
            "name": name,
            "url": url,
            "camera_type": camera_type,
            "model_path": model_path,
            "img_size": img_size,
            "is_active": True,
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }
        
        result = self.camera_configs.insert_one(camera_data)
        return camera_id
    
    def get_camera_config(self, camera_id):
        """Get camera configuration by ID"""
        return self.camera_configs.find_one({"camera_id": camera_id, "is_active": True})
    
    def get_all_cameras(self):
        """Get all active cameras"""
        return list(self.camera_configs.find({"is_active": True}))
    
    def update_camera_status(self, camera_id, is_active):
        """Update camera active status"""
        result = self.camera_configs.update_one(
            {"camera_id": camera_id},
            {"$set": {"is_active": is_active, "updated_at": datetime.now()}}
        )
        return result.modified_count > 0
    
    def log_detections(self, camera_id, detections, frame_timestamp):
        """Log detections to database"""
        log_entry = {
            "camera_id": camera_id,
            "detections": detections,
            "timestamp": datetime.now(),
            "frame_timestamp": frame_timestamp
        }
        return self.detection_logs.insert_one(log_entry)
    
    def get_detection_history(self, camera_id, hours=24, limit=100):
        """Get detection history for a camera"""
        from datetime import timedelta
        time_threshold = datetime.now() - timedelta(hours=hours)
        
        return list(self.detection_logs.find({
            "camera_id": camera_id,
            "timestamp": {"$gte": time_threshold}
        }).sort("timestamp", -1).limit(limit))
    
    def update_camera_stats(self, camera_id, frames_processed=0, detection_count=0):
        """Update camera statistics"""
        self.camera_stats.update_one(
            {"camera_id": camera_id},
            {
                "$set": {"last_activity": datetime.now()},
                "$inc": {
                    "frames_processed": frames_processed,
                    "detection_count": detection_count
                }
            },
            upsert=True
        )
    
    def get_camera_stats(self, camera_id):
        """Get camera statistics"""
        return self.camera_stats.find_one({"camera_id": camera_id})
    
    def health_check(self):
        """Check MongoDB connection health"""
        try:
            self.client.admin.command('ping')
            return True
        except Exception as e:
            logger.error(f"MongoDB health check failed: {e}")
            return False

# Global MongoDB manager instance
mongo_manager = MongoDBManager()