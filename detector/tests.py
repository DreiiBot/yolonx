from djongo import models
import uuid
from django.utils import timezone

# Create your tests here.
class CameraConfig(models.Model):
    CAMERA_TYPES = [
        ('rtsp', 'RTSP'),
        ('http', 'HTTP'),
        ('udp', 'UDP'),
        ('tcp', 'TCP'),
        ('webcam', 'WEBCAM'),
    ]

    CAMERA_TYPES = [
        ('rtsp', 'RTSP'),
        ('http', 'HTTP'),
        ('udp', 'UDP'),
        ('tcp', 'TCP'),
        ('webcam', 'WEBCAM'),
    ]
    
    _id = models.ObjectIdField()
    camera_id = models.CharField(max_length=50, unique=True)
    name = models.CharField(max_length=100)
    url = models.TextField()
    camera_type = models.CharField(max_length=10, choices=CAMERA_TYPES)
    model_path = models.CharField(max_length=200)
    img_size = models.IntegerField(default=640)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'camera_configs'
        verbose_name = 'Camera Configuration'
        verbose_name_plural = 'Camera Configurations'
    
    def __str__(self):
        return f"{self.name} ({self.camera_type}) - {self.camera_id}"
    
    def save(self, *args, **kwargs):
        if not self.camera_id:
            self.camera_id = str(uuid.uuid4())[:12]
        super().save(*args, **kwargs)

class CameraStats(models.Model):
    _id = models.ObjectIdField()
    camera = models.ForeignKey(CameraConfig, on_delete=models.CASCADE)
    frames_processed = models.BigIntegerField(default=0)
    detection_count = models.IntegerField(default=0)
    avg_processing_time = models.FloatField(default=0)
    last_activity = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'camera_stats'
        verbose_name = 'Camera Statistics'
        verbose_name_plural = 'Camera Statistics'


class DetectionLog(models.Model):
    _id = models.ObjectIdField()
    camera = models.ForeignKey(CameraConfig, on_delete=models.CASCADE)
    detections = models.JSONField()  # Store detection data as JSON
    timestamp = models.DateTimeField(auto_now_add=True)
    frame_timestamp = models.FloatField()  # Original frame timestamp
    
    class Meta:
        db_table = 'detection_logs'
        verbose_name = 'Detection Log'
        verbose_name_plural = 'Detection Logs'
        indexes = [
            models.Index(fields=['camera', 'timestamp']),
            models.Index(fields=['timestamp']),
        ]