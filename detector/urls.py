from django.urls import path
from . import views
from .views import CreateYAMLFromZipView, TrainModelView, AddCameraView, ListCamerasView, StopCameraView, StreamCameraView, GetDetectionHistoryView, GetCameraStatsView, GetDetectionsView, HealthCheckView

urlpatterns = [
    path("video_feed/", views.video_feed, name="video_feed"),
    path("detections/", views.detections_json, name="detections"),
    path("create_yaml/", CreateYAMLFromZipView.as_view(), name="create_yaml"),
    path("train/", TrainModelView.as_view(), name="train"),
    # Camera management endpoints
    path('cameras/add', AddCameraView.as_view(), name='add-camera'),
    path('cameras/list', ListCamerasView.as_view(), name='list-cameras'),
    path('cameras/<str:camera_id>/stop', StopCameraView.as_view(), name='stop-camera'),
    path('cameras/<str:camera_id>/stream', StreamCameraView.as_view(), name='stream-camera'),
    path('cameras/<str:camera_id>/detections', GetDetectionsView.as_view(), name='get-detections'),
    path('cameras/<str:camera_id>/stats', GetCameraStatsView.as_view(), name='get-camera-stats'),
    path('cameras/<str:camera_id>/detection-history', GetDetectionHistoryView.as_view(), name='detection-history'),
    path('health', HealthCheckView.as_view(), name='health-check'),
]
