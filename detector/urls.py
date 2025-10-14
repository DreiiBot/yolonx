from django.urls import path
from . import views
from .views import CreateYAMLFromZipView, TrainModelView

urlpatterns = [
    path("video_feed/", views.video_feed, name="video_feed"),
    path("detections/", views.detections_json, name="detections"),
    path("create_yaml/", CreateYAMLFromZipView.as_view(), name="create_yaml"),
    path("train/", TrainModelView.as_view(), name="train"),
]
