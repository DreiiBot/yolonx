from django.http import StreamingHttpResponse, JsonResponse
from .yolo_stream import mjpeg_generator, detections

def video_feed(request):
    return StreamingHttpResponse(
        mjpeg_generator(),
        content_type="multipart/x-mixed-replace; boundary=frame"
    )

def detections_json(request):
    return JsonResponse({"detections": detections})
