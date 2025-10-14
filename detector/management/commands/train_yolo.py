
### FILE: yolo_app/management/commands/train_yolo.py
from django.core.management.base import BaseCommand
from detector.train import train_model


class Command(BaseCommand):
    help = 'Train a YOLO model (no database)'


def add_arguments(self, parser):
    parser.add_argument('--data', required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)


def handle(self, *args, **options):
    data = options['data']
    epochs = options['epochs']
    imgsz = options['imgsz']
    batch = options['batch']
    train_model(data, epochs, imgsz, batch)