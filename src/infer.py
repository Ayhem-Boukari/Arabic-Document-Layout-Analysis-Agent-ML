"""Run inference on images with a YOLOv8 model.
"""
from typing import List, Union
from pathlib import Path
from ultralytics import YOLO
from .config import InferenceConfig


def infer(weights: str, sources: Union[str, List[str]], cfg: InferenceConfig = InferenceConfig()):
    model = YOLO(weights)
    results = model.predict(source=sources, conf=cfg.conf, iou=cfg.iou, imgsz=cfg.imgsz)
    return results


if __name__ == "__main__":
    # Example: infer("models/best.pt", "data/processed/images/val/")
    pass
