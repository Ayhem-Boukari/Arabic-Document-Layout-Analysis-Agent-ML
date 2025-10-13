"""Evaluate a trained YOLOv8 model.
"""
from ultralytics import YOLO
from .config import EvalConfig, Paths
from pathlib import Path


def evaluate(weights: str, data_yaml: str = str(Paths.data_dir / "data.yaml"), cfg: EvalConfig = EvalConfig()):
    model = YOLO(weights)
    metrics = model.val(data=data_yaml, conf=cfg.conf, iou=cfg.iou, project=str(Paths.results_dir), name="val")
    return metrics


if __name__ == "__main__":
    # Example: evaluate("models/best.pt")
    pass
