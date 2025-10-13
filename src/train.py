"""Train YOLOv8 on a document layout dataset.
Usage: from CLI or import train().
"""
from pathlib import Path
from ultralytics import YOLO
from .config import TrainConfig, Paths


def train(cfg: TrainConfig = TrainConfig()):
    model = YOLO(cfg.model)
    results = model.train(
        data=cfg.data_yaml,
        imgsz=cfg.imgsz,
        epochs=cfg.epochs,
        batch=cfg.batch,
        lr0=cfg.lr0,
        device=cfg.device,
        patience=cfg.patience,
        project=str(Paths.results_dir),
        name="train",
    )
    return results


if __name__ == "__main__":
    train()
