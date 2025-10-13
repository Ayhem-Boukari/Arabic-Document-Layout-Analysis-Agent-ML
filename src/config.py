"""Project-wide configuration and paths.
Edit these defaults to match your dataset directories.
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Paths:
    project_root: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = project_root / "data"
    raw_dir: Path = data_dir / "raw"
    processed_dir: Path = data_dir / "processed"
    models_dir: Path = project_root / "models"
    results_dir: Path = project_root / "results"


@dataclass
class TrainConfig:
    # Ultralytics YOLOv8 configs
    model: str = "yolov8s.pt"  # starting checkpoint
    data_yaml: str = str(Paths.data_dir / "data.yaml")
    # Final validated configuration from the internship report
    imgsz: int = 1280
    epochs: int = 80
    batch: int = 16
    lr0: float = 0.01
    device: str = "cpu"  # or "0" for first GPU
    patience: int = 30


@dataclass
class EvalConfig:
    conf: float = 0.25
    iou: float = 0.6


@dataclass
class InferenceConfig:
    conf: float = 0.25
    iou: float = 0.6
    # Match training resolution used for the final model
    imgsz: int = 1280
