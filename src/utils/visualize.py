"""Visualization utilities for YOLOv8 predictions."""
from pathlib import Path
from typing import List


def save_results_images(result_objects, out_dir: Path):
    """Save plotted images from ultralytics results objects to out_dir.
    result_objects: iterable of ultralytics.engine.results.Results
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(result_objects):
        if hasattr(r, "plot"):
            im = r.plot()
            out_path = out_dir / f"pred_{i}.jpg"
            im.save(out_path)
