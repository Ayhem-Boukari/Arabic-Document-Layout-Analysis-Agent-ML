# Arabic Document Layout Analysis Agent (ML-only)

A modern computer vision project for detecting document layout elements (titles, paragraphs, tables, figures, etc.) using YOLOv8.


## Overview
This is the ML-only repository (no FastAPI app). It contains training, evaluation, inference notebooks/scripts, results, and configs reflecting the real experiments reported in the internship report. For the web service, see the separate FastAPI repository.

### Motivation
- Automate document analysis for digitization and information extraction.
- Enable downstream tasks such as OCR, semantic parsing, and metadata extraction.
- Build a reusable, production-ready pipeline with an inference API.

## Repository Structure

```
.
├── data/
│   └── data.yaml                 # Dataset config (paths and class names)
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_training.ipynb
│   ├── 03_evaluation.ipynb
│   ├── 04_inference_and_api.ipynb
│   └── 05_dataset_unification_and_active_learning.ipynb
├── results/
│   ├── baseline_results.txt      # Baseline (Iteration 1) metrics
│   └── final_results.txt         # Final (Iteration 2) metrics
├── src/
│   ├── __init__.py
│   ├── config.py                 # Defaults updated to final settings (imgsz=1280, epochs=80, patience=30)
│   ├── evaluate.py
│   ├── infer.py
│   ├── train.py
│   └── utils/
│       ├── __init__.py
│       └── visualize.py
├── tools/
│   ├── convert_pdfs.py
│   ├── dataset.py
│   ├── build_unified_dataset.py
│   ├── add_new_annotations.py
│   ├── select_uncertain.py
│   ├── make_overlays_from_labels.py
│   └── synthesize_longtail.py
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
```

## Dataset Summary
- Format: YOLO (images/ and labels/ with .txt files, or data.yaml)
- Classes: configurable in `data/data.yaml` (12-class taxonomy used in report)
- Sizes (from the report):
  - Baseline training: 190 images
  - Final training: 205 images (after +15 hard pages)
  - Validation: 47 images (locked), 814 instances

> Tip: Place raw PDFs/images into `data/raw/`, preprocess into YOLO-format under `data/processed/`.

## Pipeline
- Data ingestion ➜ Preprocessing ➜ YOLOv8 training ➜ Evaluation ➜ Export weights ➜ Inference



### Training/Inference Parameters

- Baseline (Iteration 1): YOLOv8s, imgsz=640, ~50 epochs, patience=20, batch=16, lr0=0.01, device=cpu.
- Final (Iteration 2): YOLOv8s, imgsz=1280, epochs=80, patience=30, batch=16, lr0=0.01, device=cpu.

Final defaults are reflected in `src/config.py` so training/inference uses the validated settings by default.

| Setting  | Baseline (Iter. 1) | Final (Iter. 2) |
|---------:|--------------------:|-----------------:|
| Model    | YOLOv8s             | YOLOv8s          |
| imgsz    | 640                 | 1280             |
| epochs   | 50                  | 80               |
| patience | 20                  | 30               |
| batch    | 16                  | 16               |
| lr0      | 0.01                | 0.01             |
| device   | cpu                 | cpu              |

### Post-processing thresholds

Per the report, we used class-wise thresholds and light rules to reduce visible noise. A reference configuration is provided in `data/thresholds.json`:

```
{
  "nms_iou": 0.50,
  "min_area_frac": { "Table": 0.0008, "Image": 0.0008 },
  "conf": {
    "Title": 0.45, "Text": 0.40, "Caption": 0.35,
    "Table": 0.40, "Image": 0.50,
    "Footer": 0.30, "Stamp/Signature": 0.45,
    "List-item": 0.35, "Keyvalue": 0.30, "Check-box": 0.30
  }
}
```

You can load this in your inference pipeline to filter predictions post-NMS.

### Performance notes (from the report)
- CPU inference latency: ~0.6 s/image at 1280 px (YOLOv8s)
- Strong classes: Image (~0.785), Caption (~0.770), Table (~0.565)
- Challenging classes (scarce): Footer (~0.173), Keyvalue (~0.0047), Check-box (~0.0)

### Model checkpoints
- This ML-only repo does not include weights. Place your trained weights under `models/` (git-ignored) and update paths accordingly.
- Example usage: `python -c "from src.evaluate import evaluate; evaluate('models/best.pt')"`


## How to Use

### 1) Install dependencies
Use a virtual environment (recommended).

```powershell
python -m venv venv ; .\venv\Scripts\Activate.ps1 ; pip install -r requirements.txt
```

If you use CUDA, please install a torch build compatible with your GPU drivers from pytorch.org before installing requirements.

### 2) Train the model
Ensure your `data/data.yaml` points to images/labels and defines class names.

```powershell
python -m src.train
```

Advanced options: edit defaults in `src/config.py` (epochs, batch, imgsz, device).

### 3) Evaluate
Provide the path to your trained weights (e.g., `runs/detect/train/weights/best.pt`).

```powershell
python -c "from src.evaluate import evaluate; evaluate('models/best.pt')"
```

### 4) Inference (script)

```powershell
python -c "from src.infer import infer; infer('models/best.pt', 'data/processed/images/val')"
```

<!-- API instructions are omitted in this ML-only repository. -->

## Lessons Learned
- Data quality and label consistency are crucial for detection performance.
- Balanced classes help stabilize training and improve recall.
- Proper evaluation (mAP@50-95) and ablation studies guide model selection.

## Future Work
- Multi-lingual and domain adaptation for diverse document types.
- Post-processing for structure reconstruction (reading order, regions).
- Export ONNX/TensorRT for faster inference.

## Author
- Name: Ayhem Boukari
- Email: ayhem.boukari@enicar.ucar.tn
- LinkedIn: https://www.linkedin.com/in/ayhem-boukari-3889b528b/

## License
This project is licensed under the MIT License. See LICENSE for details.

## Push to GitHub (quick start)

```powershell
# Initialize git (if not already)
git init

# Set your main branch name
git branch -M main

# Add files and commit
git add .
git commit -m "Initial commit: Document Layout Detection Using YOLOv8"

# Add your GitHub remote (replace URL with your repo)
git remote add origin https://github.com/<your-username>/Arabic-Document-Layout-Analysis-Agent.git

# Push
git push -u origin main
```
