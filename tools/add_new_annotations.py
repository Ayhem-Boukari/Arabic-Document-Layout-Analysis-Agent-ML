# add_new_annotations.py
from pathlib import Path
import shutil

BASE = Path(__file__).resolve().parent
SRC_IMG = BASE / "new_annotations" / "images"
SRC_LBL = BASE / "new_annotations" / "labels"
DST_IMG = BASE / "images" / "train"
DST_LBL = BASE / "labels" / "train"

ALLOWED_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def next_free_stem(stem: str) -> str:
    """If stem already exists in DST, append _v2, _v3, ..."""
    i = 2
    new = stem
    while (DST_IMG / f"{new}.jpg").exists() or (DST_IMG / f"{new}.png").exists() or (DST_LBL / f"{new}.txt").exists():
        new = f"{stem}_v{i}"
        i += 1
    return new

added, skipped = 0, 0
DST_IMG.mkdir(parents=True, exist_ok=True)
DST_LBL.mkdir(parents=True, exist_ok=True)

for img in sorted(SRC_IMG.iterdir()):
    if img.suffix.lower() not in ALLOWED_IMG_EXT: 
        continue
    stem = img.stem
    lbl = SRC_LBL / f"{stem}.txt"
    if not lbl.exists():
        print(f"[SKIP] No label for {img.name}")
        skipped += 1
        continue

    # Ensure unique stem in destination
    out_stem = stem
    # conflict if same-stem label or image already exists
    if (DST_LBL / f"{stem}.txt").exists() or any((DST_IMG / f"{stem}{ext}").exists() for ext in ALLOWED_IMG_EXT):
        out_stem = next_free_stem(stem)
        print(f"[RENAME] {stem} -> {out_stem}")

    # Normalize image ext to keep original ext
    out_img = DST_IMG / f"{out_stem}{img.suffix.lower()}"
    out_lbl = DST_LBL / f"{out_stem}.txt"

    shutil.copy2(img, out_img)
    shutil.copy2(lbl, out_lbl)
    added += 1

print(f"Done. Added {added} image/label pairs, skipped {skipped}.")
print("You can now delete unified_dataset/new_annotations if you want.")
