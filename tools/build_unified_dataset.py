import shutil
from pathlib import Path
import random
import yaml

# ---- SETTINGS ----
CATEGORIES = ["AdminForm", "BookCover", "Invoice", "BusinessCard", "Newspaper"]
# assumes each category has:  <Category>/images/train/*.jpg  and  <Category>/labels/train/*.txt
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEST = Path("unified_dataset")  # output root
VAL_FRACTION = 0.2              # 20% for validation
RANDOM_SEED = 42
# If True: only print what would happen (no copy/remove). Set to False to actually do it.
DRY_RUN = False

# ---- CLASSES (keep your 12 names here in order) ----
NAMES = [
    "Header",
    "Title",
    "Text",
    "Table",
    "Image",
    "Footer",
    "Stamp or Signature",
    "Caption",
    "Keyvalue",
    "List-item",
    "Check-box",
    "Formulas",
]

def safe_mkdir(p: Path):
    if not p.exists():
        if not DRY_RUN:
            p.mkdir(parents=True, exist_ok=True)

def main():
    random.seed(RANDOM_SEED)

    # Dest folders
    images_train = DEST / "images" / "train"
    images_val = DEST / "images" / "val"
    labels_train = DEST / "labels" / "train"
    labels_val = DEST / "labels" / "val"
    for d in [images_train, images_val, labels_train, labels_val]:
        safe_mkdir(d)

    matched_pairs = []  # list of (img_path, label_path, new_stem)

    print("Scanning categories...\n")
    for cat in CATEGORIES:
        cat_root = Path(cat)
        # expected structure
        img_dir = cat_root / "images" / "train"
        lbl_dir = cat_root / "labels" / "train"

        if not img_dir.exists() or not lbl_dir.exists():
            print(f"[WARN] Skipping {cat}: expected {img_dir} and {lbl_dir}")
            continue

        # Build set of label stems
        label_stems = {p.stem for p in lbl_dir.glob("*.txt")}

        # Iterate images and keep only ones that have a matching label
        for img in img_dir.iterdir():
            if img.suffix.lower() not in IMAGE_EXTS:
                continue
            if img.stem in label_stems:
                lbl = lbl_dir / f"{img.stem}.txt"
                # avoid name collisions by prefixing category to the new stem
                new_stem = f"{cat}_{img.stem}"
                matched_pairs.append((img, lbl, new_stem))
            else:
                # optionally delete unmatched images in-place
                print(f"[UNMATCHED] {img} has no label, will be ignored{' (deleted)' if not DRY_RUN else ''}")
                # If you really want to delete unmatched originals, uncomment below:
                # if not DRY_RUN:
                #     try: img.unlink()
                #     except Exception as e: print(f"  delete failed: {e}")

    print(f"\nFound {len(matched_pairs)} matched (image,label) pairs across categories.")

    # Shuffle and split
    random.shuffle(matched_pairs)
    val_count = int(len(matched_pairs) * VAL_FRACTION)
    val_set = set(matched_pairs[:val_count])

    # Copy files into unified_dataset, renaming with category prefix
    copied_train, copied_val = 0, 0
    for img, lbl, new_stem in matched_pairs:
        if (img, lbl, new_stem) in val_set:
            img_out = images_val / (new_stem + img.suffix.lower())
            lbl_out = labels_val / (new_stem + ".txt")
            split = "val"
        else:
            img_out = images_train / (new_stem + img.suffix.lower())
            lbl_out = labels_train / (new_stem + ".txt")
            split = "train"

        if DRY_RUN:
            print(f"[{split.upper()}] {img} -> {img_out}")
            print(f"[{split.upper()}] {lbl} -> {lbl_out}")
        else:
            shutil.copy2(img, img_out)
            shutil.copy2(lbl, lbl_out)
            if split == "train":
                copied_train += 1
            else:
                copied_val += 1

    # Write data.yaml
    data_yaml = {
        "path": str(DEST.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(NAMES),
        "names": {i: n for i, n in enumerate(NAMES)},
    }
    if DRY_RUN:
        print("\n[data.yaml that will be written]:")
        print(yaml.dump(data_yaml, allow_unicode=True, sort_keys=False))
    else:
        with open(DEST / "data.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(data_yaml, f, allow_unicode=True, sort_keys=False)

    if not DRY_RUN:
        print(f"\nDone.\nCopied TRAIN pairs: {copied_train}\nCopied VAL pairs: {copied_val}")
        print(f"Unified dataset at: {DEST.resolve()}")
        print(f"data.yaml at: {DEST.resolve() / 'data.yaml'}")
    else:
        print("\n[DRY RUN] Nothing was copied or deleted. Set DRY_RUN=False to execute.")

if __name__ == "__main__":
    main()
