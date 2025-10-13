import os
import shutil
import random

# Paths
all_images_dir = 'images'
all_labels_dir = 'labels'
base_output = 'newspaper_yolo'

# Create output folders
for split in ['train', 'val']:
    os.makedirs(os.path.join(base_output, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(base_output, 'labels', split), exist_ok=True)

# Match images with existing labels only
label_files = [f for f in os.listdir(all_labels_dir) if f.endswith('.txt')]
image_label_pairs = []

for label_file in label_files:
    img_file = label_file.replace('.txt', '.jpg')
    img_path = os.path.join(all_images_dir, img_file)
    label_path = os.path.join(all_labels_dir, label_file)

    if os.path.exists(img_path):
        image_label_pairs.append((img_path, label_path))

# Train/val split (80/20)
random.shuffle(image_label_pairs)
split_idx = int(len(image_label_pairs) * 0.8)
train_pairs = image_label_pairs[:split_idx]
val_pairs = image_label_pairs[split_idx:]

# Move files
for img, label in train_pairs:
    shutil.copy(img, os.path.join(base_output, 'images/train'))
    shutil.copy(label, os.path.join(base_output, 'labels/train'))

for img, label in val_pairs:
    shutil.copy(img, os.path.join(base_output, 'images/val'))
    shutil.copy(label, os.path.join(base_output, 'labels/val'))

print("✅ Dataset prepared for YOLOv8 training.")
