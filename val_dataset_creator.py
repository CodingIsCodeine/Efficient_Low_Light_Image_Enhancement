import os
import random
import shutil

# ---------------- CONFIG ----------------
train_low_dir = "data/train/low"
train_high_dir = "data/train/high"

val_low_dir = "data/val/low"
val_high_dir = "data/val/high"

NUM_VAL_IMAGES = 49
RANDOM_SEED = 42
# ----------------------------------------

# Create validation directories
os.makedirs(val_low_dir, exist_ok=True)
os.makedirs(val_high_dir, exist_ok=True)

# List all training low images
all_images = sorted(os.listdir(train_low_dir))

# Sanity check
assert len(all_images) >= NUM_VAL_IMAGES, "Not enough images for validation split!"

random.seed(RANDOM_SEED)
val_images = random.sample(all_images, NUM_VAL_IMAGES)

for img_name in val_images:
    low_src = os.path.join(train_low_dir, img_name)
    high_src = os.path.join(train_high_dir, img_name)

    low_dst = os.path.join(val_low_dir, img_name)
    high_dst = os.path.join(val_high_dir, img_name)

    shutil.copy(low_src, low_dst)
    shutil.copy(high_src, high_dst)

print(f"✅ Validation set created with {NUM_VAL_IMAGES} paired images.")
