import os
import random
import shutil

train_dir = "data/train"
val_dir = "data/val"

os.makedirs(val_dir, exist_ok=True)

val_split = 0.2

for class_name in os.listdir(train_dir):
    class_path = os.path.join(train_dir, class_name)

    if not os.path.isdir(class_path):
        continue

    val_class_path = os.path.join(val_dir, class_name)
    os.makedirs(val_class_path, exist_ok=True)

    images = os.listdir(class_path)
    random.shuffle(images)

    val_count = int(len(images) * val_split)
    val_images = images[:val_count]

    for img in val_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(val_class_path, img)
        shutil.move(src, dst)

print("✅ Data successfully split into train and val folders!")
