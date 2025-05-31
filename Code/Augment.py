import albumentations as A
import cv2
import os
import random
import shutil

NUM_AUGMENTATIONS = 8
project_root = os.path.dirname(os.path.abspath(__file__))

input_images_dir = os.path.join(project_root, "raw_data", "images")
input_labels_dir = os.path.join(project_root, "raw_data", "labels")
output_dir = os.path.join(project_root, "augmented_data")

os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.8),
    A.Rotate(limit=35, p=0.7),
    A.RandomSnow(p=0.3),
    A.Blur(blur_limit=3, p=0.2),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_ids']))

for img_name in os.listdir(input_images_dir):
    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    img_path = os.path.join(input_images_dir, img_name)
    label_path = os.path.join(input_labels_dir, img_name.rsplit('.', 1)[0] + ".txt")

    if not os.path.exists(label_path):
        continue

    image = cv2.imread(img_path)
    if image is None:
        continue

    with open(label_path, 'r') as f:
        orig_annotations = [list(map(float, line.strip().split())) for line in f]

    for aug_idx in range(NUM_AUGMENTATIONS):
        transformed = transform(
            image=image,
            bboxes=[bbox[1:] for bbox in orig_annotations],
            class_ids=[int(bbox[0]) for bbox in orig_annotations]
        )

        base_name = os.path.splitext(img_name)[0]
        aug_img_name = f"aug_{base_name}_{aug_idx + 1}.jpg"
        aug_label_name = f"aug_{base_name}_{aug_idx + 1}.txt"

        cv2.imwrite(
            os.path.join(output_dir, "images", aug_img_name),
            transformed['image']
        )

        with open(os.path.join(output_dir, "labels", aug_label_name), 'w') as f:
            for bbox, class_id in zip(transformed['bboxes'], transformed['class_ids']):
                f.write(f"{class_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")

dataset_root = os.path.join(project_root, "augmented_data")
final_dataset_dir = os.path.join(project_root, "final_dataset")
splits = {'train': 0.7, 'val': 0.15, 'test': 0.15}

for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(final_dataset_dir, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(final_dataset_dir, split, 'labels'), exist_ok=True)

all_images = []
for img in os.listdir(os.path.join(dataset_root, "images")):
    if img.startswith("aug_") or img.endswith(('.jpg', '.png')):
        all_images.append(img)

random.shuffle(all_images)
total = len(all_images)
train_idx = int(total * splits['train'])
val_idx = train_idx + int(total * splits['val'])

for i, img_name in enumerate(all_images):
    if i < train_idx:
        split = 'train'
    elif i < val_idx:
        split = 'val'
    else:
        split = 'test'

    src_img = os.path.join(dataset_root, "images", img_name)
    dst_img = os.path.join(final_dataset_dir, split, "images", img_name)
    shutil.copy(src_img, dst_img)

    label_name = img_name.rsplit('.', 1)[0] + ".txt"
    src_label = os.path.join(dataset_root, "labels", label_name)
    dst_label = os.path.join(final_dataset_dir, split, "labels", label_name)
    if os.path.exists(src_label):
        shutil.copy(src_label, dst_label)