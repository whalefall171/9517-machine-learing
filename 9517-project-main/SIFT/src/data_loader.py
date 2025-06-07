# src/data_loader.py

import os
import cv2
import numpy as np
from glob import glob
from src.config import RANDOM_STATE, MAX_SAMPLES


def load_and_split_dataset(dataset_path, test_size=0.2, sample_ratio=1.0):
    """
    Load dataset from folders and split it into train/test sets per class.
    Args:
        dataset_path: path to dataset (e.g., Aerial_Landscapes/class_name/*.jpg)
        test_size: portion of test data to be held out
        sample_ratio: percentage of each class to keep (for subsampling)
    Returns:
        (train_images, train_labels), (test_images, test_labels), class_names
    """
    def _get_sorted_class_list(path):
        return sorted(os.listdir(path))

    def _load_images_from_paths(paths, label):
        imgs, labels = [], []
        for img_path in paths:
            img = cv2.imread(img_path)
            if img is not None:
                imgs.append(img)
                labels.append(label)
            else:
                print(f"[Warning] Failed to read image: {img_path}")
        return imgs, labels

    train_images, train_labels = [], []
    test_images, test_labels = [], []
    classes = _get_sorted_class_list(dataset_path)

    for class_id, class_name in enumerate(classes):
        class_dir = os.path.join(dataset_path, class_name)
        all_paths = glob(os.path.join(class_dir, '*.jpg'))

        if not all_paths:
            continue  # Skip empty class folders

        # Shuffle image paths reproducibly
        np.random.seed(RANDOM_STATE)
        np.random.shuffle(all_paths)

        # Subsample image paths
        usable_paths = all_paths[:max(int(len(all_paths) * sample_ratio), 1)]

        # Split into training and testing
        split_point = int(len(usable_paths) * (1 - test_size))
        train_paths = usable_paths[:split_point]
        test_paths = usable_paths[split_point:] if split_point < len(usable_paths) else []

        # Load images
        tr_imgs, tr_labels = _load_images_from_paths(train_paths, class_id)
        te_imgs, te_labels = _load_images_from_paths(test_paths, class_id)

        train_images.extend(tr_imgs)
        train_labels.extend(tr_labels)
        test_images.extend(te_imgs)
        test_labels.extend(te_labels)

    return (train_images, train_labels), (test_images, test_labels), classes