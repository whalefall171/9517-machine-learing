import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from .features import extract_lbp_features

def load_lbp_dataset(dataset_path):
    data, labels = [], []

    for label in os.listdir(dataset_path):
        folder = os.path.join(dataset_path, label)
        if not os.path.isdir(folder): continue
        for file in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, file), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (64, 64))
            features = extract_lbp_features(img)
            data.append(features)
            labels.append(label)

    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    return np.array(data), np.array(labels_encoded), le
