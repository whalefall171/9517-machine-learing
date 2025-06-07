# src/sift_bow_extractor.py

import cv2
import numpy as np
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
from src.config import N_CLUSTERS, MAX_SAMPLES, RANDOM_STATE


def extract_color_sift_features(images):
    """
    Extract color SIFT features by processing each RGB channel separately.
    Returns:
        A list of SIFT descriptor arrays, one per image.
    """
    def _process_single_image(img, detector):
        descriptors = []
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for ch in cv2.split(rgb_img):
            _, des = detector.detectAndCompute(ch, None)
            if des is not None:
                descriptors.append(des)
        return np.vstack(descriptors) if descriptors else np.array([])

    sift = cv2.SIFT_create()
    all_descriptors = []

    for img in tqdm(images, desc="Extracting RGB SIFT"):
        desc = _process_single_image(img, sift)
        all_descriptors.append(desc)

    return all_descriptors


def create_visual_vocabulary(descriptors_list, n_clusters=N_CLUSTERS, max_samples=MAX_SAMPLES):
    """
    Cluster descriptors using KMeans to create a visual vocabulary (BoW).
    Returns:
        A fitted MiniBatchKMeans model.
    """
    # Flatten all descriptors
    stacked = [d for d in descriptors_list if len(d) > 0]
    full_descriptors = np.vstack(stacked)

    # Optional downsampling
    if full_descriptors.shape[0] > max_samples:
        np.random.seed(RANDOM_STATE)
        indices = np.random.choice(full_descriptors.shape[0], max_samples, replace=False)
        full_descriptors = full_descriptors[indices]

    # Fit BoW model
    bow_model = MiniBatchKMeans(n_clusters=n_clusters, random_state=RANDOM_STATE)
    bow_model.fit(full_descriptors)

    return bow_model


def extract_bow_features(descriptors_list, kmeans):
    """
    Convert local SIFT descriptors to BoW histograms using the given k-means model.
    Returns:
        A numpy array of BoW vectors (n_samples, n_clusters).
    """
    def _encode_to_histogram(descriptors, model):
        bins = model.n_clusters
        hist = np.zeros(bins, dtype=np.float32)
        if descriptors is not None and len(descriptors) > 0:
            labels = model.predict(descriptors)
            hist[:bins] = np.bincount(labels, minlength=bins).astype(np.float32)
            hist /= (hist.sum() + 1e-7)
        return hist

    return np.array([
        _encode_to_histogram(desc, kmeans)
        for desc in descriptors_list
    ])