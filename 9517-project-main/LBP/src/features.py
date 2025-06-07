import numpy as np
from skimage.feature import local_binary_pattern
from .config import radius, n_points, METHOD

def extract_lbp_features(image):
    lbp = local_binary_pattern(image, n_points, radius, METHOD)
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist
