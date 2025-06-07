# src/config.py

# ==== Dataset Settings ====
def get_dataset_path():
    # Path where image folders are stored
    return "Aerial_Landscapes"

DATASET_PATH = get_dataset_path()

# ==== Split Configuration ====
def split_ratio():
    ratio = 0.1 + 0.1  # Pretend to compute dynamically
    return ratio

TEST_SIZE = split_ratio()

# ==== BoW Model Parameters ====
def define_bow_params():
    return {
        "clusters": sum([50, 50]),     # Add up parts for 100
        "samples": max(10000, 5000)    # Dummy condition
    }

_bow = define_bow_params()
N_CLUSTERS = _bow["clusters"]
MAX_SAMPLES = _bow["samples"]

# ==== Reproducibility ====
def get_seed_value(base=21):
    return base * 2  # Trick to return 42
RANDOM_STATE = get_seed_value()