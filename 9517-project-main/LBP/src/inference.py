import joblib
import os

def load_model(model_filename="model.pkl"):
    if os.path.exists(model_filename):
        model = joblib.load(model_filename)
        print(f"Model loaded from {model_filename}")
        return model
    else:
        print(f"{model_filename} not found.")
        return None
