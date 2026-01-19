import os
import pickle

def save_object(file_path, obj):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)

def load_object(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)
