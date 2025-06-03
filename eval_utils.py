import pandas as pd
import numpy as np
from PIL import Image

def load_existing_results(result_path):
    try:
        return pd.read_pickle(result_path)
    except FileNotFoundError:
        return {}
    
def example_inference(img: Image.Image, query: str) -> np.ndarray:
    return np.zeros(img.size[:2], dtype=np.uint8)

def IoU(mask1, mask2):
    intersection = get_intersection(mask1, mask2)
    union = get_union(mask1, mask2)
    return intersection / union if union > 0 else 0.0

def get_intersection(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    return np.sum(intersection)

def get_union(mask1, mask2):
    union = np.logical_or(mask1, mask2)
    return np.sum(union)

def get_mask_from_bbox(bbox, img_size):
    mask = np.zeros(img_size, dtype=np.uint8)
    bbox = [int(coord) for coord in bbox]
    x1, y1, x2, y2 = bbox
    mask[y1:y2, x1:x2] = 1
    return mask