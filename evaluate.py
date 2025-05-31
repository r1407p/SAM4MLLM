import numpy as np
from tqdm import tqdm

from data_utils import Dataset, RefCOCODataset

def example_inference(img: np.ndarray, query):
    return np.zeros(img.shape[:2], dtype=np.uint8)

def IoU(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    return np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0.0

def evaluate(inference_fn, dataset: Dataset):
    total_iou = 0.0
    count = 0

    for idx in tqdm(range(len(dataset))):
        img, query, mask_img, bbox = dataset[idx]
        img = np.array(img)
        mask = np.array(mask_img) > 0 
        
        pred_mask = inference_fn(img, query)
        pred_mask = pred_mask > 0  

        iou = IoU(mask, pred_mask)
        total_iou += iou
        count += 1

    average_iou = total_iou / count if count > 0 else 0.0
    return average_iou

if __name__ == "__main__":
    dataset = RefCOCODataset(split='test')
    average_iou = evaluate(example_inference, dataset)
    print(f"Average IoU: {average_iou:.4f}")