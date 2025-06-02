import numpy as np
from PIL import Image
from tqdm import tqdm

from data_utils import Dataset, RefCOCODataset
from sam4mllm_infer import inference as sam4mllm_inference

def example_inference(img: Image.Image, query):
    return np.zeros(img.size[:2], dtype=np.uint8)

def IoU(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    return np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0.0

def evaluate(inference_fn, dataset: Dataset, attempt=3):
    total_iou = 0.0
    count = 0

    for idx in tqdm(range(len(dataset))):
        img, query, mask_img, bbox = dataset[idx]
        mask = np.array(mask_img) > 0 
        
        for _ in range(attempt):
            try:
                pred_mask = inference_fn(img, query)
                if pred_mask is not None:
                    break
            except Exception as e:
                continue
        else:
            print(f"Failed to get prediction for index {idx}")
            continue
        
        pred_mask = pred_mask > 0  
        
        mask_img.save('output_mask_gt.png')

        iou = IoU(mask, pred_mask)
        total_iou += iou
        count += 1

    average_iou = total_iou / count if count > 0 else 0.0
    return average_iou

if __name__ == "__main__":
    dataset = RefCOCODataset(split='testA')
    inference_fn = lambda img, query: sam4mllm_inference(img, query, eval=True)
    average_iou = evaluate(inference_fn, dataset)
    print(f"Average IoU: {average_iou:.4f}")