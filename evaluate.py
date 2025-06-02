import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser

from data_utils import Dataset, RefCOCODataset
from sam4mllm_infer import inference as sam4mllm_inference

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

def evaluate(inference_fn, dataset: RefCOCODataset, attempt=3):
    total_union = 0.0
    total_intersection = 0.0
    total_iou = 0.0
    count = 0
    result = {}

    for idx in tqdm(range(len(dataset))):
        data_id, img, query, mask_img, bbox = dataset[idx]
        mask = np.array(mask_img) > 0 
        
        for _ in range(attempt):
            try:
                pred_mask = inference_fn(img, query)
                result[data_id] = pred_mask
                if pred_mask is not None:
                    break
            except Exception as e:
                continue
        else:
            print(f"Failed to get prediction for index {idx}")
            result[data_id] = None
            continue
        
        pred_mask = pred_mask > 0  
        
        mask_img.save('output_mask_gt.png')

        iou = IoU(mask, pred_mask)
        total_union += get_union(mask, pred_mask)
        total_intersection += get_intersection(mask, pred_mask)
        total_iou += iou
        count += 1

    pd.to_pickle(result, 'pred_mask.pkl')
    average_iou = total_iou / count if count > 0 else 0.0
    overall_iou = total_intersection / total_union if total_union > 0 else 0.0
    return {'average_iou': average_iou, 'overall_iou': overall_iou}

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--num', '-n', type=int, default=None, help='Number of samples to evaluate, none for all')
    parser.add_argument('--complete', action='store_true', default=False, help='Use complete dataset')
    parser.add_argument('--random', action='store_true', default=False, help='Randomly sample from the dataset if num is specified')
    args = parser.parse_args()
    num_samples = args.num
    complete = args.complete
    random = args.random
    
    dataset = RefCOCODataset(split='testA', complete=complete, num_samples=num_samples, random=random)
    inference_fn = lambda img, query: sam4mllm_inference(img, query, eval=True)
    scores = evaluate(inference_fn, dataset)
    print(f"Average IoU: {scores['average_iou']:.4f}, Overall IoU: {scores['overall_iou']:.4f}")
