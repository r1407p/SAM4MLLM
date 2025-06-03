import pandas as pd
import io

from PIL import ImageDraw
from tqdm import tqdm
from argparse import ArgumentParser

from data_utils import Dataset, RefCOCODataset
from eval_utils import *
from ntu_final_project.ntu_agent import BboxGenerator, bbox_generator
# from ntu_final_project.ntu_agent import inference_fn as ntu_inference

RESULT_PATH = 'pred_bbox.pkl'


def example_inference(image, query, bbox_generator: BboxGenerator, test_type='base'):
    image_data = io.BytesIO()
    image.save(image_data, format='JPEG')
    image_data = image_data.getvalue()
    if test_type == 'base':
        bbox = bbox_generator.generate_bounding_box_sam_mllm(image_data, query)
    elif test_type == 'qwen':
        bbox = bbox_generator.generate_bounding_box_qwen(image, query)
    elif test_type == 'yolo':
        bbox = bbox_generator.generate_bounding_box_yolo(image_data, query)
    if bbox is None:
        return None
    if test_type in ['base', 'yolo']:
        x1, y1, x2, y2 = bbox
        # normalize back to original image size
        w, h = image.size
        x1 = int(x1 / 1000 * w)
        y1 = int(y1 / 1000 * h)
        x2 = int(x2 / 1000 * w)
        y2 = int(y2 / 1000 * h)
        bbox = [x1, y1, x2, y2]
    return bbox
    
    
    
def evaluate_bbox(inference_fn, dataset: RefCOCODataset, attempt=3):
    total_union = 0.0
    total_intersection = 0.0
    total_iou = 0.0
    count = 0
    result = load_existing_results(RESULT_PATH)
    # result = {}

    for idx in tqdm(range(len(dataset))):
        data_id, img, query, mask_img, bbox = dataset[idx]
        # Visualize ground truth and predicted bounding boxes
        pred_bbox = result.get(data_id, None)
        if pred_bbox is None:
            for _ in range(attempt):
                try:
                    pred_bbox = inference_fn(img, query)
                    if pred_bbox is not None:
                        result[data_id] = pred_bbox
                        break
                except Exception as e:
                    continue
            else:
                print(f"Failed to get prediction for index {idx}")
                result[data_id] = None
                continue
            
        img_with_boxes = img.copy()
        draw = ImageDraw.Draw(img_with_boxes)
        draw.rectangle(bbox, outline='green', width=2)  # Ground truth in green
        if pred_bbox:
            draw.rectangle(pred_bbox, outline='red', width=2)  # Prediction in red
        img_with_boxes.save(f'output_image_with_boxes.png')
        
        gt_bbox_mask = get_mask_from_bbox(bbox, img.size[:2])
        pred_bbox_mask = get_mask_from_bbox(pred_bbox, img.size[:2])
        iou = IoU(gt_bbox_mask, pred_bbox_mask)
        # print(f"Data ID: {data_id}, IoU: {iou:.4f}")
        total_union += get_union(gt_bbox_mask, pred_bbox_mask)
        total_intersection += get_intersection(gt_bbox_mask, pred_bbox_mask)
        total_iou += iou
        count += 1

    pd.to_pickle(result, RESULT_PATH)
    average_iou = total_iou / count if count > 0 else 0.0
    overall_iou = total_intersection / total_union if total_union > 0 else 0.0
    return {'average_iou': average_iou, 'overall_iou': overall_iou}

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--num', '-n', type=int, default=None, help='Number of samples to evaluate, none for all')
    parser.add_argument('--complete', action='store_true', default=False, help='Use complete dataset')
    parser.add_argument('--random', action='store_true', default=False, help='Randomly sample from the dataset if num is specified')
    parser.add_argument('--type', '-t', type=str, default='base', choices=['qwen', 'yolo', 'base'], help='Type of bounding box generator to use')
    args = parser.parse_args()
    num_samples = args.num
    complete = args.complete
    random = args.random
    test_type = args.type
    RESULT_PATH.replace('.pkl', f'_{test_type}.pkl')
    
    dataset = RefCOCODataset(split='testA', complete=complete, num_samples=num_samples, random=random)
    
    inference_fn = lambda img, query: example_inference(img, query, bbox_generator, test_type=test_type)
    scores = evaluate_bbox(inference_fn, dataset)
    print(f"Average IoU: {scores['average_iou']:.4f}, Overall IoU: {scores['overall_iou']:.4f}")
