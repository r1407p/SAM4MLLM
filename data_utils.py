import numpy as np
from typing import List, Tuple

from torch.utils.data import Dataset
from PIL import Image
from datasets import load_dataset
from pycocotools import mask as maskUtils

# convert coco segmentation to binary mask
def seg_to_mask(seg, img_h, img_w):
    if isinstance(seg, list):
        rles = maskUtils.frPyObjects(seg, img_h, img_w)
        rle = maskUtils.merge(rles)
    elif isinstance(seg['counts'], list):
        rle = maskUtils.frPyObjects(seg, img_h, img_w)
    else:
        rle = seg
    return maskUtils.decode(rle)

class RefCOCODataset(Dataset):
    def __init__(self, split='test', num_samples=None, complete=False, random=False):
        self.dataset = load_dataset("lmms-lab/RefCOCO", split=split)
        self.get_dataset(complete)
        if num_samples is not None:
            if random:
                np.random.seed(42)
                indices = np.random.choice(len(self.dataset), num_samples, replace=False)
                self.dataset = [self.dataset[i] for i in indices]
            else:
                self.dataset = self.dataset[:num_samples]
        
    def get_dataset(self, complete=False):
        new_dataset = []
        data_id = 0
        for data in self.dataset:
            img: Image.Image = data['image']
            seg = data['segmentation']
            bbox = data['bbox']
            bbox_point = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            
            mask = seg_to_mask([seg], img.size[1], img.size[0])
            mask = mask.astype(np.uint8) * 255
            mask_img = Image.fromarray(mask, mode='L')
            query = data['answer'][0]
            if complete:
                for ans in data['answer']:
                    new_dataset.append({
                        'data_id': data_id,
                        'image': img,
                        'mask_image': mask_img,
                        'bbox': bbox_point,
                        'answer': ans
                    })
                    data_id += 1
            else:
                new_dataset.append({
                    'data_id': data_id,
                    'image': img,
                    'mask_image': mask_img,
                    'bbox': bbox_point,
                    'answer': query
                })
                data_id += len(data['answer'])
        self.dataset = new_dataset
            
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> Tuple[int, Image.Image, str, Image.Image, List[Tuple[int, int]]]:
        data = self.dataset[idx]
        data_id = data['data_id']
        img: Image.Image = data['image']
        mask_img: Image.Image = data['mask_image']
        bbox = data['bbox']
        query = data['answer']
        
        return data_id, img, query, mask_img, bbox

if __name__ == "__main__":
    dataset = RefCOCODataset(split='testA')
    print(f"Total samples in dataset: {len(dataset)}")
    idx = 0
    data_id, img, query, mask_img, bbox_points = dataset[idx]
    print(f"Data ID: {data_id}, Query: {query}, BBox Points: {bbox_points}")
    img.save(f"data/image_{idx}.png")
    mask_img.save(f"data/mask_{idx}.png")
    # print(f"Image {idx}: {query}, BBox Points: {bbox_points}")