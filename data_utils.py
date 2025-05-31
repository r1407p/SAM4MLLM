import numpy as np
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
    def __init__(self, split='test'):
        self.dataset = load_dataset("lmms-lab/RefCOCO", split=split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        img: Image.Image = data['image']
        seg = data['segmentation']
        bbox = data['bbox']
        query = data['question']
        
        bbox_point = [(bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1]),
                    (bbox[0] + bbox[2], bbox[1] + bbox[3]), (bbox[0], bbox[1] + bbox[3])]
        
        mask = seg_to_mask([seg], img.size[1], img.size[0])
        mask = mask.astype(np.uint8) * 255
        mask_img = Image.fromarray(mask, mode='L')
        return img, query, mask_img, bbox_point

if __name__ == "__main__":
    dataset = RefCOCODataset(split='test')
    idx = 0
    img, query, mask_img, bbox_points = dataset[idx]
    img.save(f"data/image_{idx}.png")
    mask_img.save(f"data/mask_{idx}.png")
    # print(f"Image {idx}: {query}, BBox Points: {bbox_points}")