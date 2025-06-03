import requests
from PIL import Image, ImageDraw
import io
from typing import List, Tuple
from ntu_final_project.config import URL_BASE
import numpy as np
import cv2



class MaskGenerator:
    def __init__(self):
        pass
    
    @staticmethod
    def generate_mask(image_data: bytes, prompt: str,
                      points: List[List[float]], bbox: List[float] = None
                      ):
        """
        Generate a mask for the bounding box.
        :param image_data: bytes of the image
        :param bbox: bounding box in the format [x1, y1, x2, y2]
        :return: list of points in the format [[x1, y1], [x2, y2], ...]
        """
        payload = {
            'image': list(image_data),
            's_phrase': prompt,
            'points': points,
            'bbox': bbox
        }
        response = requests.post(f'{URL_BASE}/generate_mask', json=payload)
        
        if response.status_code == 200:
            result = response.json()
            mask = result['mask']
            mask_bytes = bytes.fromhex(mask)
            mask_image = Image.open(io.BytesIO(mask_bytes))
            mask_array = np.array(mask_image)
            return mask_array, result
        else:
            raise Exception(f"Failed to generate mask: {response.text}")
        