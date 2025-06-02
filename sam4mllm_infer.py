import requests
import json
from PIL import Image
import io
import numpy as np

def inference(image, s_phrase, eval=False):
    # base_url = 'https://fa42-149-7-4-150.ngrok-free.app'
    base_url = 'https://fec7-140-112-29-180.ngrok-free.app'
    bbox_url = base_url + '/generate_bbox'
    mask_url = base_url + '/generate_mask'
    points = [
        [93, 70], [51, 89], [91, 90], [32, 32], [88, 10],
        [12, 28], [29, 52], [49, 49], [28, 12], [59, 60],
        [9, 48], [52, 29], [31, 92], [68, 13], [73, 73]
    ] # coordinates in the format [[x1, y1], [x2, y2], ...]
    # coordinates should be in the range of [0, 100] for both x and y, 
    # representing percentages of the relative position in the bounding box
    if isinstance(image, np.ndarray):
        image = image.tolist()
    elif isinstance(image, Image.Image):
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        image = list(img_byte_arr.getvalue())
    payload = {
        'image': image,
        's_phrase': s_phrase,
        'points': points,
    }
    bbox_response = requests.post(bbox_url, json=payload)
    if bbox_response.status_code == 200:
        bbox = bbox_response.json().get('bbox', None)
    else:
        if not eval:
            print('Failed to get bounding box:', bbox_response.status_code, bbox_response.text)
        return None   
    payload['bbox'] = bbox
    response = requests.post(mask_url, json=payload)

    if response.status_code == 200:
        result = response.json()
        if not eval:
            print(result.keys())
        image_with_mask_bytes = bytes.fromhex(result["image_with_mask"])
        image_with_mask = Image.open(io.BytesIO(image_with_mask_bytes))

        image_with_mask.save('./output_image_with_mask.png')

        if not eval:
            print('Mask image saved as ./output_image_with_mask.png')
        
        mask_bytes = bytes.fromhex(result["mask"])
        mask = Image.open(io.BytesIO(mask_bytes))
        mask.save('./output_mask.png')

        draw_image = Image.open(io.BytesIO(bytes.fromhex(result["draw_image"])))
        draw_image.save('./output_draw_image.png')
        if not eval:
            for point, label in zip(points, result["point_on_object"]):
                print(f'Point: {point}, Label: {label}')
        mask_array = np.array(mask, dtype=np.uint8)
        return mask_array

    else:
        if not eval:
            print('Request failed with status code:', response.status_code)
            print('Response:', response.text)
        return None

if __name__ == "__main__":
    image_path = './test_imgs/000000025515.jpg'
    image = Image.open(image_path)
    s_phrase = 'hand'
    
    inference(image, s_phrase)