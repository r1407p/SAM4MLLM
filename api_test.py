import requests
import json
from PIL import Image
import io
import numpy as np

url = 'https://fa42-149-7-4-150.ngrok-free.app/generate_mask'
image_path = './test_imgs/000000025515.jpg'
s_phrase = 'hand'
points = [
    [93, 70], [51, 89], [91, 90], [32, 32], [88, 10],
    [12, 28], [29, 52], [49, 49], [28, 12], [59, 60],
    [9, 48], [52, 29], [31, 92], [68, 13], [73, 73]
] # coordinates in the format [[x1, y1], [x2, y2], ...]
# coordinates should be in the range of [0, 100] for both x and y, 
# representing percentages of the relative position in the bounding box

bbox = [0,467,540,999] # bounding box coordinates in the format [x1, y1, x2, y2]

with open(image_path, 'rb') as f:
    image_data = f.read()

payload = {
    'image': list(image_data),
    's_phrase': s_phrase,
    'points': points,
    'bbox': bbox
}

response = requests.post(url, json=payload)

if response.status_code == 200:
    result = response.json()
    print(result.keys())
    image_with_mask_bytes = bytes.fromhex(result["image_with_mask"])
    image_with_mask = Image.open(io.BytesIO(image_with_mask_bytes))

    image_with_mask.save('./output_image_with_mask.png')

    print('Mask image saved as ./output_image_with_mask.png')
    
    mask_bytes = bytes.fromhex(result["mask"])
    mask = Image.open(io.BytesIO(mask_bytes))
    mask.save('./output_mask.png')
    mask_array = np.array(mask)
    print('Mask image as numpy array:', mask_array)

    draw_image = Image.open(io.BytesIO(bytes.fromhex(result["draw_image"])))
    draw_image.save('./output_draw_image.png')

    for point, label in zip(points, result["point_on_object"]):
        print(f'Point: {point}, Label: {label}')


else:
    print('Request failed with status code:', response.status_code)
    print('Response:', response.text)
