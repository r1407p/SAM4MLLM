import requests
import json
from PIL import Image
import io

url = 'http://localhost:5000/generate_mask'
image_path = './test_imgs/000000025515.jpg'
s_phrase = 'side view bird'
points = [
    [93, 70], [51, 89], [91, 90], [32, 32], [88, 10],
    [12, 28], [29, 52], [49, 49], [28, 12], [59, 60],
    [9, 48], [52, 29], [31, 92], [68, 13], [73, 73]
]
bbox = [281, 240, 863, 999]

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
else:
    print('Request failed with status code:', response.status_code)
    print('Response:', response.text)
