import numpy as np
import cv2
import requests
import io
from PIL import Image, ImageDraw
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import matplotlib.pyplot as plt

url = 'https://fa42-149-7-4-150.ngrok-free.app/generate_mask'
image_path = './test_imgs/meme.jpg'
s_phrase = 'white cat'  
ckpt_path = './semantic_1/sam_vit_h_4b8939.pth'
area_thresh_ratio = 0.02  
# ===================

# === 1. load image ===
image_pil = Image.open(image_path).convert('RGB')
image_np = np.array(image_pil)
H, W = image_np.shape[:2]

# === 2. SAM  ===
sam = sam_model_registry["default"](checkpoint=ckpt_path)
mask_generator = SamAutomaticMaskGenerator(sam, min_mask_region_area=5000)
masks = mask_generator.generate(image_np)

# === 3. get points and color masks ===
area_thresh = area_thresh_ratio * H * W
filtered_masks = [m for m in masks if np.sum(m["segmentation"]) >= area_thresh]

points = []
color_mask = np.zeros_like(image_np)

for i, m in enumerate(filtered_masks):
    seg = m["segmentation"]
    ys, xs = np.where(seg)
    if len(xs) == 0:
        continue

    # center point
    x_med = int(np.median(xs))
    y_med = int(np.median(ys))
    x_pct_med = int(x_med / W * 100)
    y_pct_med = int(y_med / H * 100)
    points.append([x_pct_med, y_pct_med])

    # random point
    idx = np.random.randint(len(xs))
    x_rand = xs[idx]
    y_rand = ys[idx]
    x_pct_rand = int(x_rand / W * 100)
    y_pct_rand = int(y_rand / H * 100)
    points.append([x_pct_rand, y_pct_rand])

    # mask color
    color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
    color_mask[seg] = color

# === 4. visulization ===
blended = (0.5 * image_np + 0.5 * color_mask).astype(np.uint8)
vis_image = Image.fromarray(blended)
draw = ImageDraw.Draw(vis_image)
for x_pct, y_pct in points:
    x = int(x_pct / 100 * W)
    y = int(y_pct / 100 * H)
    draw.ellipse((x-4, y-4, x+4, y+4), fill=(255, 0, 0))
vis_image.save('./2output_sam_mask_points.png')
print('Saved: ./2output_sam_mask_points.png')

# === 5. request API ===
bbox = [0, 0, 1000, 1000]  # whole image bounding box
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

    Image.open(io.BytesIO(bytes.fromhex(result["image_with_mask"]))).save('./2output_image_with_mask.png')
    Image.open(io.BytesIO(bytes.fromhex(result["mask"]))).save('./2output_mask.png')
    Image.open(io.BytesIO(bytes.fromhex(result["draw_image"]))).save('./2output_draw_image.png')
    print("Saved: SAM result masks and draw image.")

    for point, label in zip(points, result["point_on_object"]):
        print(f'Point: {point}, Label: {label}')
else:
    print('Request failed:', response.status_code)
    print(response.text)
