import requests
import json
from PIL import Image
import io
import numpy as np
from ultralytics import YOLO
import cv2

# 載入模型（可選 yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt）
model = YOLO('yolov8n.pt')  # n 是最輕量版，可跑在 CPU 上

# 載入圖片
image_path = './test_imgs/000000025515.jpg'
results = model(image_path)

# 顯示結果圖片 + 存檔
results[0].save(filename='output_detect.jpg')  # 存下有框框的圖片

# 挑出信心值最高的物件
best_box = None
best_conf = 0.0
for box in results[0].boxes:
    conf = box.conf[0].item()
    if conf > best_conf:
        best_conf = conf
        best_box = box

if best_box is None:
    print("沒有偵測到任何物體，無法送到 API")
    exit()

# 擷取資訊
cls_id = int(best_box.cls[0])
label = model.names[cls_id]
x1, y1, x2, y2 = best_box.xyxy[0].tolist()
print(f'使用信心值最高的Object: Label={label}, Confidence={best_conf:.2f}')

# 丟入 API 的 s_phrase
s_phrase = label

# 圖片尺寸資訊
with Image.open(image_path) as img:
    absolute_width_img, absolute_height_img = img.size

# 計算相對 bbox（scale to 1000）
width_img = absolute_width_img
height_img = absolute_height_img
x_min = x1 / width_img * 1000
y_min = y1 / height_img * 1000
x_max = x2 / width_img * 1000
y_max = y2 / height_img * 1000
bbox = [x_min, y_min, x_max, y_max]

# 測試點（假設這些是你要的點）
points = [
    [93, 70], [51, 89], [91, 90], [32, 32], [88, 10],
    [12, 28], [29, 52], [49, 49], [28, 12], [59, 60],
    [9, 48], [52, 29], [31, 92], [68, 13], [73, 73]
] # coordinates in the format [[x1, y1], [x2, y2], ...]
# coordinates should be in the range of [0, 100] for both x and y, 
# representing percentages of the relative position in the bounding box


# 送出請求
url = 'https://fa42-149-7-4-150.ngrok-free.app/generate_mask'
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
    print('✔️ Mask image saved as ./output_image_with_mask.png')
    
    mask_bytes = bytes.fromhex(result["mask"])
    mask = Image.open(io.BytesIO(mask_bytes))
    mask.save('./output_mask.png')
    mask_array = np.array(mask)
    print('✔️ Mask image as numpy array:', mask_array)
    
    draw_image = Image.open(io.BytesIO(bytes.fromhex(result["draw_image"])))
    draw_image.save('./output_draw_image.png')
    print('✔️ Draw image saved as ./output_draw_image.png')
    
    for point, label in zip(points, result["point_on_object"]):
        print(f'Point: {point}, Label: {label}')

else:
    print('Request failed with status code:', response.status_code)
    print('Response:', response.text)
