import requests
from PIL import Image, ImageDraw
import io
from typing import List, Tuple
from ntu_final_project.config import URL_BASE
from ultralytics import YOLO
import numpy as np
import cv2



class BboxGenerator:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')  # Load YOLO model, can be changed to other versions
        

    @staticmethod
    def generate_bounding_box_sam_mllm(image_data, prompt: str) -> List[int]:
        """
        this method is baseline and uses the paper method to ask the MLLM to generate a bounding box
        and we build a api server to provide this service.
        api server code: ntu_final_project/inference_server.py

        :param image_data: bytes of the image
        :param prompt: the prompt to describe the bounding box
        """
        payload = {
            'image': list(image_data),
            's_phrase': prompt,
        }
        response = requests.post(f'{URL_BASE}/generate_bbox', json=payload)
        print(response)
        if response.status_code == 200:
            result = response.json()
            bbox = result['bbox'] # [float, float, float, float]
            return bbox
        else:
            raise Exception(f"Failed to generate bounding box: {response.text}")


    def generate_bounding_box_yolo(self, image_data, prompt: str) -> List[int]:
        # Convert byte data to numpy array
        image_array = np.frombuffer(image_data, np.uint8)

        # Decode the image array to an OpenCV image
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # Step 3: Perform object detection
        results = self.model(image)  
        results[0].save(filename='output_detect.jpg')  # 存下有框框的圖片

        # Step 4: Extract bounding boxes
        best_box = None
        best_conf = 0.0
        for box in results[0].boxes:
            conf = box.conf[0].item()
            if conf > best_conf:
                best_conf = conf
                best_box = box
        if best_box is None:
            return [0, 0, 1000, 1000]
        
        cls_id = int(best_box.cls[0])
        label = self.model.names[cls_id]
        x1, y1, x2, y2 = best_box.xyxy[0].tolist()
        print(f'Using the object with highest confidence: Label={label}, Confidence={best_conf:.2f}')
        
        
        w, h = image.shape[1], image.shape[0]
        # Calculate relative bbox (scale to 1000)
        x_min = x1 / w * 1000
        y_min = y1 / h * 1000
        x_max = x2 / w * 1000
        y_max = y2 / h * 1000
        bbox = [x_min, y_min, x_max, y_max]
        return bbox
        

    @staticmethod
    def generate_bounding_box_mllm(image_data, prompt: str) -> List[int]:
        raise NotImplementedError("MLLM bounding box generation is not implemented yet.")
