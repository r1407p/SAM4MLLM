import requests
from PIL import Image, ImageDraw
import io
from typing import List, Tuple
from ntu_final_project.config import URL_BASE
from ultralytics import YOLO
import numpy as np
import cv2
import re
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch



class BboxGenerator:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')  # Load YOLO model, can be changed to other versions
        
        self.model_name = "Qwen/Qwen2.5-VL-7B-Instruct" # for better performance
        # model_name = "Qwen/Qwen2.5-VL-3B-Instruct" # for faster inference with some lost in performance

        self.qwdn_model = model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name, torch_dtype=torch.float16, device_map="cuda"
        )
        self.processor = AutoProcessor.from_pretrained(self.model_name)


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
        

    def generate_bounding_box_qwen(self, image, prompt: str) -> List[int]:
        """
        Generate bounding box using Qwen model based on the image and prompt.
        
        :param image_data: PIL Image or bytes of the image
        :param prompt: the prompt to describe the bounding box
        :return: List[int] representing the bounding box coordinates in [x1, y1, x2, y2]
        """
        # Convert raw image data to PIL Image
        # image = Image.open(io.BytesIO(image_data))

        # Create message input for the Qwen model
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": f'Where is the "{prompt}"? Answer in the format of "bbox_2d": [x1, y1, x2, y2]'},
                ],
            }
        ]
        # messages = [
        #         {
        #             "role": "user",
        #             "content": [
        #                 {
        #                     "type": "image",
        #                     "image": image_path,
        #                 },
        #                 {"type": "text", "text": f'Where is the "{prompt}"? Answer in the format of "bbox_2d": [x1, y1, x2, y2]'},
        #             ],
        #         }
        #     ]
        # Process the messages to generate the text and image inputs
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        # Prepare the inputs for the model
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")  # Ensure the input is moved to the same device as the model

        # Generate the bounding box using the model
        generated_ids = self.qwdn_model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # Decode the generated text
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # Extract the bounding box from the output text using regex
        bbox_list = self.extract_bbox_2d(output_text[0])
        
        if not bbox_list:
            print("No bounding box found in the output. Returning raw output.")
            return output_text[0]
        
        if len(bbox_list) > 1:
            print(f"Warning: Multiple bounding boxes found: {bbox_list}")
        
        return bbox_list[0]  # Return the first bounding box found

    @staticmethod
    def extract_bbox_2d(string: str) -> List[int]:
        """
        Extract bounding box from the generated text using regex.
        
        :param string: the string containing the bounding box in the format "bbox_2d": [x1, y1, x2, y2]
        :return: List[int] representing the bounding box [x1, y1, x2, y2]
        """
        pattern = r'"bbox_2d":\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'  # Regex pattern for bbox format
        matches = re.findall(pattern, string)
        
        # Convert matches to a list of integers
        bbox_list = [tuple(map(int, match)) for match in matches]
        
        return bbox_list
