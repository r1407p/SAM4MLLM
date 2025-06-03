import requests
from PIL import Image, ImageDraw
import io
from typing import List, Tuple
from ntu_final_project.config import URL_BASE
from ntu_final_project.module.bbox_generator import BboxGenerator
from ntu_final_project.module.point_generator import PointGenerator
from ntu_final_project.module.mask_generator import MaskGenerator

def draw_bounding_box_points(image: Image, bbox:List[int], points: List[Tuple[float, float]], point_labels: List[bool], store=False) -> None:
    draw = ImageDraw.Draw(image)
    img_w, img_h = image.size
    x1, y1, x2, y2 = bbox # x1, y1, x2, y2 is relative coordinates in the range [0, 1000]
    abs_x1 = x1 / 1000 * img_w
    abs_y1 = y1 / 1000 * img_h
    abs_x2 = x2 / 1000 * img_w
    abs_y2 = y2 / 1000 * img_h
    draw.rectangle([abs_x1, abs_y1, abs_x2, abs_y2], outline='red', width=5)
    
    for (x, y), label in zip(points, point_labels):
        pass
    if store:
        image.save('./output_image_with_bbox.png')
    return draw    


def inferene(image_path, prompt):
    image_data = open(image_path, 'rb').read()
    bbox_generator = BboxGenerator()
    point_generator = PointGenerator()
    mask_generator = MaskGenerator()
    # bbox = bbox_generator.generate_bounding_box_mllm(image_data, prompt)
    # bbox = bbox_generator.generate_bounding_box_yolo(image_data, prompt)
    bbox = bbox_generator.generate_bounding_box_qwen(image_path, prompt)
    print(f"Generated bounding box: {bbox}")
    points = point_generator.generate_edge_points(image_data, bbox)
    # points = point_generator.generate_random_points(image_data, bbox, num_points=30)
    # points = point_generator.generate_point_grid(image_data, bbox)
    
    print(f"Generated points: {points}")
    mask, result = mask_generator.generate_mask(image_data, prompt, points, bbox)
    print(mask)
    draw_image = Image.open(io.BytesIO(bytes.fromhex(result["draw_image"])))
    draw_image.save('ntu_final_project/output_draw_image.png')
    
    image_with_mask_bytes = bytes.fromhex(result["image_with_mask"])
    image_with_mask = Image.open(io.BytesIO(image_with_mask_bytes))
    image_with_mask.save('ntu_final_project/output_image_with_mask.png')
    
    for point, label in zip(points, result["point_on_object"]):
        print(f'Point: {point}, Label: {label}')
    # breakpoint()
    return mask

if __name__ == "__main__":
    image_path = './test_imgs/000000025515.jpg'
    s_phrase = 'side view bird'
    inferene(image_path, s_phrase)
    
#  CUDA_VISIBLE_DEVICES=7 python3 -m ntu_final_project.ntu_agent