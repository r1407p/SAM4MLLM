# requirement: qwen-vl-utils[decord]==0.0.8
#
# sample usage:
# from bbox_detector import gen_bbox
# bbox = gen_bbox("path/to/000000025515.jpg", "side view bird")
# x1, y1, x2, y2 = bbox


from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

device = "cuda" # "cpu"
model_name = "Qwen/Qwen2.5-VL-7B-Instruct" # for better performance
# model_name = "Qwen/Qwen2.5-VL-3B-Instruct" # for faster inference with some lost in performance

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map=device
)

processor = AutoProcessor.from_pretrained(model_name)


pattern = r'"bbox_2d":\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]' # Regex to match bbox format

# ⭐ main function to load
def gen_bbox(image_path: str, query: str) -> tuple[int, int, int, int]:
    """
    Generate bounding box in the image for the query.
    Args:
        image_path (str): Path to the image file.
        query (str): The query to search for in the image.
    Returns:
        tuple[int, int, int, int]: Bounding box coordinates in the format (x1, y1, x2, y2).
    """
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": f'Where is the "{query}"? Answer in the format of "bbox_2d": [x1, y1, x2, y2]'},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    # Extract bbox_2d from the output text
    bbox_list = extract_bbox_2d(output_text[0])
    if not bbox_list:
        print("No bounding box found in the output. Returning raw output.")
        return output_text[0]
    if len(bbox_list) > 1:
        print(f"Warning: Multiple bounding boxes found: {bbox_list}")
    return bbox_list[0]  # Return the first bounding box found


def extract_bbox_2d(string: str):
    # Use regex to find the bbox_2d values
    matches = re.findall(pattern, string)

    # Convert matches to a list of tuples of integers
    bbox_list = [tuple(map(int, match)) for match in matches]

    return bbox_list
