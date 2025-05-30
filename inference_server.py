import os
import sys
import torch
import numpy as np
from PIL import Image, ImageDraw
from flask import Flask, request, jsonify, send_file
from io import BytesIO
from transformers import LlamaTokenizer
from peft import PeftModel
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX
from efficientvit.sam_model_zoo import create_efficientvit_sam_model
from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor

# Set up the Flask app
app = Flask(__name__)

# Paths to your models
PRETRAINED = "lmms-lab/llama3-llava-next-8b"
ADAPTER_PATH = './checkpoint/sam4mllm_plus/'
EFFVIT_SAM_PATH = "./checkpoint/efficientvit_sam_xl1.pt"

# Load models (you may want to adjust this based on your setup)
def load_model():
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        PRETRAINED,
        None,
        "llava_llama3",
        device_map='cuda:0',
        # for v100
        torch_dtype=torch.float16,
        attn_implementation='eager',
        
        # for a100 or later 
        # torch_dtype=torch.bfloat16,
        # attn_implementation='flash_attention_2',
    )
    tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    model.tie_weights()

    if ADAPTER_PATH:
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    effvit_sam = create_efficientvit_sam_model(
        name="efficientvit-sam-xl1", weight_url=EFFVIT_SAM_PATH,
    )


    effvit_sam = effvit_sam.cuda().eval()

    sd = torch.load('./checkpoint/effvit_xl1_decoder_coco_ft.pt') 
    sd = {k.replace("model.", ""): v for k, v in sd.items()} 
    r = effvit_sam.load_state_dict(sd, strict=False) 
    print('unexpected_keys:', r.unexpected_keys)

    effvit_sam_predictor = EfficientViTSamPredictor(effvit_sam)
    
    return model, tokenizer, image_processor, effvit_sam_predictor


model, tokenizer, image_processor, effvit_sam_predictor = load_model()
model = model.eval()
config = model.config
model = model.merge_and_unload()
device = 'cuda:0'
def sel_points(points, all_probs_2, neg_thres=0.2, pos_thres=0.8):
    sel_points, sel_labels = [], []
    for (x, y), score in zip(points, all_probs_2):
        if score[0] > neg_thres:
            sel_points.append((x, y)), sel_labels.append(0)
        elif score[1] > pos_thres:
            sel_points.append((x, y)), sel_labels.append(1)
    
    sel_points, sel_labels = np.array(sel_points), np.array(sel_labels)
    
    return sel_points, sel_labels

def process_points(points, bbox):
    points = np.array(points) / 100
    x1, y1, x2, y2 = bbox
    points = np.array(points) * np.array([x2 - x1, y2 - y1]) + np.array([x1, y1])
    points = points.astype(int)
    points_txt = ' '.join([f"({p[0]:03d},{p[1]:03d})" for p in points])
    return points, points_txt

def sam_pred_mask(effvit_sam_predictor, points_sel, labels_sel, bbox, ori_img_w, ori_img_h):
    if len(points_sel) != 0:
        scaled_sel_points = points_sel / [1000, 1000] * [ori_img_w, ori_img_h]
    else:
        scaled_sel_points, labels_sel = None, None
    scaled_bbox = np.array(bbox) / 1000 * [ori_img_w, ori_img_h, ori_img_w, ori_img_h]
    pred_masks, scores, logits = effvit_sam_predictor.predict(
        point_coords=scaled_sel_points,
        point_labels=labels_sel,
        box=scaled_bbox[None, :],
        multimask_output=True,
    )
    pred_mask = pred_masks[scores.argmax()]
    
    return pred_mask

def draw_rectangle_and_points(image, bbox, points_sel, labels_sel):
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    img_w, img_h = image.size
    x1, y1, x2, y2 = bbox
    draw.rectangle([x1/1000*img_w, y1/1000*img_h, x2/1000*img_w, y2/1000*img_h], outline='red', width=5)

    for (x, y), label in zip(points_sel, labels_sel):
        draw.ellipse([x/1000*img_w-5, y/1000*img_h-5, x/1000*img_w+5, y/1000*img_h+5], fill='red' if label else 'blue')

    return draw_image


@app.route('/generate_mask', methods=['POST'])
def generate_mask():
    """
    Generate a mask for the specified bounding box and points in the image.
    :return: JSON response containing the mask and drawn image
    """
    global model, tokenizer, image_processor, effvit_sam_predictor, config, device
    data = request.json
    image_data = data.get('image')
    s_phrase = data.get('s_phrase')
    points = data.get('points')
    bbox = data.get('bbox')

    if not all([image_data, s_phrase, points, bbox]):
        return jsonify({"error": "Missing required parameters"}), 400

    # Convert image data to PIL Image
    image = Image.open(BytesIO(bytes(image_data)))

    bbox_txt = f'{bbox}'
    x1, y1, x2, y2 = bbox
    points, points_txt = process_points(points, bbox)

    answer_counts = '1'

    image_tensor = process_images([image], image_processor, config)
    image_tensor = [_image.to(dtype=torch.bfloat16) for _image in image_tensor]
    image_sizes = [image.size]
    prompt_question = tokenizer.apply_chat_template([
        {"role": "system", "content": "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."},
        {"role": "user", "content": f'<image>\nPlease provide the bounding box coordinate of the region this sentence describes ({answer_counts}):\n"{s_phrase}".'},
    ], tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)


    with torch.backends.cuda.sdp_kernel(enable_flash=False):
        output = model.generate(
            input_ids,
            images=[x.float() for x in image_tensor],
            image_sizes=image_sizes,
            max_new_tokens=30,
        )

    text_output = tokenizer.decode(output[0], skip_special_tokens=True)
    print(text_output)

    question_points = f'Check if the points listed below are located on the object with coordinates {bbox_txt}:\n{points_txt}'

    prompt_question = tokenizer.apply_chat_template([
        {"role": "system", "content": "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."},
        {"role": "user", "content": f'<image>\nPlease provide the bounding box coordinate of the region this sentence describes ({answer_counts}):\n"{s_phrase}".'},
        {"role": "assistant", "content": text_output},
        {"role": "user", "content": question_points},
    ], tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

    output_2 = model.generate(
        input_ids,
        images=[x.float() for x in image_tensor],
        image_sizes=image_sizes,
        max_new_tokens=30,
        output_logits=True,
        return_dict_in_generate=True,
    )
    print(output_2[0])
    text_output_2 = tokenizer.decode(output_2[0][0], skip_special_tokens=True)
    print(text_output_2)

    yesno_probs = torch.stack(output_2['logits'], dim=1).softmax(dim=-1)
    yesno_probs = yesno_probs[0, :30, [2822, 9642]].float().cpu().numpy()

    print(yesno_probs)

    points_sel, labels_sel = sel_points(points, yesno_probs, neg_thres=0.9, pos_thres=0.75)
    print(points_sel, labels_sel)

    img_w, img_h = image.size
    draw_image = draw_rectangle_and_points(image, bbox, points_sel, labels_sel)

    effvit_sam_predictor.set_image(np.array(image))
    pred_mask = sam_pred_mask(
        effvit_sam_predictor, points_sel, labels_sel, bbox, img_w, img_h
    )
    mask = Image.fromarray((pred_mask * 255).astype(np.uint8))  # Convert to black/white image

    image_with_mask = Image.fromarray((np.array(image) * 0.5 + np.array(pred_mask)[:, :, None] * [255, 0, 0] * 0.5).astype(np.uint8))

    # Convert images to bytes for JSON response
    mask_io = BytesIO()
    mask.save(mask_io, format='PNG')
    mask_io.seek(0)

    draw_image_io = BytesIO()
    draw_image.save(draw_image_io, format='PNG')
    draw_image_io.seek(0)

    image_with_mask_io = BytesIO()
    image_with_mask.save(image_with_mask_io, format='PNG')
    image_with_mask_io.seek(0)

    return jsonify({
        "point_on_object": labels_sel.tolist(),
        "mask": mask_io.getvalue().hex(),
        "draw_image": draw_image_io.getvalue().hex(),
        "image_with_mask": image_with_mask_io.getvalue().hex(),
    })

# Start the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)