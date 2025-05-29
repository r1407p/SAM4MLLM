import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append('../')

import re
import glob
import json
import random
import pickle
import numpy as np
from tqdm.auto import tqdm
from itertools import product

import cv2
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import torch

from transformers import BitsAndBytesConfig, LogitsProcessor
from peft import PeftModel

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX

# from efficientvit.sam_model_zoo import create_sam_model
from efficientvit.sam_model_zoo import create_efficientvit_sam_model
from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor

PRETRAINED = "lmms-lab/llama3-llava-next-8b"
ADAPTER_PATH = './checkpoint/sam4mllm_plus/' # or './checkpoint/sam4mllm_plus/'
# EFFVIT_SAM_PATH = "./checkpoint/xl1.pt"
EFFVIT_SAM_PATH= "./checkpoint/efficientvit_sam_xl1.pt"


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


grid_rand_points = [
    (93, 70), (51, 89), (91, 90), (32, 32), (88, 10),
    (12, 28), (29, 52), (49, 49), (28, 12), (59, 60),
    (9, 48), (52, 29), (31, 92), (68, 13), (73, 73),
    (69, 53), (48, 9), (19, 18), (71, 93), (53, 69),
    (89, 50), (11, 88), (33, 72), (39, 41), (72, 33),
    (13, 68), (79, 82), (8, 8), (81, 22), (92, 30),
]
grid_rand_points = np.array(grid_rand_points) / 100
grid_rand_points = grid_rand_points[:15]
number_tokens = tokenizer(' '.join([str(i) for i in range(1000)]), add_special_tokens=False)['input_ids'][::2]
print(grid_rand_points)
test_img_path = './test_imgs/000000025515.jpg'
image = Image.open(test_img_path)
answer_counts = '1'
s_phrase = 'side view bird'
device = 'cuda:0'
image_tensor = process_images([image], image_processor, model.config)
image_tensor = [_image.to(dtype=torch.bfloat16) for _image in image_tensor]
image_sizes = [image.size]
prompt_question = tokenizer.apply_chat_template([
    {"role": "system", "content": "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."},
    {"role": "user", "content": f'<image>\nPlease provide the bounding box coordinate of the region this sentence describes ({answer_counts}):\n"{s_phrase}".'},
], tokenize=False, add_generation_prompt=True)
input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)


model = model.merge_and_unload()
with torch.backends.cuda.sdp_kernel(enable_flash=False):
    output = model.generate(
        input_ids,
        images=[x.float() for x in image_tensor],
        image_sizes=image_sizes,
        max_new_tokens=30,
    )

text_output = tokenizer.decode(output[0], skip_special_tokens=True)
print(text_output)
def point_to_str(point):
    return f"({point[0]:03d},{point[1]:03d})"
bbox = [281,240,863,999]
bbox_txt = '[281,240,863,999]'
x1, y1, x2, y2 = bbox

rand_points = grid_rand_points * np.array([x2 - x1, y2 - y1]) + np.array([x1, y1])
rand_points = rand_points.astype(int)

points_txt = ' '.join([point_to_str(p) for p in rand_points])
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


def sel_points(rand_points, all_probs_2, neg_thres=0.2, pos_thres=0.8):
    sel_points, sel_labels = [], []
    for (x, y), score in zip(rand_points, all_probs_2):
        if score[0] > neg_thres:
            sel_points.append((x, y)), sel_labels.append(0)
        elif score[1] > pos_thres:
            sel_points.append((x, y)), sel_labels.append(1)
    
    sel_points, sel_labels = np.array(sel_points), np.array(sel_labels)
    
    return sel_points, sel_labels

points_sel, labels_sel = sel_points(rand_points, yesno_probs, neg_thres=0.9, pos_thres=0.75)


draw_image = image.copy()
draw = ImageDraw.Draw(draw_image)
img_w, img_h = image.size
draw.rectangle([x1/1000*img_w, y1/1000*img_h, x2/1000*img_w, y2/1000*img_h], outline='red', width=5)

for (x, y), label in zip(points_sel, labels_sel):
    draw.ellipse([x/1000*img_w-5, y/1000*img_h-5, x/1000*img_w+5, y/1000*img_h+5], fill='red' if label else 'blue')

# store draw_image
draw_image.save('./test_imgs/000000025515_draw.jpg')

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

print(points_sel)

effvit_sam_predictor.set_image(np.array(image))
pred_mask = sam_pred_mask(
    effvit_sam_predictor, points_sel, labels_sel, bbox, img_w, img_h
)
plt.imshow(image)
plt.imshow(pred_mask, alpha=0.5)
# save the mask
plt.savefig('./test_imgs/000000025515_mask.jpg')