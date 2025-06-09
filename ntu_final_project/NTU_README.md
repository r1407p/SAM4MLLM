# Sam4MLLM-Plus
This project need 40GB GPU memory to run.

## Create venv
```
python3.10 -m venv venv
source venv/bin/activate
```
require python 3.10 cannot be 3.8 or 3.12

## install dependency
llava module
```
git clone git@github.com:LLaVA-VL/LLaVA-NeXT.git
cd LLaVA-NeXT
pip install --upgrade pip
pip install -e ".[train]"
cd ..
```

efficientvit module
```
git clone git@github.com:mit-han-lab/efficientvit.git
cd efficientvit
pip install -e .
cd ..
```


## Download checkpoint

```
mkdir checkpoint
cd checkpoint
wget https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main/efficientvit_sam_xl1.pt
mkdir sam4mllm_plus

```
download efficientvit_xl1_decoder_coco_ft.pt[https://drive.google.com/drive/folders/14burV34SxcQnxqkoiQ9Ax-OB26XmSf8S]

download sam4mllm_plus checkpoints and metadata [https://drive.google.com/drive/folders/1ytEfGRa6bxThTXQn5MLVKKy4jsxxBo6M]

download the semantic method sam_vit_h_4b8939.pth [https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth]
**sam4mllm_plus not sam4mllm**

### structure
```
checkpoint/
├── efficientvit_sam_xl1.pt
├── effvit_xl1_decoder_coco_ft.pt
└── sam4mllm_plus/
    ├── adapter_config.json
    ├── adapter_model.safetensors
    ├── README.md
    ├── special_tokens_map.json
    ├── tokenizer_config.json
    └── tokenizer.json
semantic_1/
    └── sam_vit_h_4b8939.pth
```

## Dataset preparation and inference preparation
### Setup
- Dataset: [RefCOCO](https://huggingface.co/datasets/lmms-lab/RefCOCO)  (using testA split)
- Amount: 
    - Partial: 1975 samples (only the first caption per ground truth segmentation is used.)
    - Complete: 5657 samples (all captions for each ground truth segmentation are used.)
- Metrics: average IoU, overall IoU, ...

###  How to Run
In `evaluate.py`, replace `inference_fn` with our own function. The function should take the following inputs and return a predicted binary mask:
```
def inference_fn(image: PIL.Image.Image, query: str) -> np.ndarray:
```
I have implemented one in sam4mllm_infer.py for the baseline method (API version). 

Run `evaluate.py`
```
python -m evaluate
```
```
usage: evaluate.py [-h] [--num NUM] [--complete] [--random]

options:
  -h, --help         show this help message and exit
  --num NUM, -n NUM  Number of samples to evaluate, none for all
  --complete         Use complete dataset
  --random           Randomly sample from the dataset if num is specified
```
Then, you will get two scores, **average and overall IoU**, and all predicted masks will be saved to `pred_masks.pkl`.

`data_utils.py` contains the code for loading and preprocessing data.


## run the inference
step 1: startup the server for sam server:
```
CUDA_VISIBLE_DEVICES=0 python3 -m ntu_final_project.api_server.inference_server 
```
step 2: modify the config file in `ntu_final_project/config.py` if your server is not running on localhost or port 5000.

step 3: run the inference
```
CUDA_VISIBLE_DEVICES=1 python3 -m evaluate 
```
