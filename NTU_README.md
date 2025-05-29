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

## run the inference
```
CUDA_VISIBLE_DEVICES=0 python simple_infer.py
```