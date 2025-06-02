# Evaluation
## Install Dependencies
```
pillow
torch
datasets
pandas
numpy
pycocotools
```

## Setup
- Dataset: [RefCOCO](https://huggingface.co/datasets/lmms-lab/RefCOCO)  (using testA split)
- Amount: 
    - Partial: 1975 samples (only the first caption per ground truth segmentation is used.)
    - Complete: 5657 samples (all captions for each ground truth segmentation are used.)
- Metrics: average IoU, overall IoU, ...

## How to Run
In `evaluate.py`, replace `inference_fn` with our own function. The function should take the following inputs and return a predicted binary mask:
```
def inference_fn(image: PIL.Image.Image, query: str) -> np.ndarray:
```
I have implemented one in sam4mllm_infer.py for the baseline method (API version). 

Run `evaluate.py`
```
python evaluate.py 
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

