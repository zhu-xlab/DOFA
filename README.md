
# [DOFA](https://github.com/ShadowXZT/DOFA-pytorch)

>
> 🚨 Examples for using DOFA and DOFAv2 for object detection and instance segmentation with [terratorch](https://github.com/xiong-zhitong/terratorch/tree/main/examples/confs/od_dofav1).
>
> 🚨 Please use the new version of codes and [weights](https://huggingface.co/XShadow/DOFA). The performance is much better!
> 
> 🚨 If you are using the new version of weights, please make sure that the new version of [wave_dynamic_layer.py](https://github.com/zhu-xlab/DOFA/blob/master/wave_dynamic_layer.py#L209) is used.
> 

--- 

## Dynamic One-For-All foundation model for Remote sensing and Earth observation

**Object Detection**: DOFA and DOFA v2 can be used for object detection tasks, which are fundamental remote sensing applications.
<p align="center">
<img src="assets/dofa_od.png" width="1200">
</p>

**What is DOFA**: DOFA is a unified multimodal foundation model for different data modalities in remote sensing and Earth observation.
<p align="center">
<img src="assets/DOFA-main.png" width="500">
</p>

**Differences with existing foundation models**: DOFA is pre-trained using five different data modalities in remote sensing and Earth observation. It can handle images with any number of input channels.

**DOFA is inspired by neuroplasticity** Neuroplasticity is an
important brain mechanism for adjusting to new experiences or environmental shifts. Inspired by this concept, we design DOFA to emulate this mechanism for processing multimodal EO data.

<p align="center">
<img src="assets/DOFA-model.png" width="500">
</p>


For more details, please take a look at the paper [Neural Plasticity-Inspired Foundation Model for Observing the Earth Crossing Modalities](https://arxiv.org/abs/2403.15356).

## Why develop DOFA
- The learned multimodal representation may not effectively capture such an intersensor relationship.

- The performance of foundation models will degrade when downstream tasks require the utilization of data from unseen sensors with varying numbers of spectral bands and spatial resolutions or different wavelength regimes.

- The development of individual, customized foundation models requires considerably more computing resources and human efforts.

- The increasing number of specialized foundation models makes it difficult to select the most appropriate one for a specific downstream task.

## Installation

### Requirements

The requirements of DOFA can be installed as follows:

```console
> pip install -r requirements.txt
```

### Weights

Pre-trained model weights can be downloaded from [HuggingFace](https://huggingface.co/XShadow/DOFA).

## Usage

Please refer to [demo.ipynb](https://github.com/zhu-xlab/DOFA/blob/master/demo.ipynb) for more details.

DOFA supports input images with any number of channels using our pre-trained foundation models. The following examples show how to use DOFA for **Sentinel-1 (SAR)**, **Sentinel-2**, **NAIP RGB**. We will add example usage for Gaofen Multispectral, and Hyperspectral data soon.

---

### Using `torch.hub` to Load the DOFA ViT Base Model

This snippet demonstrates how to load a ViT model—specifically, **DOFA ViT Base**—from a GitHub repository that includes a `hubconf.py` entrypoint. The model weights are hosted on Hugging Face via a direct download URL, so **no additional dependencies** beyond PyTorch are required.

```python
import torch

model = torch.hub.load(
    'zhu-xlab/DOFA',
    'vit_base_dofa',  # The entry point defined in hubconf.py
    pretrained=True,
)

model = model.cuda()
model.eval()
```

Now the model is ready for inference or further fine-tuning.
If you would like to fine-tune DOFA on different downstream tasks, please refer to [DOFA-pytorch](https://github.com/xiong-zhitong/DOFA-pytorch).


### TorchGeo

Alternatively, DOFA can be used via the [TorchGeo](https://github.com/microsoft/torchgeo) library:

```python
import torch
from torchgeo.models import DOFABase16_Weights, dofa_base_patch16_224

# Example NAIP image (wavelengths in $\mu$m)
x = torch.rand(2, 4, 224, 224)
wavelengths = [0.48, 0.56, 0.64, 0.81]

# Use pre-trained model weights
model = dofa_base_patch16_224(weights=DOFABase16_Weights.DOFA_MAE)

# Make a prediction (model may need to be fine-tuned first)
y = model(x, wavelengths)
```

---


### Demo usage of DOFA on different data modalities

---

### Download the pre-trained weights for DOFA from huggingface
Please move to the `checkpoints` directory and run
```python
python download_weights.py
```

Please download the `data.zip` for the demo usage from [HF](https://huggingface.co/earthflow/DOFA/resolve/main/data.zip). Then unzip it to the `data` directory.


```python
from models_dwv import vit_base_patch16

check_point = torch.load('./checkpoints/DOFA_ViT_base_e100.pth')
vit_model = vit_base_patch16()
msg = vit_model.load_state_dict(check_point, strict=False)
# missing_keys=['fc_norm.weight', 'fc_norm.bias', 'head.weight', 'head.bias'], unexpected_keys=['mask_token', 'norm.weight', 'norm.bias', 'projector.weight', 'projector.bias']
vit_model = vit_model.cuda()
```

Now you can use **the loaded single DOFA model** to process image data from **different modalities** with **any number of channels**!


### Preprare for the data loading and preprocessing

```python
# Step 1: Data preprocessing (normalization and resize)

import torch
import rasterio
import kornia as K
import numpy as np
# vh,vv
S1_MEAN = [166.36275909, 88.45542715]# / 255.0
S1_STD = [64.83126309, 43.07350145]# /255.0

S2_MEAN = [114.1099739 , 114.81779093, 126.63977424,  84.33539309,
        97.84789168, 103.94461911, 101.435633  ,  72.32804172,
        56.66528851]
S2_STD = [77.84352553, 69.96844919, 67.42465279, 64.57022983, 61.72545487,
       61.34187099, 60.29744676, 47.88519516, 42.55886798]

NAIP_MEAN = [123.675, 116.28, 103.53] # ImageNet stats for now
NAIP_STD = [58.395, 57.12, 57.375] # ImageNet stats for now

Gaufen_MEAN = [123.94924583,  92.58088583,  97.28130189,  90.31526596]
Gaufen_STD = [67.34487297, 62.8271046 , 60.5856767 , 60.3946299]
```


```python
class DataAugmentation(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.transform = torch.nn.Sequential(
            K.augmentation.RandomResizedCrop(size=(224,224), scale=(0.2,1.0)),
            K.augmentation.Normalize(mean=mean,std=std)
        )
    @torch.no_grad()
    def forward(self,x):
        x_out = self.transform(x)
        return x_out
```

### Load Sentinel-1 data with 2 channels

```python
transform = DataAugmentation(mean=S1_MEAN,std=S1_STD)

def preprocess_s1(vh_path, vv_path):
    with rasterio.open(vh_path) as f1:
        vh = f1.read()
    with rasterio.open(vv_path) as f2:
        vv = f2.read()
    s1_img = np.concatenate((vh,vv),0).astype('float32')
    s1_img = torch.from_numpy(s1_img)
    s1_img = transform(s1_img).squeeze(0)
    return s1_img
```


```python
import matplotlib.pyplot as plt
# Load Sentinel-1 images from the given example data
C = 2  # can be 2,3,4,6,9,12,13,202 or any number if you can provide the wavelengths of them

image1 = './data/s1/vv/1869_3575.png'
image2 = './data/s1/vh/1869_3575.png'
s1_img = preprocess_s1(image1,image2)

fig, ax = plt.subplots(nrows=1, ncols=C, figsize=(10, 10))

for i,row in enumerate(ax):
    row.imshow(s1_img[i])

s1_img = s1_img.view([1,2,224,224]).cuda()
```


    
![png](assets/demo_4_0.png)


```python
wavelengths = [5.405, 5.405]
out_feat = vit_model.forward_features(s1_img, wave_list=wavelengths)
out_logits = vit_model.forward(s1_img, wave_list=wavelengths)
print(out_feat.shape)
print(out_logits.shape)
```


### Load Sentinel-2 data with 9 channels



```python
import glob

C = 9

transform = DataAugmentation(mean=S2_MEAN,std=S2_STD)

def preprocess_s2(img_path):
    chs = []
    s2_files = glob.glob(f"{img_path}/*/*.png")
    for path in s2_files:
        with rasterio.open(path) as f:
            ch = f.read()
        chs.append(ch)
    s2_img = np.concatenate(chs, 0).astype("float32")
    s2_img = torch.from_numpy(s2_img)
    s2_img = transform(s2_img).squeeze(0)
    return s2_img
```

Visualize the Sentinel-2 imagery


```python
s2_img = preprocess_s2("data/s2/S2A_MSIL1C_20170528T050611_N0205_R076_T44NMM_20170528T050606")
fig, ax = plt.subplots(nrows=1, ncols=C, figsize=(10, 10))

for i,row in enumerate(ax):
    row.imshow(s2_img[i])
    row.axis("off")
s2_img = s2_img.view([1,C,224,224]).cuda()
```

    /opt/miniconda3/lib/python3.11/site-packages/rasterio/__init__.py:304: NotGeoreferencedWarning: Dataset has no geotransform, gcps, or rpcs. The identity matrix will be returned.
      dataset = DatasetReader(path, driver=driver, sharing=sharing, **kwargs)



    
![png](assets/demo_11_1.png)
    


### We use the same DOFA model to inference the Sentinel-2 image



```python

wavelengths = [0.665, 0.56, 0.49, 0.705, 0.74, 0.783, 0.842, 1.61, 2.19]

out_feat = vit_model.forward_features(s2_img, wave_list=wavelengths)
out_logits = vit_model.forward(s2_img, wave_list=wavelengths)
print(out_feat.shape)
print(out_logits.shape)
```

### What if I only want to use a subset of Sentinel-2 data?


```python
# Let's only keep the first 5 channels
wavelengths = [0.665, 0.56, 0.49, 0.705, 0.74]

out_feat = vit_model.forward_features(s2_img[:,:5,...], wave_list=wavelengths)
out_logits = vit_model.forward(s2_img[:,:5,...], wave_list=wavelengths)
print(out_feat.shape)
print(out_logits.shape)
```

### Usage for RGB optical data


```python
C = 3

transform = DataAugmentation(mean=NAIP_MEAN, std=NAIP_STD)


def preprocess_rgb(img_path):
    with rasterio.open(img_path) as f:
        rgb_img = f.read().astype("float32")
    rgb_img = torch.from_numpy(rgb_img)
    rgb_img = transform(rgb_img).squeeze(0)
    return rgb_img
```


```python
import cv2

rgb_path = 'data/naip/36861_49963.png'
naip_img = preprocess_rgb(rgb_path)

plt.imshow(cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB))
plt.axis("off")

naip_img = naip_img.view([1,C,224,224]).cuda()
```


    
![png](assets/demo_18_0.png)
    



```python
wavelengths = [0.665, 0.56, 0.49]

out_feat = vit_model.forward_features(naip_img, wave_list=wavelengths)
out_logits = vit_model.forward(naip_img, wave_list=wavelengths)
print(out_feat.shape)
print(out_logits.shape)
```

Usage for Hyperspectral images is similar to other images.



---

If you find the DOFA useful in your research, please kindly cite our paper:
```
@article{xiong2024neural,
  title={Neural Plasticity-Inspired Foundation Model for Observing the {Earth} Crossing Modalities},
  author={Xiong, Zhitong and Wang, Yi and Zhang, Fahong and Stewart, Adam J and Hanna, Jo{\"e}lle and Borth, Damian and Papoutsis, Ioannis and Saux, Bertrand Le and Camps-Valls, Gustau and Zhu, Xiao Xiang},
  journal={arXiv preprint arXiv:2403.15356},
  year={2024}
}
```
