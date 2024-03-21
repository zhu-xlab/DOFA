# DOFA

Dynamic One-For-All (DOFA) reference implementation.

## Installation

### Requirements

The requirements of DOFA can be installed as follows:

```console
> pip install -r requirements.txt
```

### Weights

Pre-trained model weights can be downloaded from [HuggingFace](https://huggingface.co/XShadow/DOFA).

## Usage

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
