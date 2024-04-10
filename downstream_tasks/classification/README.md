# DOFA for image classification

DOFA is a versatile foundation model for remote sensing images from multi-sensors.  
Here, we show example usages for the image classification downstream task.
## Usage

### Setup Environment
Please install the [mmpretrain](https://github.com/open-mmlab/mmpretrain) first.
You can either install it as a Python package
```python
pip install -U openmim
mim install "mmpretrain>=1.0.0rc8"
```
or refer to [Get Started](https://mmpretrain.readthedocs.io/en/latest/get_started.html) to install MMPreTrain by other means.

At first, add the current folder to `PYTHONPATH`, so that Python can find your code. Run command in the current directory to add it.

> Please run it every time after you opened a new shell.

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

### Data Preparation

We use the [RESISC-45](https://1drv.ms/u/s!AmgKYzARBl5ca3HNaHIlzp_IXjs) dataset as an example.
You can also prepare any datasets following the same directory structure.

###### 1. Please download the NWPU-RESISC45 dataset and extract the files to datasets folder. 
###### 2. Put the split files in dataset/NWPU-RESISC45-splits to the NWPU-RESISC45 folder.

Then the dataset structure should be:
```
dataset/NWPU-RESISC45/
|
│train.txt
│val.txt
|
├── airplane/
│   ├── airplane_415.jpg
│   ├── airplane_261.jpg
│   ├── airplane_364.jpg
├── desert/
│   ├── desert_148.jpg
│   ├── desert_165.jpg
│   ├── desert_118.jpg
...
```
More details about the dataset structure can be found at [instruction](https://mmpretrain.readthedocs.io/en/latest/user_guides/dataset_prepare.html#imagenet).


### Set up the path of pretrained weights 

Edit the #90 line of the config file: *configs/dofa_base_resisc45.py* to set up the path to the pre-trained weights.


### Training commands

**To train with single GPU:**

```bash
mim train mmpretrain configs/dofa_base_resisc45.py
```

**To train with multiple GPUs:**

```bash
mim train mmpretrain configs/dofa_base_resisc45.py --launcher pytorch --gpus 8
```

**To train with multiple GPUs by slurm:**

```bash
mim train mmpretrain configs/dofa_base_resisc45.py --launcher slurm \
    --gpus 16 --gpus-per-node 8 --partition $PARTITION
```

## Results

|       Model        |   Pretrain   | Top-1 (%) | Top-5 (%) |                 Config                  |                Download                |
| :----------------: | :----------: | :-------: | :-------: | :-------------------------------------: | :------------------------------------: |
|  ResNet50   | From scratch |   89.33   |   -   | [config](./configs/resnet50_8xb32_in1k.py)  | [model]() \| [log]() |
| DOFA-base | [foundation model](https://huggingface.co/XShadow/DOFA) |   97.25   |   99.86   | [config](./configs/dofa_base_resisc45.py) | [model]() \| [log]() |
| DOFA-large  | [foundation model](https://huggingface.co/XShadow/DOFA) |   -   |   -   | [config]()  |        [model]() \| [log]() |

*More results are comming soon...*

In the next version, we will integrate DOFA to the [EarthNets](https://earthnets.github.io/) platform.
## Citation

<!-- Replace to the citation of the paper your project refers to. -->

```BibTeX
@misc{xiong2024earthnets,
      title={EarthNets: Empowering AI in Earth Observation}, 
      author={Zhitong Xiong and Fahong Zhang and Yi Wang and Yilei Shi and Xiao Xiang Zhu},
      year={2024},
      eprint={2210.04936},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{xiong2024neural,
      title={Neural Plasticity-Inspired Foundation Model for Observing the Earth Crossing Modalities}, 
      author={Zhitong Xiong and Yi Wang and Fahong Zhang and Adam J. Stewart and Joëlle Hanna and Damian Borth and Ioannis Papoutsis and Bertrand Le Saux and Gustau Camps-Valls and Xiao Xiang Zhu},
      year={2024},
      eprint={2403.15356},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```