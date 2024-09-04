# DOFA for semantic segmentation

DOFA is a versatile foundation model for remote sensing images from multi-sensors.  
Here, we show example usages for the semantic segmentation downstream tasks.
## Usage

### Setup Environment
Please install the [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) first.
You can either install it as a Python package
```python
pip install -U openmim
mim install mmengine
mim install "mmcv==2.1.0"
pip install "mmsegmentation>=1.0.0"
```
or refer to [Get Started](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/get_started.md#installation) to install it by other means.

At first, add the current folder to `PYTHONPATH`, so that Python can find your code. Run command in the current directory to add it.

> Please run it every time after you opened a new shell.

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

### Data Preparation

We use the [SegMunich](https://huggingface.co/datasets/XShadow/SegMunich) dataset as an example.
You can also prepare any datasets following the same directory structure.

###### 1. Please download the NWPU-RESISC45 dataset and extract the files to datasets folder. 
###### 2. Put the split files in dataset/NWPU-RESISC45-splits to the NWPU-RESISC45 folder.

Then the dataset structure should be:
```
dataset/SegMunich/
├── dataset/
│   ├── train.txt
│   ├── val.txt
|
├── train/
│   ├── img
│   |   ├── xxx.tif
│   |   ├── xxx.tif
│   ├── label
│   |   ├── xxx.tif
│   |   ├── xxx.tif
├── val/
│   ├── img
│   |   ├── xxx.tif
│   |   ├── xxx.tif
│   ├── label
│   |   ├── xxx.tif
│   |   ├── xxx.tif
...
```
More details about the dataset structure can be found at [instruction](https://mmsegmentation.readthedocs.io/en/latest/advanced_guides/add_datasets.html).


### Set up the path of pretrained weights 

Edit the #89 line of the config file: *configs/dofa_vit_seg.py* to set up the path to the pre-trained weights.


### Training commands

**To train with single GPU:**

```bash
mim train mmsegmentation configs/dofa_vit_seg.py
```

**To train with multiple GPUs:**

```bash
mim train mmsegmentation configs/dofa_vit_seg.py --launcher pytorch --gpus 8
```

**To train with multiple GPUs by slurm:**

```bash
mim train mmsegmentation configs/dofa_vit_seg.py --launcher slurm \
    --gpus 16 --gpus-per-node 8 --partition $PARTITION
```

## Results on SegMunich dataset (mIoU )

|      Class      |  IoU  |  Acc  |
|-----------------|-------|-------|
|    Background   | 89.38 | 95.62 |
|   Arable land   | 73.52 | 87.38 |
|   Perm. Crops   | 25.78 | 35.45 |
|     Pastures    | 52.19 | 69.24 |
|     Forests     |  84.2 | 91.29 |
|  Surface water  | 76.15 | 87.03 |
|      Shrub      | 17.96 | 24.15 |
|   Open spaces   |  34.7 | 49.39 |
|     Wetlands    | 41.04 | 54.36 |
|    Mine dump    | 39.54 | 51.51 |
| Artificial veg. | 24.37 | 32.43 |
|   Urban fabric  | 68.31 | 83.07 |
|    Buildings    | 53.73 | 68.35 |

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
