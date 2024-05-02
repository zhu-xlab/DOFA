#
# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.datasets.basesegdataset import BaseSegDataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class SegMunich_Dataset(BaseSegDataset):
    """Gaofen Image Dataset (GID)

    Dataset paper link:
    https://arxiv.org/pdf/2311.07113
    https://github.com/danfenghong/IEEE_TPAMI_SpectralGPT/blob/main/downstream_tasks/SegMunich/TUM_128.py

    SegMunich 13 classes: Background, Arable land, Perm. Crops, Pastures, Forests, Surface water, Shrub, Open spaces, Wetlands, Mine dump, Artificial veg., Urban fabric, Buildings
    The ``img_suffix`` is fixed to '.tif' and ``seg_map_suffix`` is
    fixed to '.tif'.
    """
    METAINFO = dict(
        classes=('Background', 'Arable land', 'Perm. Crops', 'Pastures', 'Forests', 'Surface water', 
                    'Shrub', 'Open spaces', 'Wetlands', 'Mine dump', 'Artificial veg.', 'Urban fabric', 'Buildings'),
        palette=[[0, 0, 0], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                 [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
                 [107, 142, 35], [152, 251, 152], [70, 130, 180],
                 [220, 20, 60], [255, 0, 0]],
        label_map = {21:1,22:2,23:3,31:4,32:6,33:7,41:8,13:9,14:10},
        )

    def __init__(self,
                 img_suffix='.tif',
                 seg_map_suffix='.png',
                 reduce_zero_label=None,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)