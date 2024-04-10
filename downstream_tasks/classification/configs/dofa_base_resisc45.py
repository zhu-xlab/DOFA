# Directly inherit the entire recipe you want to use.
_base_ = 'mmpretrain::_base_/default_runtime.py'

# dataset settings
dataset_type = 'CustomDataset'
data_preprocessor = dict(
    num_classes=45,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=224,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeEdge',
        scale=224,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=256,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root='./dataset/NWPU-RESISC45',
        ann_file='train.txt',      # The path of annotation file relative to the data_root.
        with_label=True,                # or False for unsupervised tasks
        classes=['airplane', 'airport', 'baseball_diamond', 'basketball_court', 'beach', 'bridge',\
            'chaparral', 'church', 'circular_farmland', 'cloud', 'commercial_area', 'dense_residential',\
            'desert', 'forest', 'freeway', 'golf_course', 'ground_track_field', 'harbor', 'industrial_area',\
            'intersection', 'island', 'lake', 'meadow', 'medium_residential', 'mobile_home_park', 'mountain',\
            'overpass', 'palace', 'parking_lot', 'railway', 'railway_station', 'rectangular_farmland', 'river',\
            'roundabout', 'runway', 'sea_ice', 'ship', 'snowberg', 'sparse_residential', 'stadium', 'storage_tank', \
            'tennis_court', 'terrace', 'thermal_power_station', 'wetland'], 
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=256,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root='./dataset/NWPU-RESISC45',
        ann_file='val.txt',      # The path of annotation file relative to the data_root.
        with_label=True,                # or False for unsupervised tasks
        classes=['airplane', 'airport', 'baseball_diamond', 'basketball_court', 'beach', 'bridge',\
            'chaparral', 'church', 'circular_farmland', 'cloud', 'commercial_area', 'dense_residential',\
            'desert', 'forest', 'freeway', 'golf_course', 'ground_track_field', 'harbor', 'industrial_area',\
            'intersection', 'island', 'lake', 'meadow', 'medium_residential', 'mobile_home_park', 'mountain',\
            'overpass', 'palace', 'parking_lot', 'railway', 'railway_station', 'rectangular_farmland', 'river',\
            'roundabout', 'runway', 'sea_ice', 'ship', 'snowberg', 'sparse_residential', 'stadium', 'storage_tank', \
            'tennis_court', 'terrace', 'thermal_power_station', 'wetland'], 
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1, 5))

test_dataloader = val_dataloader
test_evaluator = val_evaluator

# model setting
custom_imports = dict(imports='models')

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='vit_base_patch16',
        wave_list=[0.665, 0.56, 0.49],
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/home/zhitong/OFALL/OFALL_baseline/mae/github/DOFA/DOFA_ViT_base_e100.pth')),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=45,
        in_channels=768,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5))
)


# optimizer
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=6.25e-05, eps=1e-8, betas=(0.9, 0.999), weight_decay=0.05))

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        by_epoch=True,
        begin=0,
        end=20,
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        #T_max=280,
        T_max=80,
        by_epoch=True,
        begin=20,
        #end=300,
        end=100,
        eta_min=1.25e-06)
]

# train, val, test setting
#train_cfg = dict(by_epoch=True, max_epochs=300, val_interval=1)
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=128 * 8)
# This line is to import your own modules.
custom_imports = dict(imports='models')