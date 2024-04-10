auto_scale_lr = dict(base_batch_size=256)
custom_imports = dict(imports='models')
data_preprocessor = dict(
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    num_classes=1000,
    std=[
        58.395,
        57.12,
        57.375,
    ],
    to_rgb=True)
dataset_type = 'ImageNet'
default_hooks = dict(
    checkpoint=dict(_scope_='mmpretrain', interval=1, type='CheckpointHook'),
    logger=dict(_scope_='mmpretrain', interval=100, type='LoggerHook'),
    param_scheduler=dict(_scope_='mmpretrain', type='ParamSchedulerHook'),
    sampler_seed=dict(_scope_='mmpretrain', type='DistSamplerSeedHook'),
    timer=dict(_scope_='mmpretrain', type='IterTimerHook'),
    visualization=dict(
        _scope_='mmpretrain', enable=False, type='VisualizationHook'))
default_scope = 'mmpretrain'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = None
log_level = 'INFO'
model = dict(
    _scope_='mmpretrain',
    backbone=dict(depth=18, type='ExampleNet'),
    head=dict(
        in_channels=512,
        loss=dict(loss_weight=1.0, type='CrossEntropyLoss'),
        num_classes=1000,
        topk=(
            1,
            5,
        ),
        type='LinearClsHead'),
    neck=dict(type='GlobalAveragePooling'),
    type='ImageClassifier')
optim_wrapper = dict(
    optimizer=dict(
        _scope_='mmpretrain',
        lr=0.1,
        momentum=0.9,
        type='SGD',
        weight_decay=0.0001))
param_scheduler = dict(
    _scope_='mmpretrain',
    by_epoch=True,
    gamma=0.1,
    milestones=[
        30,
        60,
        90,
    ],
    type='MultiStepLR')
randomness = dict(deterministic=False, seed=None)
resume = False
test_cfg = dict()
test_dataloader = dict(
    batch_size=32,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        _scope_='mmpretrain',
        data_root='data/imagenet',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(edge='short', scale=256, type='ResizeEdge'),
            dict(crop_size=224, type='CenterCrop'),
            dict(type='PackInputs'),
        ],
        split='val',
        type='ImageNet'),
    num_workers=5,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(_scope_='mmpretrain', shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    _scope_='mmpretrain', topk=(
        1,
        5,
    ), type='Accuracy')
test_pipeline = [
    dict(_scope_='mmpretrain', type='LoadImageFromFile'),
    dict(_scope_='mmpretrain', edge='short', scale=256, type='ResizeEdge'),
    dict(_scope_='mmpretrain', crop_size=224, type='CenterCrop'),
    dict(_scope_='mmpretrain', type='PackInputs'),
]
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
train_dataloader = dict(
    batch_size=32,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        _scope_='mmpretrain',
        data_root='data/imagenet',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(scale=224, type='RandomResizedCrop'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(type='PackInputs'),
        ],
        split='train',
        type='ImageNet'),
    num_workers=5,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(_scope_='mmpretrain', shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(_scope_='mmpretrain', type='LoadImageFromFile'),
    dict(_scope_='mmpretrain', scale=224, type='RandomResizedCrop'),
    dict(
        _scope_='mmpretrain',
        direction='horizontal',
        prob=0.5,
        type='RandomFlip'),
    dict(_scope_='mmpretrain', type='PackInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=32,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        _scope_='mmpretrain',
        data_root='data/imagenet',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(edge='short', scale=256, type='ResizeEdge'),
            dict(crop_size=224, type='CenterCrop'),
            dict(type='PackInputs'),
        ],
        split='val',
        type='ImageNet'),
    num_workers=5,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(_scope_='mmpretrain', shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    _scope_='mmpretrain', topk=(
        1,
        5,
    ), type='Accuracy')
vis_backends = [
    dict(_scope_='mmpretrain', type='LocalVisBackend'),
]
visualizer = dict(
    _scope_='mmpretrain',
    type='UniversalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/examplenet_8xb32_in1k'
