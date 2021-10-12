# dataset settings
dataset_type = 'CocoDataset'
data_root = "/opt/ml/detection/dataset/"

classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", "Plastic",
           "Styrofoam", "Plastic bag", "Battery", "Clothing")

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        # img_scale=[(640, 640), (704, 704), (768, 768), (832, 832), (896, 896),
        #            (960, 960), (1024, 1024)],
        img_scale=[(640, 640), (768, 768), (896, 896), (1024, 1024)],
        multiscale_mode="value",
        # img_scale=(1024, 1024),
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiScaleFlipAug',
         img_scale=(1024, 1024),
         flip=False,
         transforms=[
             dict(type='Resize', keep_ratio=True),
             dict(type='RandomFlip'),
             dict(type='Normalize', **img_norm_cfg),
             dict(type='Pad', size_divisor=32),
             dict(type='ImageToTensor', keys=['img']),
             dict(type='Collect', keys=['img']),
         ])
]
data = dict(
    samples_per_gpu=13,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=
        "/opt/ml/Git/object-detection-level2-cv-10/data/stratified_train.10fold.wArea.json",
        # img_prefix=data_root + 'train/',
        img_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=
        "/opt/ml/Git/object-detection-level2-cv-10/data/stratified_valid.10fold.wArea.json",
        # img_prefix=data_root + 'train/',
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'test.json',
        #   img_prefix=data_root + 'test/',
        img_prefix=data_root,
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='bbox')
