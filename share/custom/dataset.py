# dataset settings
dataset_type = 'CocoDataset'
data_root = '../../dataset/'




classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)


albu_train_transforms = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.0,
        rotate_limit=0,
        interpolation=1,
        p=0.5), 
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.2),
    dict(
        type='RGBShift',
        r_shift_limit=10,
        g_shift_limit=10,
        b_shift_limit=10,
        p=0.2),
    dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
    dict(type='ChannelShuffle', p=0.2),
    dict(type='CLAHE', p = 0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.2),
]


'''train_pipeline = [
    dict(type='Mosaic', img_scale=(1024, 1024)),
    dict(type='Resize', img_scale= (1024,1024), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio= [0.25, 0.25, 0.25] , direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='RandomShift'),
    dict(type='Normalize', **img_norm_cfg),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]'''


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale= (1024,1024), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio= [0.25, 0.25, 0.25] , direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='Normalize', **img_norm_cfg),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale= (1024,1024),
        flip=True,
        flip_direction = ['horizontal', 'vertical' ,'diagonal'],
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

''' train=dict(
        type = 'MultiImageMixDataset',
        dataset = dict(
        type=dataset_type,
        classes = classes,
        ann_file=data_root + 'stratified_train.json',
        img_prefix=data_root,
        pipeline=[dict(type='LoadImageFromFile'),
                  dict(type='LoadAnnotations', with_bbox=True)
                  ]),
        pipeline = train_pipeline,
        dynamic_scale= (1024, 1024)
),'''
'''train=dict(
        type=dataset_type,
        classes = classes,
        ann_file=data_root + 'stratified_train.json',
        img_prefix=data_root,
        pipeline=train_pipeline,
),'''
data = dict(
    samples_per_gpu= 8,
    workers_per_gpu= 2,
    train=dict(
        type=dataset_type,
        classes = classes,
        ann_file=data_root + 'stratified_train.json',
        img_prefix=data_root,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        classes = classes,
        ann_file=data_root + 'stratified_valid.json',
        img_prefix=data_root,
        pipeline=test_pipeline,
        ),
    test=dict(
        type=dataset_type,
        classes = classes,
        ann_file=data_root + 'test.json',
        img_prefix=data_root,
        pipeline=test_pipeline))

evaluation = dict(interval=1,
                  metric='bbox',
                  classwise = True,
                  save_best='bbox_mAP')
