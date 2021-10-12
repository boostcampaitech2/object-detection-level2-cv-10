_base_ = [
    'retinanet_r50_fpn.py',
    'dataset.py',
    'schedule_1x.py',
    'default_runtime.py'
]
# optimizer
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
