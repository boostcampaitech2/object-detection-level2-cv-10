_base_ = [
    '../htc/htc_soft-nms_without_mask_r50_fpn_1x_coco.py',
    '../../datasets/recycle_dataset_albu.py',
    '../../schedules/schedule_20e_AdamW.py',
    '../../default_runtime.py',
]

# pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa TINY

model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 18, 2],  # Small
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        # init_cfg=dict(type='Pretrained', checkpoint=pretrained)
    ),
    neck=dict(in_channels=[96, 192, 384, 768]))

log_config = dict(interval=50,
                  hooks=[
                      dict(type='TextLoggerHook'),
                      dict(type='WandbLoggerHook',
                           init_kwargs=dict(project='mmdet-htc-swin', )),
                  ])

fp16 = dict(loss_scale=dict(init_scale=512))

seed = 123456