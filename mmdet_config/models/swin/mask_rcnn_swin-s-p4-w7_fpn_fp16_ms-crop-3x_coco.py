_base_ = [
    '../htc/htc_without_mask_r50_fpn_1x_coco.py',
    # '../htc/htc_without_mask_r50_fpn_1x_coco_focal.py',
    '../../datasets/recycle_dataset_albu.py',
    # '../../schedules/schedule_3x_AdamW.py',
    '../../schedules/schedule_1x_AdamW.py',
    '../../default_runtime.py',
]

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa TINY
# pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth'  # noqa SMALL

model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        # depths=[2, 2, 18, 2],  # Small
        depths=[2, 2, 6, 2],  # Tiny
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
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[96, 192, 384, 768]))
