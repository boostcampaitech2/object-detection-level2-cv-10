checkpoint_config = dict(max_keep_ckpts=5, interval=1)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook',
             init_kwargs=dict(project='mmdet-htc-swin', )),
        # name='')),
        # dict(type='TensorboardLoggerHook')
    ])

custom_hooks = [dict(type='NumClassCheckHook')]

fp16 = dict(loss_scale=dict(init_scale=512))
seed = 123456

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
