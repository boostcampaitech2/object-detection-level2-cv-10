# optimizer

# AdamW
optimizer = dict(type='AdamW',
                 lr=0.0002,
                 betas=(0.9, 0.999),
                 weight_decay=0.05,
                 paramwise_cfg=dict(
                     custom_keys={
                         'absolute_pos_embed': dict(decay_mult=0.),
                         'relative_position_bias_table': dict(decay_mult=0.),
                         'norm': dict(decay_mult=0.)
                     }))

# learning policy
lr_config = dict(policy='step',
                 warmup='linear',
                 warmup_iters=1000,
                 warmup_ratio=0.001,
                 step=[27, 33])

optimizer_config = dict(grad_clip=None)

# optimizer = dict(type='SGD', lr=0.04, momentum=0.9, weight_decay=0.0001)

runner = dict(type='EpochBasedRunner', max_epochs=36)
# runner = dict(max_epochs=36)
