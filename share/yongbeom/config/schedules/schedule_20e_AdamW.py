# optimizer

# AdamW
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    # lr=0.00005,
    # lr=0.00001,
    # lr=0.000005,
    # lr=0.000001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

# learning policy
lr_config = dict(policy='step', step=[11, 19])
# step=[16, 19])

optimizer_config = dict(grad_clip=None)

runner = dict(type='EpochBasedRunner', max_epochs=24)
