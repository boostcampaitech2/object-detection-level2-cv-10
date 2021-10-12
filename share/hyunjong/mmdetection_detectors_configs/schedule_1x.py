# # optimizer
# optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)
# # optimizer = dict(
# #     #_delete_=True,
# #     type='AdamW',
# #     #type='SGD',
# #     lr=0.0001,
# #     #momentum=0.9,
# #     betas=(0.9, 0.999),
# #     weight_decay=0.05,
# #     paramwise_cfg=dict(
# #         custom_keys={
# #             'absolute_pos_embed': dict(decay_mult=0.),
# #             'relative_position_bias_table': dict(decay_mult=0.),
# #             'norm': dict(decay_mult=0.)
# #         })
# #         )

# optimizer_config = dict(grad_clip=None)
# # learning policy
# # lr_config = dict(
# #     policy='CosineAnnealing',
# #     warmup='linear',
# #     warmup_iters=1000,
# #     warmup_ratio=1.0 / 10,
# #     min_lr_ratio=1e-5)
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[8, 11])
# runner = dict(type='EpochBasedRunner', max_epochs=12)

# optimizer
# optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=None)
# # learning policy
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[8, 11])
# runner = dict(type='EpochBasedRunner', max_epochs=12)

# optimizer = dict(
#     #_delete_=True,
#     type='AdamW',
#     lr=0.0001,
#     betas=(0.9, 0.999),
#     weight_decay=0.05,
#     paramwise_cfg=dict(
#         custom_keys={
#             'absolute_pos_embed': dict(decay_mult=0.),
#             'relative_position_bias_table': dict(decay_mult=0.),
#             'norm': dict(decay_mult=0.)
#         }))
# lr_config = dict(warmup_iters=1000, step=[8, 11])
# runner = dict(max_epochs=12)


# optimizer
optimizer = dict(type='AdamW', lr=0.0003, weight_decay=0.0001)
optimizer_config = dict(
     grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
# lr_config = dict(
#     policy='step',
#     warmup='linear', 
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[8, 11])
lr_config = dict(  # Learning rate scheduler config used to register LrUpdater hook
    policy='step',  # The policy of scheduler, also support CosineAnnealing, Cyclic, etc. Refer to details of supported LrUpdater from https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py#L9.
    warmup='linear',  # The warmup policy, also support `exp` and `constant`.
    warmup_iters=500,  # The number of iterations for warmup
    warmup_ratio=
    0.001,  # The ratio of the starting learning rate used for warmup
    step=[8, 11])  # Steps to decay the learning rate
runner = dict(type='EpochBasedRunner', max_epochs=12)
