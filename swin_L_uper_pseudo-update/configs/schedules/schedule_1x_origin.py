# optimizer
# optimizer = dict(type='Adam', lr=0.0001)

optimizer = dict(
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.001,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# fp16 settings
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic', grad_clip=dict(max_norm=35, norm_type=2))
# fp16 placeholder
fp16 = dict()


# learning policy
# lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[30, 50, 60])
lr_config = dict(
        policy='CosineAnnealing',
        warmup='linear',
        warmup_iters=1000, 
        warmup_ratio=0.1,
        min_lr=1e-6,
    )


# runtime settings
# runner = dict(type='IterBasedRunner', max_iters=160000)
# evaluation = dict(interval=16000, metric='mIoU', pre_eval=True)
# checkpoint_config = dict(by_epoch=True, interval=1)

runner = dict(type='EpochBasedRunner', max_epochs=70)