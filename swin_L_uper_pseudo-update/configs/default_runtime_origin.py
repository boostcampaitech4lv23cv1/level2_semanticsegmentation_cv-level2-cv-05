# yapf:disable
import datetime

checkpoint_config = dict(max_keep_ckpts=1, interval=1)

now = (datetime.datetime.now().replace(microsecond=0) + datetime.timedelta(hours=9)).strftime("%m-%d %H:%M")

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True),
        # dict(type='TensorboardLoggerHook')
        # dict(type='PaviLoggerHook') # for internal services
        dict(type='WandbLoggerHook',
         init_kwargs={
            'project': 'trash_segmentation',
            'entity': 'wandb name',
            'name' : f'mmseg_swinL_uper_CE_fold1_pseudo_defaultaug_{now}_pkt',
            'tags': ['swin_L', 'upernet'] 
            },
         interval=10, by_epoch=True
         # log_checkpoint=False,
         # log_checkpoint_metadata=False,
         # num_eval_images=10,
         # bbox_score_thr=0.3
        )
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
