# mmcls.datasets.pipelines.transforms에 mmcls의 albu register 따로 해줘야 함


# dataset settings
dataset_type = 'CustomDataset'
img_dir='/opt/ml/input/data/mmseg/images/'
ann_dir= '/opt/ml/input/data/mmseg/annotations/'

classes = ("Backgroud","General trash", "Paper", "Paper pack", "Metal", "Glass", 
            "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
palette =  [[0,0,0], [192,0,128], [0,128,192], [0,128,64], [128,0,0], [64,0,128],
           [64,0,192] ,[192,128,64], [192,192,128], [64,64,128], [128,0,192]]
crop_size = (512, 512)

# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

img_norm_cfg = dict(
    mean=[106.44883, 111.83318, 117.12853], std=[55.131298, 52.915947, 53.72297], to_rgb=True)

# train_all mean, std

albu_train_transforms = [
    
    
    dict(
        type='OneOf',
        transforms=[
            dict(type='HorizontalFlip'),
            dict(type='VerticalFlip')
                ],
        p=0.5),
    
#     dict(
#         type='OneOf',
#         transforms=[
#             dict(type='RandomRotate90'),
#             dict(type='ShiftScaleRotate',
#                 rotate_method='largest_box',
#                 p=0.75),
#                 ],
#         p=0.3),
    
#     dict(
#         type='OneOf',
#         transforms=[
#             dict(type='RandomBrightness', p=0.7),
#             dict(type='RandomContrast'),
#                 ],
#         p=0.3),
#     # dict(type='RandomBrightness',p=0.5),
    
#     dict(
#         type='OneOf',
#         transforms=[
#             dict(type='GaussianBlur'),
#             dict(type='ColorJitter'),
#                 ],
#         p=0.3),
    
#     dict(
#         type='OneOf',
#         transforms=[
#             dict(type='RandomRain'),
#             dict(type='RandomShadow',p=0.6),
#             dict(type='RandomSnow',p=0.7)
#                 ],
#         p=0.3),
    
]


train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='Resize', img_scale=(512,512), ratio_range=(0.5, 2.0)
# [(256,256),(384,384),(512,512),(640,640),(768,768)]
             # keep_ratio=True
            ),
        dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
        dict(type='RandomFlip', prob=0.5),
        
        # dict(
        # type='Albu',
        # transforms=albu_train_transforms,
        # keymap={
        #     'img': 'image',
        #     'gt_semantic_seg': 'mask',
        # },
        # update_pad_shape=False,
        # ),
        dict(type='PhotoMetricDistortion'),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg']),
    ]

val_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(512, 512),
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip',flip_ratio=0.0),
                dict(type='Normalize', **img_norm_cfg),
                # dict(type='Pad', size_divisor=32),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img']),
            ])
    ]

# test_pipeline = [
#         dict(type='LoadImageFromFile'),
#         dict(
#             type='MultiScaleFlipAug',
#             img_scale=(512,512),#[(1024, 1024),(512,512),(1333,800)],
#             # flip= False,
#             # flip_direction =  ["horizontal", "vertical" ],
#             transforms=[
#                 dict(type='Resize', keep_ratio=True),
#                 dict(type='RandomFlip',flip_ratio=0.0),
#                 dict(type='Normalize', **img_norm_cfg),
#                 dict(type='Pad', size_divisor=32),
#                 dict(type='ImageToTensor', keys=['img']),
#                 dict(type='Collect', keys=['img']),
#             ])
#     ]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5],  # for TTA
        flip=True,
        flip_direction=['horizontal', 'vertical'],
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]


# train=dict(
#         type=dataset_type,
#         ann_dir=ann_dir + 'train',
#         img_dir=img_dir + 'train',
#         classes = classes,
#         palette= palette,
#         pipeline=train_pipeline)

# leak=dict(
#         type=dataset_type,
#         ann_dir=ann_dir + 'leak',
#         img_dir=img_dir + 'leak',
#         classes = classes,
#         palette= palette,
#         pipeline=train_pipeline)

data = dict(
    samples_per_gpu=4, # 4
    workers_per_gpu=8, # 8 
    train =dict(
        type=dataset_type,
        ann_dir=ann_dir + 'split1_train_MultiStfKFold_pseudo',
        img_dir=img_dir + 'split1_train_MultiStfKFold_pseudo',
        classes = classes,
        palette= palette,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_dir=ann_dir + 'split1_val_MultiStfKFold',
        img_dir=img_dir + 'split1_val_MultiStfKFold',
        classes = classes,
        palette= palette,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        img_dir=img_dir+'test' ,
        classes = classes,
        palette= palette,
        pipeline=test_pipeline))

evaluation = dict(interval=1, save_best='mIoU', metric='mIoU', classwise=True)