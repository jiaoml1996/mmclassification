# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNeXt',
        in_channels=16,
        depth=101,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        style='pytorch',
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=2,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1,),
    ))


# dataset settings
dataset_type = 'Medical'
train_pipeline = [
    dict(type='LoadImageFromFile', mode='train'),
    dict(type='Resize', size=(224, 224)),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', flip_prob=0.5, direction='vertical'),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

val_pipeline = [
    dict(type='LoadImageFromFile', mode='val'),
    dict(type='Resize', size=(224, 224)),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

test_pipeline = [
    dict(type='LoadImageFromFile', mode='test'),
    dict(type='Resize', size=(224, 224)),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

data_root = '/mnt/lustre/jiaomenglei/code/medical_by/data/data_all/数据重新整理'
data_root_val = '/mnt/lustre/jiaomenglei/code/medical_by/data/data_all/数据重新整理'
data_root_test = '/mnt/lustre/jiaomenglei/code/medical_by/data/data_all/数据重新整理_验证组'

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=6,
    train=dict(
        type='RepeatDataset',
        times=64,
        dataset=dict(type=dataset_type,
            data_prefix=data_root,
            ann_file=data_root + '/训练索引/mri_矢状位/fold_4/train.json',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        data_prefix=data_root_val,
        ann_file=data_root_val + '/训练索引/mri_矢状位/fold_4/test.json',
        pipeline=val_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix=data_root_test,
        ann_file=data_root_test + '/测试索引/mri_矢状位_98良性恶性/test.json',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric=['accuracy', 'precision', 'recall', 'f1_score'])

# optimizer
optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)
# optimizer = dict(type='Adam', lr=0.001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 19])
runner = dict(type='EpochBasedRunner', max_epochs=20)

# checkpoint saving
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl', port=20001)
log_level = 'INFO'
# load_from = '/mnt/lustre/jiaomenglei/data/medical_by/实验数据/pretrained/medical_ct_AP03_0.843_AP05_0.620-d2a05ed8.pth'
resume_from = None
load_from = None
workflow = [('train', 1)]

