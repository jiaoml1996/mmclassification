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


evaluation = dict(interval=20, metric=['accuracy', 'precision', 'recall', 'f1_score'])

# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
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
        dict(type='PaviLoggerHook')
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
log_level = 'INFO'
# load_from = '/mnt/lustre/jiaomenglei/data/medical_by/实验数据/pretrained/medical_ct_AP03_0.843_AP05_0.620-d2a05ed8.pth'
resume_from = None
load_from = None
workflow = [('train', 1)]
