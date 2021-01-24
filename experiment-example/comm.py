# model settings
model = dict(
    type='ImageClassifier',
    pretrained='torchvision://resnet101',
    backbone=dict(
        type='ResNet',
        depth=101,
        in_channels=16,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=2,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1,),
    ))

evaluation = dict(interval=10, metric=['accuracy', 'precision', 'recall', 'f1_score'])

# optimizer
optimizer = dict(type='SGD', lr=0.045, momentum=0.9, weight_decay=0.00004)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', gamma=0.98, step=1)
runner = dict(type='EpochBasedRunner', max_epochs=300)

# checkpoint saving
checkpoint_config = dict(interval=10)
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
