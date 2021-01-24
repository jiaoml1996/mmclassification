_base_ = [
    './comm.py',
]

dataset_type = 'Medical'
# dataset settings
train_pipeline = [
    dict(type='LoadImageFromFile', mode='train'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', flip_prob=0.5, direction='vertical'),
    dict(type='RandomCrop', size=(64, 64)),
    dict(type='Resize', size=(224, 224)),
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
data_root_val = '/mnt/lustre/jiaomenglei/code/medical_by/data/data_all/数据重新整理_验证组'
data_root_test = '/mnt/lustre/jiaomenglei/code/medical_by/data/data_all/数据重新整理_验证组'

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=6,
    train=dict(
        type='ClassBalancedDataset',
        oversample_thr=10,
        dataset=dict(type=dataset_type,
            data_prefix=data_root,
            ann_file=data_root + '/训练索引/mri_矢状位/all_train/train.json',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        data_prefix=data_root_val,
        ann_file=data_root_val + '/测试索引/mri_矢状位_98良性恶性/test.json',
        pipeline=val_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix=data_root_test,
        ann_file=data_root_test + '/测试索引/mri_矢状位_98良性恶性/test.json',
        pipeline=test_pipeline))

dist_params = dict(backend='nccl', port=20001)