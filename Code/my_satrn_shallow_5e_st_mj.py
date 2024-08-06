_base_ = [
    '../_base_/datasets/pointmeter_lmdb.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_adam_step_5e.py',
    '_base_satrn_shallow.py',
]

load_from='/project/train/src_repo/mmocr/checkpoints/satrn_shallow_5e_st_mj.pth'

# dataset settings
pointmeter_lmdb_textrecog_train = _base_.pointmeter_lmdb_textrecog_train
pointmeter_lmdb_textrecog_train.pipeline = [
    dict(type='LoadImageFromNDArray', ignore_empty=True, min_size=2),
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(type='Resize', scale=(100, 32), keep_ratio=False),
    dict(
        type='PackTextRecogInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio'))
]

pointmeter_lmdb_textrecog_test = _base_.pointmeter_lmdb_textrecog_test
pointmeter_lmdb_textrecog_test.pipeline = [
    dict(type='LoadImageFromNDArray'),
    dict(type='Resize', scale=(100, 32), keep_ratio=False),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(
        type='PackTextRecogInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio'))
]

# optimizer
optim_wrapper = dict(type='OptimWrapper', optimizer=dict(type='Adam', lr=3e-4))

train_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=pointmeter_lmdb_textrecog_train)

test_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=pointmeter_lmdb_textrecog_test)


val_dataloader = test_dataloader

auto_scale_lr = dict(base_batch_size=64 * 8)

default_hooks=_base_.default_hooks
default_hooks.update(logger=dict(type='LoggerHook', interval=10,out_dir='/project/train/log'),
                     checkpoint=dict(type='CheckpointHook', interval=2,save_best='auto',max_keep_ckpts=1,out_dir='/project/train/models'))

train_cfg=_base_.train_cfg
train_cfg.update( max_epochs=30, val_interval=2)

param_scheduler=_base_.param_scheduler
param_scheduler = [
    dict(type='MultiStepLR', milestones=[8, 12], end=20),
]



