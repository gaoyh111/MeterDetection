_base_ = [
    '_base_dbnetpp_resnet50-dcnv2_fpnc.py',
    '../_base_/default_runtime.py',
    '../_base_/datasets/pointmeter.py',
    '../_base_/schedules/schedule_sgd_1200e.py',
]

load_from = '/project/train/src_repo/mmocr/checkpoints/dbnetpp_resnet50-dcnv2_fpnc_1200e_icdar2015pth'  # noqa

# dataset settings
icdar2015_textdet_train = _base_.pointmeter_textdet_train
icdar2015_textdet_train.pipeline = _base_.train_pipeline
icdar2015_textdet_test = _base_.pointmeter_textdet_test
icdar2015_textdet_test.pipeline = _base_.test_pipeline


train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=icdar2015_textdet_train)

val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=icdar2015_textdet_test)

test_dataloader = val_dataloader

auto_scale_lr = dict(base_batch_size=16)


param_scheduler = [
    dict(type='PolyLR', power=0.9, eta_min=1e-7, end=90),
]


default_hooks=_base_.default_hooks
default_hooks.update(logger=dict(type='LoggerHook', interval=5,out_dir='/project/train/log'),
                     checkpoint=dict(type='CheckpointHook', interval=5,save_best='auto',max_keep_ckpts=1,out_dir='/project/train/models'))

train_cfg=_base_.train_cfg
train_cfg.update( max_epochs=90, val_interval=3)

val_evaluator=_base_.val_evaluator
val_evaluator.update(type='HmeanIOUMetric',strategy='max_matching')
test_evaluator=_base_.test_evaluator
test_evaluator = val_evaluator


