pointmeter_textdet_data_root = 'data/pointmeter'

pointmeter_textdet_train = dict(
    type='OCRDataset',
    data_root=pointmeter_textdet_data_root,
    ann_file='textdet_train.json',
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)

pointmeter_textdet_test = dict(
    type='OCRDataset',
    data_root=pointmeter_textdet_data_root,
    ann_file='textdet_test.json',
    test_mode=True,
    pipeline=None)
