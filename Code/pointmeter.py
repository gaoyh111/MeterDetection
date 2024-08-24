pointmeter_textrecog_data_root = 'data/pointmeter'

pointmeter_textrecog_train = dict(
    type='OCRDataset',
    data_root=pointmeter_textrecog_data_root,
    ann_file='textrecog_train.json',
    pipeline=None)

pointmeter_textrecog_test = dict(
    type='OCRDataset',
    data_root=pointmeter_textrecog_data_root,
    ann_file='textrecog_test.json',
    test_mode=True,
    pipeline=None)
