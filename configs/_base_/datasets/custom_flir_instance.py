dataset_type = 'CocoDataset'
data_root = 'data/flir_custom/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
# custom classes for rodeo ground
custom_classes = ('person', 'bicycle', 'vehicle', 'motorcycle', 'fixed wing', 'military_person',
               'train', 'military_vehicle', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'phantom', 'toothbrush')


# custom classes for test data
# custom_classes = ('person', 'bicycle', 'vehicle', 'motorcycle', 'airplane', 'bus',
#                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
#                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
#                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
#                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
#                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
#                'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
#                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
#                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
#                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
#                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
#                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
#                'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
#                'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
# for original <200 img ATR dataset
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + '/train/random_ATR_anno.json',
        img_prefix=data_root + 'train/',
        pipeline=train_pipeline,
        classes=custom_classes),
    val=dict(
        type=dataset_type,
        ann_file=data_root + '/val/random_ATR_anno.json',
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline,
        classes=custom_classes),
    test=dict(
        type=dataset_type,
        ann_file=data_root + '/val/random_ATR_anno.json',
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline,
        classes=custom_classes))

# for 2.5k dataset 2
# data = dict(
#     samples_per_gpu=2,
#     workers_per_gpu=2,
#     train=dict(
#         type=dataset_type,
#         ann_file=data_root + '/train/None.json',
#         img_prefix=data_root + 'train/',
#         pipeline=train_pipeline,
#         classes=custom_classes),
#     val=dict(
#         type=dataset_type,
#         ann_file=data_root + '/val/None.json',
#         img_prefix=data_root + 'val/',
#         pipeline=test_pipeline,
#         classes=custom_classes),
#     test=dict(
#         type=dataset_type,
#         ann_file=data_root + '/val/None.json',
#         img_prefix=data_root + 'val/',
#         pipeline=test_pipeline,
#         classes=custom_classes))
evaluation = dict(metric=['bbox', 'segm'])
# data = dict(
#     samples_per_gpu=2,
#     workers_per_gpu=2,
#     train=dict(
#         type=dataset_type,
#         ann_file=data_root + '/train/rvss2_val_v3.json',
#         img_prefix=data_root + 'train/',
#         pipeline=train_pipeline),
#     val=dict(
#         type=dataset_type,
#         ann_file=data_root + '/val/rvss2_val_v3.json',
#         img_prefix=data_root + 'val/',
#         pipeline=test_pipeline),
#     test=dict(
#         type=dataset_type,
#         ann_file=data_root + '/val/rvss2_val_v3.json',
#         img_prefix=data_root + 'val/',
#         pipeline=test_pipeline))
# evaluation = dict(metric=['bbox', 'segm'])
