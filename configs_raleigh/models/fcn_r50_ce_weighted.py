_base_ = [
    '../dataset.py',
    '../runtime_20k.py'
]

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='FCNHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        num_convs=2,
        concat_input=True,
        dropout_ratio=0.1,
        num_classes=8,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,
            # class_weight=[0.81613419, 1.02067147, 0.98366622, 0.801115, 0.9483719, 0.9817077, 0.96933678, 1.47899673]
            class_weight=[9.7886e-04, 2.1092e-02, 1.3271e-02, 7.4172e-04, 7.8445e-03, 1.3065e-02, 1.0547e-02, 9.3245e-01]
            )),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=8,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4,
            # class_weight=[0.81613419, 1.02067147, 0.98366622, 0.801115, 0.9483719, 0.9817077, 0.96933678, 1.47899673]
            class_weight=[9.7886e-04, 2.1092e-02, 1.3271e-02, 7.4172e-04, 7.8445e-03, 1.3065e-02, 1.0547e-02, 9.3245e-01]
            )),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))


work_dir = "experiments/fcn_r50_ce_weighted/"
