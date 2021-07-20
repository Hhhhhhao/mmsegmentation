_base_ = [
    '../dataset.py',
    '../runtime_40k.py'
]

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='UNet',
        in_channels=3,
        base_channels=64,
        num_stages=5,
        strides=(1, 1, 1, 1, 1),
        enc_num_convs=(2, 2, 2, 2, 2),
        dec_num_convs=(2, 2, 2, 2),
        downsamples=(True, True, True, True),
        enc_dilations=(1, 1, 1, 1, 1),
        dec_dilations=(1, 1, 1, 1),
        with_cp=False,
        conv_cfg=None,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU'),
        upsample_cfg=dict(type='InterpConv'),
        norm_eval=False),
    decode_head=dict(
        type='PSPHead',
        in_channels=64,
        in_index=4,
        channels=16,
        pool_scales=(1, 2, 3, 6),
        dropout_ratio=0.1,
        num_classes=8,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CombinedLoss', 
            losses=['CrossEntropyLoss', 'LovaszLoss'],
            lambdas=[1.0, 1.0],
            loss_weight=1.0,
            class_weight=[0.81613419, 1.02067147, 0.98366622, 0.801115, 0.9483719, 0.9817077, 0.96933678, 1.47899673])),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=128,
        in_index=3,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=8,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CombinedLoss', 
            losses=['CrossEntropyLoss', 'LovaszLoss'],
            lambdas=[1.0, 1.0],
            loss_weight=0.4,
            class_weight=[0.81613419, 1.02067147, 0.98366622, 0.801115, 0.9483719, 0.9817077, 0.96933678, 1.47899673])),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))


work_dir = "experiments/pspnet_unet_s5-d16_ce_lovasz_weighted/"
