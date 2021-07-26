_base_ = [
    '../dataset.py',
    '../runtime_40k.py'
]

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='CascadeEncoderDecoder',
    num_stages=2,
    pretrained='open-mmlab://msra/hrnetv2_w48',
    backbone=dict(
        type='HRNet',
        norm_cfg=norm_cfg,
        norm_eval=False,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(48, 96)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(48, 96, 192)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(48, 96, 192, 384)))),
    decode_head=[
        dict(
            type='FCNHead',
            in_channels=[48, 96, 192, 384],
            channels=sum([48, 96, 192, 384]),
            in_index=(0, 1, 2, 3),
            input_transform='resize_concat',
            kernel_size=1,
            num_convs=1,
            concat_input=False,
            dropout_ratio=-1,
            num_classes=8,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='DiceLoss', 
                loss_weight=0.4,
                class_weight=[0.81613419, 1.02067147, 0.98366622, 0.801115, 0.9483719, 0.9817077, 0.96933678, 1.47899673])),
        dict(
            type='OCRHead',
            in_channels=[48, 96, 192, 384],
            in_index=(0, 1, 2, 3),
            input_transform='resize_concat',
            channels=512,
            ocr_channels=256,
            dropout_ratio=-1,
            num_classes=8,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='DiceLoss', 
                loss_weight=1.0,
                class_weight=[0.81613419, 1.02067147, 0.98366622, 0.801115, 0.9483719, 0.9817077, 0.96933678, 1.47899673])),
    ],
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))


work_dir = "experiments/ocrnet_hr48_dice_weighted/"
