from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, Normalize, RandomCrop, RandomScale
from albumentations import OneOf, Compose
import ever as er
from albumentations import (
    CLAHE,
    RandomGamma,
    HueSaturationValue,
    IAAAdditiveGaussianNoise,
    GaussNoise,
    RandomBrightnessContrast,
    IAASharpen,
    IAAEmboss
)

TARGET_SET = 'REFUGE'
source_dir = dict(
    image_dir=[
        '.../REFUGE/REFUGE-Train/image/',
    ],
    mask_dir=[
        '.../REFUGE/REFUGE-Train/mask/',
    ],
)
target_dir = dict(
    image_dir=[
        '.../REFUGE/REFUGE-Val-0/Train/image/',
    ],
    mask_dir=[
        '.../REFUGE/REFUGE-Val-0/Train/mask/',
    ],
)
target_dir_Test = dict(
    image_dir=[
        '.../REFUGE/REFUGE-Val-0/Test/image/',
    ],
    mask_dir=[
        '.../REFUGE/REFUGE-Val-0/Test/mask/',
    ],
)

SOURCE_DATA_CONFIG = dict(
    image_dir=source_dir['image_dir'],
    mask_dir=source_dir['mask_dir'],
    transforms_image=Compose([
        OneOf([
            IAAAdditiveGaussianNoise(p=0.5),
            GaussNoise(p=0.5)
        ], p=0.2),

        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(p=0.5),
            IAAEmboss(p=0.5),
            RandomBrightnessContrast(p=0.5),
        ], p=0.2),
        HueSaturationValue(p=0.2),
        RandomGamma(p=0.2)
    ]),
    transforms=Compose([
        # RandomCrop(512, 512),
        OneOf([
            HorizontalFlip(True),
            VerticalFlip(True),
            RandomRotate90(True)
        ], p=0.75),
        Normalize(mean=(123.675, 116.28, 103.53),
                  std=(58.395, 57.12, 57.375),
                  max_pixel_value=1, always_apply=True),
        er.preprocess.albu.ToTensor()
    ]),
    CV=dict(k=10, i=-1),
    training=True,
    batch_size=2,
    num_workers=0,
)


TARGET_DATA_CONFIG = dict(
    image_dir=target_dir['image_dir'],
    mask_dir=target_dir['mask_dir'],
    transforms_image=Compose([
        OneOf([
            IAAAdditiveGaussianNoise(p=0.5),
            GaussNoise(p=0.5)
        ], p=0.2),

        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(p=0.5),
            IAAEmboss(p=0.5),
            RandomBrightnessContrast(p=0.5),
        ], p=0.2),
        HueSaturationValue(p=0.2),
        RandomGamma(p=0.2)
    ]),
    transforms=Compose([
        # RandomCrop(512, 512),
        OneOf([
            HorizontalFlip(True),
            VerticalFlip(True),
            RandomRotate90(True)
        ], p=0.75),
        Normalize(mean=(123.675, 116.28, 103.53),
                  std=(58.395, 57.12, 57.375),
                  max_pixel_value=1, always_apply=True),
        er.preprocess.albu.ToTensor()

    ]),
    CV=dict(k=10, i=-1),
    training=True,
    batch_size=2,
    num_workers=0,
)

EVAL_DATA_CONFIG = dict(
    image_dir=target_dir_Test['image_dir'],
    mask_dir=target_dir_Test['mask_dir'],
    transforms=Compose([
        Normalize(mean=(123.675, 116.28, 103.53),
                  std=(58.395, 57.12, 57.375),
                  max_pixel_value=1, always_apply=True),
        er.preprocess.albu.ToTensor()

    ]),
    CV=dict(k=10, i=-1),
    training=False,
    batch_size=1,
    num_workers=0,
)
