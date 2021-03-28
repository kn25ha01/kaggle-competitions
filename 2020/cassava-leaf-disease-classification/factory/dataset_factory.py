import os
import cv2
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data.dataset import Dataset


def get_transform(mode, width, height, logger=None):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if mode == 'train':
        transform = A.Compose([
            A.RandomResizedCrop(width, height, p=1.0),
            A.Transpose(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            A.CoarseDropout(p=0.5),
            A.Cutout(p=0.5),
            #A.OneOf([
            #    A.RandomGridShuffle(grid=(2, 2), p=1),
            #    A.RandomGridShuffle(grid=(3, 3), p=1),
            #    A.RandomGridShuffle(grid=(4, 4), p=1),
            #], p=0.5),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0), # (0, 255) -> mean=0.0 and std=1.0
            ToTensorV2(p=1.0), # to tensor
        ], p=1.0)

    elif mode == 'valid':
        transform = A.Compose([
            A.CenterCrop(width, height, p=1.0),
            A.Resize(width, height, p=1.0),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)

    else:
        transform = A.Compose([
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)

    if logger:
        logger.debug(transform)

    return transform


class CassavaDataset(Dataset):

    def __init__(self, df: pd.DataFrame, config: dict, mode: str, logger=None):
        self.df = df
        self.mode = mode
        self.image_dir = config['image_dir']
        self.transform = get_transform(self.mode, config['width'], config['height'], logger)

        if logger:
            logger.debug(f"mode         : {self.mode}")
            logger.debug(f"dataset name : {config['name']}")
            logger.debug(f"image dir    : {self.image_dir}")
            logger.debug(f'images       : {self.df.shape[0]}')
            logger.debug(f"value counts : {self.df['label'].value_counts()}")

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index: int):
        image_path = os.path.join(self.image_dir, self.df.loc[index, 'image_id'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)['image']

        if self.mode in ['train', 'valid']:
            label = self.df.loc[index, 'label']
            return image, label

        else:
            return image


def split_dataframe():
    pass


def get_CassavaDataset(df: pd.DataFrame, config: dict, mode: str, logger=None):
    dataset = CassavaDataset(df, config, mode, logger)
    return dataset


def get_dataset(df, config: dict, mode: str, logger=None):
    f = globals().get('get_' + config['name'])
    return f(df, config, mode, logger)
