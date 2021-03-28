import os
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms


def get_transform(split):
    if split=='train':
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        #print('get train transforms')
    elif split=='valid':
        transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        #print('get valid transforms')
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        #print('get test transforms')
    return transform


class MelanomaDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: str, split: str, meta_names=None, transforms=None):
        self.df = df
        self.image_dir = image_dir
        self.split = split
        self.meta_names = meta_names
        self.transforms = transforms
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        # image
        image_path = os.path.join(self.image_dir, self.df.iloc[index]['image_name'] + '.jpg')
        image = Image.open(image_path)
        
        # meta
        meta = np.array(self.df.iloc[index][self.meta_names].values, dtype=np.float32)
        
        # transform
        if self.transforms:
            image = self.transforms(image)
        
        # return
        if self.split in ['train', 'valid']:
            target = self.df.iloc[index]['target']
            return (image, meta), target
        else:
            return (image, meta)


def get_dataset(config, df, image_dir, split, meta_names, transforms):
    dataset = MelanomaDataset(df, image_dir, split, meta_names, transforms)
    return dataset


def get_dataloader(config, df, image_dir, split, meta_names):
    transforms = get_transform(split)
    dataset = get_dataset(config, df, image_dir, split, meta_names, transforms)
    
    is_train = False
    if split=='train':
        batch_size = config.train.batch_size
        is_train = True
    elif split=='valid':
        batch_size = config.valid.batch_size
    else:
        batch_size = config.test.batch_size

    dataloader = DataLoader(dataset,
                            shuffle=is_train,
                            batch_size=batch_size,
                            drop_last=is_train,
                            num_workers=config.transform.num_preprocessor,
                            pin_memory=False)
    return dataloader

