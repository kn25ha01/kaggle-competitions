import random
import numpy as np
import jpegio as jio
import torch
from torch.utils.data import Dataset


def np_RandomCrop(image, crop_size=(224, 224)):
    height, width, _ = image.shape
    top = np.random.randint(0, height - crop_size[0])
    left = np.random.randint(0, width - crop_size[1])
    bottom = top + crop_size[0]
    right = left + crop_size[1]
    image = image[top:bottom, left:right, :]
    return image


def np_CenterCrop(image, crop_size=(224, 224)):
    height, width, _ = image.shape
    mid_h = height//2
    mid_w = width//2
    top = mid_h - crop_size[0]//2
    left = mid_w - crop_size[0]//2
    bottom = top + crop_size[0]
    right = left + crop_size[1]
    image = image[top:bottom, left:right, :]
    return image


def np_RandomHorizontalFlip(image, p=0.5):
    if random.random() >= p:
        image = image[:, ::-1, :]
    return image


def np_ToTensor(image):
    image = image.transpose((2, 0, 1)) # (512, 512, 3) -> (3, 512, 512)
    image = torch.from_numpy(np.flip(image, axis=0).copy())
    return image


class DCT_Dataset(Dataset):
    def __init__(self, images, labels=None, split='train', crop_size=224):
        self.images = images
        self.labels = labels
        self.split = split
        self.crop_size = crop_size
        print(f'{split} Dataset: DCT')
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = np.zeros([512, 512, 3], dtype=np.int8) # (512, 512, 3)
        jimage = jio.read(self.images[idx])
        for c in range(3):
            image[:, :, c] = jimage.coef_arrays[c]
        
        if self.split=='train':
            image = np_RandomCrop(image, crop_size=(self.crop_size, self.crop_size))
            image = np_RandomHorizontalFlip(image)
            return {'image':np_ToTensor(image), 'label':self.labels[idx]}
        
        elif self.split=='valid':
            image = np_CenterCrop(image, crop_size=(self.crop_size, self.crop_size))
            return {'image':np_ToTensor(image), 'label':self.labels[idx]}
        
        elif self.split=='test':
            image = np_CenterCrop(image, crop_size=(self.crop_size, self.crop_size))
            return {'image':np_ToTensor(image)}
        
        else:
            return None


