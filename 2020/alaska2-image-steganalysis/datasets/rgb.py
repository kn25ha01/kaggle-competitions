from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import RandomCrop, RandomHorizontalFlip, CenterCrop, ToTensor


class RGB_Dataset(Dataset):
    def __init__(self, images, labels=None, split='train', crop_size=224):
        self.images = images
        self.labels = labels
        self.split = split
        self.crop_size = crop_size
        print(f'{split} Dataset: RGB')
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]) # (3, 512, 512)
        
        if self.split=='train':
            image = RandomCrop((self.crop_size, self.crop_size))(image)
            image = RandomHorizontalFlip()(image)
            return {'image':ToTensor()(image), 'label':self.labels[idx]}
        
        elif self.split=='valid':
            image = CenterCrop((self.crop_size, self.crop_size))(image)
            return {'image':ToTensor()(image), 'label':self.labels[idx]}
        
        elif self.split=='test':
            image = CenterCrop((self.crop_size, self.crop_size))(image)
            return {'image':ToTensor()(image)}
        
        else:
            return None


