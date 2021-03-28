import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from .rgb import RGB_Dataset
from .dtc import DCT_Dataset


def get_train_valid_data(cover_dir, stego_dir, test_size=0.1, random_state=2020, shuffle=True):
    # images
    all_img_names = sorted(os.listdir(cover_dir))
    
    # split
    train_img_names, valid_img_names = train_test_split(all_img_names, test_size=test_size, random_state=random_state, shuffle=shuffle)
    print(f'split : {len(train_img_names)}, {len(valid_img_names)}')
    
    # train images and labels
    x_train = [os.path.join(cover_dir, img_name) for img_name in train_img_names]
    x_train.extend([os.path.join(stego_dir, img_name) for img_name in train_img_names])
    y_train = [0] * len(train_img_names) + [1] * len(train_img_names)
    print(f'train images : {len(x_train)}')
    #print(f'train labels : {len(y_train)}')
    
    # valid images and labels
    x_valid = [os.path.join(cover_dir, img_name) for img_name in valid_img_names]
    x_valid.extend([os.path.join(stego_dir, img_name) for img_name in valid_img_names])
    y_valid = [0] * len(valid_img_names) + [1] * len(valid_img_names)
    print(f'valid images : {len(x_valid)}')
    #print(f'valid labels : {len(y_valid)}')
    
    return x_train, x_valid, y_train, y_valid


def get_local_test_data(cover_dir, stego_dir):
    # images
    all_img_names = sorted(os.listdir(cover_dir))
    
    # test images and labels
    x_test = [os.path.join(cover_dir, img_name) for img_name in all_img_names]
    x_test.extend([os.path.join(stego_dir, img_name) for img_name in all_img_names])
    y_test = [0] * len(all_img_names) + [1] * len(all_img_names)
    print(f'test images : {len(x_test)}')
    #print(f'test labels : {len(y_test)}')
    
    return x_test, y_test


def get_dataset(config, split, images, labels=None):
    f = globals().get(config.data.name)
    return f(images, labels, split, config.data.params.crop_size)


def get_dataloader(config, split, images, labels=None):
    dataset = get_dataset(config, split, images, labels)
    is_train = (split == 'train')
    batch_size = config.train.batch_size if is_train else config.eval.batch_size

    dataloader = DataLoader(dataset,
                            shuffle=is_train,
                            batch_size=batch_size,
                            drop_last=is_train,
                            num_workers=config.transform.num_preprocessor,
                            pin_memory=False)
    return dataloader


