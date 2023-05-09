import os

from PIL import Image
from torch.utils import data
import numpy as np
from config.config import cfg

processed_train_path = os.path.join(cfg.DATA.DATA_PATH, 'train')  # Path of training imgs
processed_val_path = os.path.join(cfg.DATA.DATA_PATH, 'val')  # Path of validation imgs


def default_loader(path):
    return Image.open(path)


def make_dataset(mode):
    # returns a list of all image paths of the training set (mode='train') or of the validation set (mode='val')
    images = []
    if mode == 'train':
        processed_train_img_path = processed_train_path
        processed_train_mask_path = cfg.DATA.DATA_PATH
        for img_name in os.listdir(processed_train_img_path):
            item = (os.path.join(processed_train_img_path, img_name),
                    os.path.join(processed_train_mask_path + '/labels/train/', img_name))
            images.append(item)
    elif mode == 'val':
        processed_val_img_path = processed_val_path
        processed_val_mask_path = cfg.DATA.DATA_PATH
        for img_name in os.listdir(processed_val_img_path):
            item = (os.path.join(processed_val_img_path, img_name),
                    os.path.join(processed_val_mask_path + '/labels/val/', img_name))
            images.append(item)
    return images


class resortit(data.Dataset):
    def __init__(self, mode, simul_transform=None, transform=None, target_transform=None):
        self.imgs = make_dataset(mode)  # Load imgs
        if len(self.imgs) == 0:
            raise (RuntimeError('Found 0 images, please check the data set'))
        self.loader = default_loader  # How to load imgs from the path
        # Define how to transform imgs
        self.simul_transform = simul_transform
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path, mask_path = self.imgs[index]
        img = self.loader(img_path)
        mask = np.array(self.loader(mask_path))
        # In the dataset we have 5 classes, for Binary Seg we need only two
        # All the trash categories becomes 1 -> mask[mask>0] = 1
        mask[mask > 0] = 1   ##########Only Binary Segmentation#####
        mask = Image.fromarray(mask)
        if self.simul_transform is not None:
            img, mask = self.simul_transform(img, mask)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        return img, mask

    def __len__(self):
        return len(self.imgs)
