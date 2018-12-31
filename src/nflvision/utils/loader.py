# import cv2
import torch
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler
from matplotlib.image import imread

from nflvision.cfg.meta import PROJECT_HOME


class ImgLoader:

    @staticmethod
    def load_img(path):

        img = imread(path)
        # img = cv2.imread(path, mode='RGB')

        return img

    @classmethod
    def build(cls, image_size=256, batch_size=64, img_folder=PROJECT_HOME + "/data", valid_size=0.5, shuffle=False,
              seed=42, augment=False):

        train_transform = cls._get_pipeline_transform(image_size, augment=augment)
        valid_transform = cls._get_pipeline_transform(image_size, augment=False)

        train_img_dataset = datasets.ImageFolder(root=img_folder, transform=train_transform)
        valid_img_dataset = datasets.ImageFolder(root=img_folder, transform=valid_transform)

        train_sampler, valid_sampler = cls._get_samplers(train_img_dataset, valid_size, shuffle, seed)

        train_loader = torch.utils.data.DataLoader(train_img_dataset, batch_size=batch_size, sampler=train_sampler)
        valid_loader = torch.utils.data.DataLoader(valid_img_dataset, batch_size=batch_size, sampler=valid_sampler)

        return train_loader, valid_loader

    @staticmethod
    def _get_pipeline_transform(image_size, augment=False):

        pipeline = [transforms.Resize(image_size), transforms.RandomCrop(image_size)]

        if augment:
            pipeline += [transforms.RandomHorizontalFlip()]

        pipeline += [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        return transforms.Compose(pipeline)

    @staticmethod
    def _get_samplers(dataset, valid_size, shuffle, seed):

        num_train = len(dataset)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))

        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]

        return SubsetRandomSampler(train_idx), SubsetRandomSampler(valid_idx)
