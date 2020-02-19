# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision import datasets
import os
import torch


def get_dataset(cls):
    MEAN = [0.49139968, 0.48215827, 0.44653124]
    STD = [0.24703233, 0.24348505, 0.26158768]
    transf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip()
    ]
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]

    train_transform = transforms.Compose(transf + normalize)
    valid_transform = transforms.Compose(normalize)

    if cls == "cifar10":
        dataset_train = CIFAR10(root="./data", train=True, download=True, transform=train_transform)
        dataset_valid = CIFAR10(root="./data", train=False, download=True, transform=valid_transform)

    elif cls == "ntu120":
        resize = 224 
        data_dir = os.environ["DATASET_FOLDER"]+"/ntu/ntu_no_border/cross_subject"
        data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(224),
            #transforms.Resize(self.hparams.transform_resize),
            transforms.ToTensor(),
        ]),
        'test': transforms.Compose([
            transforms.Resize(224),
            # transforms.Resize((self.hparams.transform_resize)),
             transforms.ToTensor(),
        ]),
        }

        dataset_train = datasets.ImageFolder(os.path.join(data_dir, "train"),
                                          data_transforms["train"])
        
        dataset_valid = datasets.ImageFolder(os.path.join(data_dir, "test"),
                                                  data_transforms["test"])
    else:
        raise NotImplementedError
        


    return dataset_train, dataset_valid
