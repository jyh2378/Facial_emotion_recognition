import os
import numpy as np
import h5py
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


class FacialDataset(Dataset):
    def __init__(self, split='Train', kfold=1, transform=None):
        self.transform = transform
        self.split = split  # training set or test set
        self.fold = kfold # the k-fold cross validation
        self.data = h5py.File("datasets/facial48.h5", 'r', driver='core')

        number = len(self.data['data_label']) # 2116
        sum_number = [0, 379, 952, 1263, 1890, 2116] # the sum of class number
        test_number = [19, 47, 63, 94, 105] # the number of each class

        test_index = []
        train_index = []
        for j in range(len(test_number)):
            for k in range(test_number[j]):
                if self.fold != 10:  # the last fold start from the last element
                    test_index.append(sum_number[j] + (self.fold - 1) * test_number[j] + k)
                else:
                    test_index.append(sum_number[j + 1] - 1 - k)

        for i in range(number):
            if i not in test_index:
                train_index.append(i)

        if self.split == 'Train':
            self.train_data = []
            self.train_labels = []
            for ind in range(len(train_index)):
                self.train_data.append(self.data['data_pixel'][train_index[ind]])
                self.train_labels.append(self.data['data_label'][train_index[ind]])

        elif self.split == 'Valid':
            self.test_data = []
            self.test_labels = []
            for ind in range(len(test_index)):
                self.test_data.append(self.data['data_pixel'][test_index[ind]])
                self.test_labels.append(self.data['data_label'][test_index[ind]])

    def __getitem__(self, index):
        if self.split == 'Train':
            img, target = self.train_data[index], self.train_labels[index]
        elif self.split == 'Valid':
            img, target = self.test_data[index], self.test_labels[index]

        img = img[:, :, np.newaxis]
        img = np.concatenate((img, img, img), axis=2)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        if self.split == 'Train':
            return len(self.train_data)
        elif self.split == 'Valid':
            return len(self.test_data)


class FacialDataLoader:
    def __init__(self, batch_size):

        train_transform = transforms.Compose([
            transforms.RandomCrop(112),
            transforms.RandomRotation(25),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.6287, 0.4851, 0.4217), (0.1988, 0.1799, 0.1705)),
            # transforms.Normalize((5209,), (0.1789,))
        ])

        valid_transform = transforms.Compose([
            transforms.CenterCrop(112),
            transforms.ToTensor(),
            # transforms.Normalize((0.6287, 0.4851, 0.4217), (0.1988, 0.1799, 0.1705))
            # transforms.Normalize((5209,), (0.1789,))
        ])

        trainset = FacialDataset(split='Train', kfold=1, transform=train_transform)
        validset = FacialDataset(split='Valid', kfold=1, transform=valid_transform)

        self.train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
        self.valid_loader = DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=0)

        self.train_iterations = len(self.train_loader)
        self.valid_iterations = len(self.valid_loader)


class CKDataset(Dataset):
    def __init__(self, split='Train', kfold=1, transform=None):
        self.transform = transform
        self.split = split  # training set or test set
        self.fold = kfold # the k-fold cross validation
        self.data = h5py.File("datasets/CK_data.h5", 'r', driver='core')

        number = len(self.data['data_label']) #981
        sum_number = [0,135,312,387,594,678,927,981] # the sum of class number
        test_number = [12,18,9,21,9,24,6] # the number of each class

        test_index = []
        train_index = []
        for j in range(len(test_number)):
            for k in range(test_number[j]):
                if self.fold != 10:  # the last fold start from the last element
                    test_index.append(sum_number[j] + (self.fold - 1) * test_number[j] + k)
                else:
                    test_index.append(sum_number[j + 1] - 1 - k)

        for i in range(number):
            if i not in test_index:
                train_index.append(i)

        if self.split == 'Train':
            self.train_data = []
            self.train_labels = []
            for ind in range(len(train_index)):
                self.train_data.append(self.data['data_pixel'][train_index[ind]])
                self.train_labels.append(self.data['data_label'][train_index[ind]])

        elif self.split == 'Valid':
            self.test_data = []
            self.test_labels = []
            for ind in range(len(test_index)):
                self.test_data.append(self.data['data_pixel'][test_index[ind]])
                self.test_labels.append(self.data['data_label'][test_index[ind]])

    def __getitem__(self, index):
        if self.split == 'Train':
            img, target = self.train_data[index], self.train_labels[index]
        elif self.split == 'Valid':
            img, target = self.test_data[index], self.test_labels[index]

        img = img[:, :, np.newaxis]
        img = np.concatenate((img, img, img), axis=2)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        if self.split == 'Train':
            return len(self.train_data)
        elif self.split == 'Valid':
            return len(self.test_data)


class CKDataLoader:
    def __init__(self, batch_size):

        train_transform = transforms.Compose([
            transforms.RandomCrop(44),
            transforms.RandomRotation(25),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        valid_transform = transforms.Compose([
            transforms.CenterCrop(44),
            transforms.ToTensor(),
        ])

        trainset = CKDataset(split='Train', kfold=1, transform=train_transform)
        validset = CKDataset(split='Valid', kfold=1, transform=valid_transform)

        self.train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
        self.valid_loader = DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=0)

        self.train_iterations = len(self.train_loader)
        self.valid_iterations = len(self.valid_loader)






# class CKDataLoader:
#     def __init__(self, batch_size):
#
#         train_transform = transforms.Compose([
#             # transforms.Grayscale(num_output_channels=1),
#             transforms.RandomCrop(44),
#             transforms.RandomRotation(25),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#         ])
#
#         valid_transform = transforms.Compose([
#             # transforms.Grayscale(num_output_channels=1),
#             transforms.CenterCrop(44),
#             transforms.ToTensor(),
#         ])
#
#         dataset = datasets.ImageFolder('datasets/CK+48', transform=train_transform)
#         dataset_no_aug = datasets.ImageFolder('datasets/CK+48', transform=valid_transform)
#         valid_ratio = 0.2
#         dataset_size = len(dataset)
#         indices = list(range(dataset_size))
#         split = int(np.floor(valid_ratio * dataset_size))
#         train_indices, val_indices = indices[split:], indices[:split]
#
#         train_dataset = Subset(dataset, train_indices)
#         valid_dataset = Subset(dataset_no_aug, val_indices)
#
#         self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
#         self.valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
#
#         self.train_iterations = len(self.train_loader)
#         self.valid_iterations = len(self.valid_loader)
