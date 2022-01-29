from __future__ import print_function
import warnings
from PIL import Image
import os
import os.path
import numpy as np
import torch
import codecs
import string
import time
import math
from collections import OrderedDict
import random

class HAR():
    """
    Args:
        root (string): Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    class_set = ['Call','Hop','typing','Walk','Wave']
    label = [0,1,2,3,4] 
    DIMENSION_OF_FEATURE = 900
    NUM_OF_TOTAL_USERS = 120
    count_user_data = np.zeros(NUM_OF_TOTAL_USERS)
    classes = []

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(self, root, train=True, transform=None, target_transform=None, imgview=False,num_classes=5):
        
        self.train = train  # training set or test set
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train_test_ratio=0.9
        self.client_mapping=OrderedDict()
        self.client_label_distribution=OrderedDict()
        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You have to download it')

        # load class information
        self.data, self.targets = self.load_data(num_classes)


        # load data and targets

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return self.root

    @property
    def processed_folder(self):
        return self.root

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self):
        return (os.path.exists(self.processed_folder))

    def load_data(self,num_classes):

        # dataset append and split

        coll_class = []
        coll_label = []

        total_class = 0
        for current_user_id in range(1,self.NUM_OF_TOTAL_USERS+1):
            current_path = os.path.join(self.processed_folder, str(current_user_id))
            total_class_per_user=0
            cur_label_distribution=[0]*num_classes
            for class_id in range(num_classes):
                current_file = os.path.join(current_path, str(self.class_set[class_id]) + '_train.txt')

                if os.path.exists(current_file):

                    temp_original_data = np.loadtxt(current_file)
                    temp_reshape = temp_original_data.reshape(-1, 100, 10)
                    temp_coll = temp_reshape[:, :, 1:10].reshape(-1, self.DIMENSION_OF_FEATURE)
                    random.shuffle(temp_coll)
                    count_img = math.floor(temp_coll.shape[0]*self.train_test_ratio)
                    # print(temp_original_data.shape)
                    # print(temp_coll.shape)
                    cur_label_distribution[class_id]=count_img
                    if self.train:                       
                        temp_label = class_id * np.ones(count_img, dtype=int)
                        coll_class.extend(temp_coll[:count_img,:])
                        coll_label.extend(temp_label)
                        total_class_per_user+=count_img
                    else:
                        temp_label = class_id * np.ones(temp_coll.shape[0]-count_img-1, dtype=int)
                        coll_class.extend(temp_coll[count_img+1:,:])
                        coll_label.extend(temp_label)
                        total_class_per_user+=temp_coll.shape[0]-count_img-1
                    
            self.client_label_distribution[current_user_id-1]=cur_label_distribution
            self.client_mapping[current_user_id-1]=[i for i in range(total_class,total_class+total_class_per_user)]
            total_class+=total_class_per_user

        coll_class = np.array(coll_class)
        coll_label = np.array(coll_label)

        # print(coll_class.shape)
        # print(coll_label.shape)

        return coll_class, coll_label



