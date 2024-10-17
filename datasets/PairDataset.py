import json
import math
import os
import random
import h5py
import torch
import numpy as np
import torch.utils.data as data
from .build import DATASETS
from utils.logger import *

def load_data_h5(file_path, subset, task):
    f = h5py.File(file_path, 'r')
    data = []
    label = []
    rotation = []
    if subset == 'test' and task == 'registration':
        rotation = f['rotation'][:]
    data = f['data'][:]
    label = f['label'][:]
    f.close()
    return data, label, rotation

def load_data_h5_test(file_path, subset, task, dataset):
    source_file_path = os.path.join(file_path, 'sources', dataset + '.h5')
    target_file_path = os.path.join(file_path, 'targets', dataset + '.h5')
    f_source = h5py.File(source_file_path, 'r')
    f_target = h5py.File(target_file_path, 'r')
    source_data = []
    target_data = []
    label = []
    rotation = []
    if subset == 'test' and task == 'registration':
        rotation = f_source['rotation'][:]
    source_data = f_source['data'][:]
    target_data = f_target['data'][:]
    label = f_source['label'][:]
    f_source.close()
    f_target.close()
    return source_data, target_data, label, rotation

@DATASETS.register_module()
class PairDataset(data.Dataset):
    def __init__(self, config, dataset, task, subset):
        self.data_root = config.data_path
        self.subset = subset
        self.npoints = config.npoints
        self.dataset = dataset
        self.task = task
        self.data = []
        self.target_data = []
        self.label = []
        self.rotation = []

        if self.subset == "train":
            file_path = os.path.join(self.data_root, 'Train_datasets', self.dataset + '.h5')
            self.data, self.label, self.rotation = load_data_h5(file_path, self.subset, self.task)
            print_log(f'[DATASET] Open file {file_path}', logger='DGPIC_Dataset')
        else:
            file_path = os.path.join(self.data_root, 'Test_datasets', self.task)
            self.data, self.target_data, self.label, self.rotation = load_data_h5_test(file_path, self.subset, self.task, self.dataset)
            print_log(f'[DATASET] Open source/target files in {file_path}', logger = 'DGPIC_Dataset')

        
        print_log(f'[DATASET] {self.dataset} dataset was loaded', logger = 'DGPIC_Dataset')

    def random_rotate_together(self, pointcloud1, pointcloud2, level=0):
        """
        Randomly rotate the point cloud
        :param pointcloud: input point cloud
        :param level: severity level
        :return: corrupted point cloud
        """
        angle_clip = math.pi / 3
        angle_clip = angle_clip / 3 * (level + 1)
        angles = np.random.uniform(-angle_clip, angle_clip, size=(3))
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
        R = np.dot(Rz, np.dot(Ry, Rx))
        pointcloud1 = np.dot(pointcloud1, R)
        pointcloud2 = np.dot(pointcloud2, R)
        return pointcloud1, pointcloud2

    def random_rotate(self, pointcloud, level=0):
        """
        Randomly rotate the point cloud
        :param pointcloud: input point cloud
        :param level: severity level
        :return: corrupted point cloud
        """
        angle_clip = math.pi / 3
        angle_clip = angle_clip / 3 * (level + 1)
        angles = np.random.uniform(-angle_clip, angle_clip, size=(3))
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
        R = np.dot(Rz, np.dot(Ry, Rx))
        pointcloud = np.dot(pointcloud, R)
        return pointcloud, R

    def y_flip(self, pointcloud1, pointcloud2):
        angles = [0, 0, math.pi]
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
        R = np.dot(Rz, np.dot(Ry, Rx))
        pointcloud1 = np.dot(pointcloud1, R)
        pointcloud2 = np.dot(pointcloud2, R)
        return pointcloud1, pointcloud2

    def y_flip_single(self, pointcloud1):
        angles = [0, 0, math.pi]
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
        R = np.dot(Rz, np.dot(Ry, Rx))
        pointcloud1 = np.dot(pointcloud1, R)
        return pointcloud1

    def random_dropout_global(self, pointcloud, level=0):
        """
        Drop random points globally
        :param pointcloud: input point cloud
        :param level: severity level
        :return: corrupted point cloud
        """
        # drop_rate = [0.25, 0.375, 0.5, 0.625, 0.75][level]
        drop_rate = [0.5, 0.75, 0.875, 0.9375, 0.96875][level]
        num_points = pointcloud.shape[0]
        # choice = random.sample(range(0, num_points), int(drop_rate * num_points))
        pointcloud[(1 - int(drop_rate * num_points)):, :] = 0
        return pointcloud

    def random_add_noise(self, pointcloud, level=0, sigma=0.2):
        """
        Randomly add noise data to point cloud
        :param pointcloud: input point cloud
        :param num_noise: number of noise points
        :return: corrupted point cloud
        """
        N, _ = pointcloud.shape
        num_noise = 100 * (level + 1)
        noise = np.clip(sigma * np.random.randn(num_noise, 3), -1, 1)
        idx = np.random.randint(0, N, num_noise)
        pointcloud[idx, :3] = pointcloud[idx, :3] + noise
        return pointcloud

    def __getitem__(self, idx):
        pointset = self.data[idx]
        rotation = np.ones([3, 3])
        if self.task == 'reconstruction':
            if self.subset == 'train':
                target = pointset.copy()
                pointset = self.random_dropout_global(pointset)
            else:
                target = self.target_data[idx]
        elif self.task == 'registration':
            if self.subset == 'train':
                target = pointset.copy()
                pointset, rotation = self.random_rotate(pointset)
                target = self.y_flip_single(target)
            else:
                rotation = self.rotation[idx]
                target = self.target_data[idx]
        elif self.task == 'denoising':
            if self.subset == 'train':
                target = pointset.copy()
                pointset = self.random_add_noise(pointset)
            else:
                target = self.target_data[idx]
        else:
            raise NotImplementedError()


        pointset = torch.from_numpy(pointset).float()
        target = torch.from_numpy(target).float()
        rotation = torch.from_numpy(rotation).float()

        return pointset, target, rotation, self.dataset

    def __len__(self):
        return self.data.shape[0]
