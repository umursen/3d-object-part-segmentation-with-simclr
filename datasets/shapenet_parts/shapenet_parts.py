from pathlib import Path
import json
import os
import os.path

import numpy as np
import torch
from torch.utils.data import Dataset


class ShapeNetParts(Dataset):

    def __init__(self, split, transforms=None, class_choice=None):
        assert split in ['train', 'val', 'test']

        self.root = os.getcwd()
        dataset_path = Path(
            "datasets/shapenet_parts/shapenetcore_partanno_segmentation_benchmark_v0/")  # path to point cloud data

        self.catfile = os.path.join(self.root, dataset_path, 'synsetoffset2category.txt')

        self.cat = {}
        self.seg_classes = {}
        self.datapath = []
        self.npoints = 2500
        self.data_augmentation = False

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]

        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        self.id2cat = {v: k for k, v in self.cat.items()}

        self.meta = {}
        splitfile = os.path.join(self.root,dataset_path,'train_test_split', 'shuffled_{}_file_list.json'.format(split))
        filelist = json.load(open(splitfile, 'r'))
        for item in self.cat:
            self.meta[item] = []

        for file in filelist:
            _, category, uuid = file.split('/')
            if category in self.cat.values():
                self.meta[self.id2cat[category]].append((os.path.join(self.root,dataset_path, category, 'points', uuid+'.pts'),
                                        os.path.join(self.root,dataset_path, category, 'points_label', uuid+'.seg')))

        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1]))

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))

        with open(os.path.join(self.root, 'datasets/shapenet_parts/misc/num_seg_classes.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.seg_classes[ls[0]] = int(ls[1])
        self.num_seg_classes = self.seg_classes[list(self.cat.keys())[0]]

        # Transforms
        self.transforms = transforms

    def __getitem__(self, index):
        point_set, seg = self.get_point_cloud_with_labels(index)

        if self.transforms:
            sample = {
                'point': point_set,
                'seg': seg
            }
            i, j = self.transforms(sample)

            x1, y1 = self.resample_points(i['point'], i['seg'])
            x2, y2 = self.resample_points(j['point'], j['seg'])
            x = torch.from_numpy(x1).T, torch.from_numpy(x2).T
            y = torch.from_numpy(y1).T, torch.from_numpy(y2).T

            # x = torch.from_numpy(i['point']).T, torch.from_numpy(j['point']).T
            # y = torch.from_numpy(i['seg']), torch.from_numpy(j['seg'])
        else:
            x, y = self.resample_points(point_set, seg)
            x = torch.from_numpy(x).T
            y = torch.from_numpy(y).T

        return x, y

    def __len__(self):
        return len(self.datapath)

    def get_point_cloud_with_labels(self,index):
        fn = self.datapath[index]
        point_set = np.loadtxt(fn[1]).astype(np.float32)
        seg = np.loadtxt(fn[2]).astype(np.int64)
        #print(f"point_set.shape, seg.shape:{point_set.shape, seg.shape}")
       # print(f'point_set_size: {point_set.shape}')
       # print(f'shape size: {seg.shape}')

        # if self.data_augmentation:
        #     theta = np.random.uniform(0,np.pi*2)
        #     rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
        #     point_set[:,[0,2]] = point_set[:,[0,2]].dot(rotation_matrix) # random rotation
        #     point_set += np.random.normal(0, 0.02, size=point_set.shape) # random jitter

        # if self.transform:
        #     input = {"point": point_set, "seg": seg}
        #     input = self.transform(input)
        #     point_set = input['point']
        #     seg = input['seg']

        # print(f"point_set: {point_set.shape}")
        # print(f"seg: {seg.shape}")

        # point_set, seg = self.resample_points(point_set, seg)

        return point_set, seg

    def resample_points(self, point_set, seg):

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        # resample
        point_set = point_set[choice, :]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale

        seg = seg[choice]
        return point_set, seg
