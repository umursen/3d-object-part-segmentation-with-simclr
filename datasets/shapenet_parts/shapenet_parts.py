from pathlib import Path
import json
import os
import os.path

import numpy as np
import torch
from torch.utils.data import Dataset
import pdb


class ShapeNetParts(Dataset):

    def __init__(self, split, limit_ratio=None, transforms=None, class_choice=None, fine_tuning=False):
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
        self.fine_tuning = fine_tuning

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
        self.num_classes = len(self.classes)

        with open(os.path.join(self.root, 'datasets/shapenet_parts/misc/num_seg_classes.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.seg_classes[ls[0]] = int(ls[1])
        self.num_seg_classes = sum(self.seg_classes.values())

        # Segmentation label mapping
        index = 0
        self.seg_class_map = {}
        for k, v in self.seg_classes.items():
            # if k not in self.seg_class_map.keys():
            self.seg_class_map[k] = []
            for class_index in range(v):
                self.seg_class_map[k].append(index)
                index += 1

        # Transforms
        self.transforms = transforms

        # Limit dataset
        if limit_ratio:
            limited_datapaths = []
            for k, v in self.cat.items():
                class_paths = list(filter(lambda x: k==x[0], self.datapath))
                selected_len = int(len(class_paths)*limit_ratio)
                if selected_len < 3:
                    selected_len = 3
                selected_class_paths = np.random.choice(len(class_paths), selected_len)
                limited_datapaths += list(np.asarray(class_paths)[selected_class_paths])
            self.datapath = limited_datapaths

    def __getitem__(self, index):
        point_set, seg, class_id = self.get_point_cloud_with_labels(index)

        if self.transforms:
            sample = {
                'point': point_set,
                'seg': seg
            }
            if self.fine_tuning:
                i = self.transforms(sample)

                x, y = self.resample_points(i['point'], i['seg'])
                x = torch.from_numpy(x).T
                y = torch.from_numpy(y).T
            else:
                i, j, x_online = self.transforms(sample)

                x1, y1 = self.resample_points(i['point'], i['seg'])
                x2, y2 = self.resample_points(j['point'], j['seg'])
                x_online, y_online = self.resample_points(x_online['point'], x_online['seg'])
                x = torch.from_numpy(x1).T, torch.from_numpy(x2).T, x_online.T
                y = torch.from_numpy(y1).T, torch.from_numpy(y2).T, y_online.T
        else:
            x, y = self.resample_points(point_set, seg)
            x = torch.from_numpy(x).T
            y = torch.from_numpy(y).T

        return x, y, class_id

    def __len__(self):
        return len(self.datapath)

    def get_point_cloud_with_labels(self, index):
        fn = self.datapath[index]
        point_set = np.loadtxt(fn[1]).astype(np.float32)
        seg = np.loadtxt(fn[2]).astype(np.int64)

        class_name = fn[0]
        cls_id = self.classes[class_name]

        seg = np.asarray(list(map(lambda k: self.seg_class_map[class_name][k-1], seg))).astype(np.int64)

        return point_set, seg, cls_id

    def resample_points(self, point_set, seg):

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        # resample
        point_set = point_set[choice, :]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale

        seg = seg[choice]
        return point_set, seg
