import torch
from pathlib import Path
import json
import os
import os.path

import numpy as np
import torch
from tqdm import tqdm

class ShapeNetParts():

    def __init__(self, split, transform):
        assert split in ['train', 'val', 'test']

        self.root = os.getcwd()
        dataset_path = Path(
            "datasets/shapenet_parts/shapenetcore_partanno_segmentation_benchmark_v0/")  # path to point cloud data

        self.catfile = os.path.join(self.root,dataset_path,'synsetoffset2category.txt')

        self.cat = {}
        self.seg_classes = {}
        self.datapath = []
        self.npoints = 2500
        self.data_augmentation = False
        self.transform = transform

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]

        class_choice = ['Airplane']

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

    def __getitem__(self, index):
        point_set, seg = self.get_point_cloud_with_labels(index)

        return point_set, seg

    def __len__(self):
        return len(self.datapath)

    def get_all_data(self):
        pass

    def get_point_cloud_with_labels(self,index):
        fn = self.datapath[index]
        point_set = np.loadtxt(fn[1]).astype(np.float32)
        seg = np.loadtxt(fn[2]).astype(np.int64)
        #print(f"point_set.shape, seg.shape:{point_set.shape, seg.shape}")

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        #resample
        point_set = point_set[choice, :]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
        point_set = point_set / dist #scale

        seg = seg[choice]

        # if self.data_augmentation:
        #     theta = np.random.uniform(0,np.pi*2)
        #     rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
        #     point_set[:,[0,2]] = point_set[:,[0,2]].dot(rotation_matrix) # random rotation
        #     point_set += np.random.normal(0, 0.02, size=point_set.shape) # random jitter

        if self.transform:
            input = {"point": point_set, "seg": seg}
            input = self.transform(input)
            point_set = input['point']
            seg = input['seg']

        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)

        print(f"point_set: {point_set.shape}")
        print(f"seg: {seg.shape}")

        return point_set, seg
