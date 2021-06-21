from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset

from datasets.shapenet_parts.shapenet_parts import ShapeNetParts

import numpy as np
import torch


class USLDataset(Dataset):
    def __init__(self, point_data, transform=None, device='cuda:0'):
        self.X = point_data
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        xi, xj = self.transform(self.X[idx])

        xi = xi.to(device=self.device)
        xj = xj.to(device=self.device)

        x = (xi, xj)

        return x, None


class FineTuningDataset(Dataset):
    def __init__(self, point_data, labels, transform=None, device='cuda:0'):
        self.X = point_data
        self.Y = labels
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.X)
    #
    # def __getitem__(self, idx):
    #     xi, xj = self.transform(self.X[idx])
    #
    #     xi = xi.to(device=self.device)
    #     xj = xj.to(device=self.device)
    #
    #     x = (xi, xj)
    #
    #     return x, None



class PartSegmentationUSLDataModule(LightningDataModule):

    def __init__(self, batch_size, data_dir, num_workers=1):
        super().__init__()

        self.train_end_index = 0

        self.train_data = None
        self.val_data = None

        self.batch_size = batch_size
        self.data_dir = data_dir
        self.included_video_ids = {}
        self.num_workers = num_workers

    def prepare_data(self):
        self.train_data = ShapeNetParts('train').get_all_data()
        self.val_data = ShapeNetParts('val').get_all_data()


    def train_dataloader(self):
        dataset = USLDataset(point_data=self.train_data, transform=self.train_transforms)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        dataset = USLDataset(point_data=self.val_data, transform=self.val_transforms)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return None
