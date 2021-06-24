from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset
import torch

from datasets.shapenet_parts.shapenet_parts import ShapeNetParts


# class USLDataset(Dataset):
#     def __init__(self, loader, transform=None, device='cuda:0'):
#         # self.X = point_data
#         self.loader = loader
#         self.transform = transform
#         self.device = device
#
#     def __len__(self):
#         return
#
#     def __getitem__(self, idx):
#
#
#         xi, xj = self.transform(self.X[idx])
#
#         xi = xi.to(device=self.device)
#         xj = xj.to(device=self.device)
#
#         x = (xi, xj)
#
#         return x, None

class PartSegmentationUSLDataModule(LightningDataModule):

    def __init__(self, batch_size, num_workers=1):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        dataset = ShapeNetParts('train', transforms=self.train_transforms)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        dataset = ShapeNetParts('val', transforms=self.val_transforms)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return None
