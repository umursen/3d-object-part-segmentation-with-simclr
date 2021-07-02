from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset
import torch

from datasets.shapenet_parts.shapenet_parts import ShapeNetParts


class PartSegmentationDataModule(LightningDataModule):

    def __init__(self, batch_size, num_workers=1, limit_ratio=None):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.limit_ratio = limit_ratio

    def train_dataloader(self):
        dataset = ShapeNetParts('train', transforms=self.train_transforms, limit_ratio=self.limit_ratio)
        ##TODO: Should we shuffle? Last item comes with batch size 1 should we use drop last in this case or what can we do?
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                           shuffle=True, drop_last=True)

    def val_dataloader(self):
        dataset = ShapeNetParts('val', transforms=self.val_transforms, limit_ratio=None)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                           shuffle=False, drop_last=True)

    def test_dataloader(self):
        return None
