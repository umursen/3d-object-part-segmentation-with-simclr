from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset
import torch

from datasets.shapenet_parts.shapenet_parts import ShapeNetParts


class PartSegmentationDataModule(LightningDataModule):

    def __init__(self, batch_size, num_workers=1, limit_ratio=None, fine_tuning=False):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.limit_ratio = limit_ratio
        self.fine_tuning = fine_tuning

        self.num_seg_classes, self.num_classes, self.npoints, self.seg_class_map = self.get_number_of_seg_classes()

    def get_number_of_seg_classes(self):
        data = ShapeNetParts('train', transforms=self.train_transforms, limit_ratio=self.limit_ratio,
                             fine_tuning=self.fine_tuning)
        return data.num_seg_classes, data.num_classes, data.npoints, data.seg_class_map

    def train_dataloader(self):
        dataset = ShapeNetParts('train', transforms=self.train_transforms, limit_ratio=self.limit_ratio,
                                fine_tuning=self.fine_tuning)
        ##TODO: Should we shuffle? Last item comes with batch size 1 should we use drop last in this case or what can we do?
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                           shuffle=True, drop_last=True)

    def val_dataloader(self):
        dataset = ShapeNetParts('val', transforms=self.val_transforms, limit_ratio=None, fine_tuning=self.fine_tuning)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                           shuffle=False, drop_last=True)

    def test_dataloader(self):
        dataset = ShapeNetParts('test', transforms=None)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                           shuffle=False)
