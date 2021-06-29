import os
import sys;

from datasets.shapenet_parts.shapenet_parts import ShapeNetParts

sys.path.append(os.getcwd())

from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch import nn
from torch.nn import functional as F
from models.pointnet import PointNetEncoder, PointNetSegmentation
from util.logger import get_logger


class SupervisedPointNet(pl.LightningModule):

    def __init__(
        self,
        gpus: int,
        batch_size: int,
        num_nodes: int = 1,
        arch: str = 'pointnet',
        hidden_mlp: int = 2048,
        feat_dim: int = 128,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        temperature: float = 0.1,
        optimizer: str = 'adam',
        exclude_bn_bias: bool = False,
        start_lr: float = 0.,
        learning_rate: float = 1e-3,
        final_lr: float = 0.,
        weight_decay: float = 1e-6,
        **kwargs
    ):
        """
        Args:
            batch_size: the batch size
            num_samples: num samples in the dataset
            warmup_epochs: epochs to warmup the lr for
            lr: the optimizer learning rate
            opt_weight_decay: the optimizer weight decay
            loss_temperature: the loss temperature
        """
        super().__init__()
        self.save_hyperparameters()

        self.gpus = gpus
        self.num_nodes = num_nodes
        self.arch = arch
        self.batch_size = batch_size

        self.hidden_mlp = hidden_mlp
        self.feat_dim = feat_dim

        self.optim = optimizer
        self.exclude_bn_bias = exclude_bn_bias
        self.weight_decay = weight_decay
        self.temperature = temperature

        self.start_lr = start_lr
        self.final_lr = final_lr
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

        #self.model = PointNetSegmentation(ShapeNetParts.num_seg_classes)
        self.t = ShapeNetParts('train')
        self.v = ShapeNetParts('val')
        self.te = ShapeNetParts('test')
        print(self.t.num_seg_classes)
        print(self.v.num_seg_classes)
        print(self.te.num_seg_classes)
        self.loss_criterion = torch.nn.CrossEntropyLoss()



    def shared_step(self, y, prediction):

        iou = 0
        for part in range(ShapeNetParts.num_seg_classes):
            I = torch.sum(torch.logical_and(prediction == part, y == part))
            U = torch.sum(torch.logical_or(prediction == part, y == part))
            if U == 0:
                iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)

        _, predicted_label = torch.max(prediction, dim=1)

        total = predicted_label.numel()
        correct = (predicted_label == y).sum().item()
        accuracy = 100 * correct / total

        return accuracy, iou

    def train_dataloader(self):
        dataset = ShapeNetParts('train', transforms=self.train_transforms)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                           shuffle=True)

    def val_dataloader(self):
        dataset = ShapeNetParts('val', transforms=self.val_transforms)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                           shuffle=False)

    def test_dataloader(self):
        dataset = ShapeNetParts('test', transforms=self.val_transforms)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                           shuffle=False)

    def training_step(self, batch, batch_idx):
        x, y = batch
        prediction = self.model(x)
        loss = self.loss_criterion(prediction, y)
        self.log('train_loss', loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        prediction = self.model(x)
        loss = self.loss_criterion(prediction, y)
        acc, iou = self.shared_step(y, prediction)

        self.log({'val_loss': loss, 'val_acc': acc, 'val_iou': iou}, on_step=False, on_epoch=True, sync_dist=True)
        return {'loss': loss, 'accuracy': acc, 'iou': iou}

    def test_step(self, batch, batch_idx):
        x, y = batch
        prediction = self.model(x)
        loss = self.loss_criterion(prediction, y)
        acc, iou = self.shared_step(y, prediction)

        self.log({'test_loss': loss, 'test_acc': acc, 'test_iou': iou}, on_step=False, on_epoch=True, sync_dist=True)
        return {'loss': loss, 'accuracy': acc, 'iou': iou}

    def configure_optimizers(self, learning_rate):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # model params

        # training params
        parser.add_argument("--fast_dev_run", default=0, type=int)
        parser.add_argument("--num_nodes", default=1, type=int, help="number of nodes for training")
        parser.add_argument("--gpus", default=1, type=int, help="number of gpus to train on")
        parser.add_argument("--num_workers", default=8, type=int, help="num of workers per GPU")
        parser.add_argument("--optimizer", default="adam", type=str, help="choose between adam/lars")
        parser.add_argument('--exclude_bn_bias', action='store_true', help="exclude bn/bias from weight decay")
        parser.add_argument("--max_epochs", default=100, type=int, help="number of total epochs to run")
        parser.add_argument("--max_steps", default=-1, type=int, help="max steps")
        parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
        parser.add_argument("--batch_size", default=32, type=int, help="batch size per gpu")

        parser.add_argument("--weight_decay", default=0, type=float, help="weight decay")
        parser.add_argument("--learning_rate", default=1e-3, type=float, help="base learning rate")
        parser.add_argument("--start_lr", default=0, type=float, help="initial warmup learning rate")
        parser.add_argument("--final_lr", type=float, default=1e-6, help="final learning rate")

        return parser


def cli_main():

    parser = ArgumentParser()

    # model args
    parser = SupervisedPointNet.add_model_specific_args(parser)
    args = parser.parse_args()

    model = SupervisedPointNet(**args.__dict__)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_checkpoint = ModelCheckpoint(save_last=True, save_top_k=1, monitor='val_loss')
    callbacks = [model_checkpoint, lr_monitor]

    print(args.gpus)

    trainer = pl.Trainer(
        logger=get_logger(),
        max_epochs=args.max_epochs,
        max_steps=None if args.max_steps == -1 else args.max_steps,
        gpus=args.gpus,
        num_nodes=args.num_nodes,
        distributed_backend='ddp' if args.gpus > 1 else None,
        sync_batchnorm=True if args.gpus > 1 else False,
        precision=32 if args.fp32 else 16,
        callbacks=callbacks,
        fast_dev_run=args.fast_dev_run
    )

    trainer.fit(model)


if __name__ == '__main__':
    cli_main()
