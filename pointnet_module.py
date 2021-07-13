import os
import sys;

import numpy as np

from datasets.data_modules import PartSegmentationDataModule
from datasets.shapenet_parts.shapenet_parts import ShapeNetParts

sys.path.append(os.getcwd())

from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from models.pointnet import PointNetEncoder, PointNetSegmentation, get_supervised_loss
from util.logger import get_logger
from util.training import to_categorical, test_val_shared_step, test_val_shared_epoch, inplace_relu, weights_init
import pdb


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
        num_classes: int = 16,
        num_seg_classes: int = 50,
        npoints: int = 2500,
        seg_class_map: dict = None,
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

        self.model = PointNetSegmentation(num_classes=num_seg_classes)
        self.loss_criterion = get_supervised_loss()
        self.num_classes = num_classes
        self.num_seg_classes = num_seg_classes
        self.npoints = npoints
        self.model.apply(inplace_relu)
        self.model.apply(weights_init)
        self.seg_class_map = seg_class_map

        self.seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
        for cat in self.seg_class_map.keys():
            for label in self.seg_class_map[cat]:
                self.seg_label_to_cat[label] = cat

    def training_step(self, batch, batch_idx):
        x, y, cls_id = batch
        prediction, trans_feat = self.model(x, to_categorical(cls_id, self.num_classes))

        prediction = prediction.contiguous().view(-1, self.num_seg_classes)
        target = y.view(-1, 1)[:, 0]
        pred_choice = prediction.data.max(1)[1]

        correct = pred_choice.eq(target.data).cpu().sum()
        mean_correct = correct.item() / (self.batch_size * self.npoints)

        loss = self.loss_criterion(prediction, target, trans_feat)
        self.log('train_loss', loss, on_step=True, on_epoch=False)
        return {'loss': loss, 'mean_correct': mean_correct}

    def training_epoch_end(self, training_step_outputs):
        mean_corrects = []
        for out in training_step_outputs:
            mean_corrects.append(out['mean_correct'])
        train_instance_acc = np.mean(mean_corrects)
        self.log('train_acc', train_instance_acc, on_step=False, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        x, y, cls_id = batch
        prediction, _ = self.model(x, to_categorical(cls_id, self.num_classes))
        # prediction = prediction.view(-1, self.num_seg_classes) # TODO: MODIFIED HERE

        return test_val_shared_step(x, y, prediction, self.seg_label_to_cat, self.seg_class_map, self.num_seg_classes)

    def validation_epoch_end(self, validation_epoch_outputs):
        shape_ious, accuracy, class_avg_accuracy, class_avg_iou, instance_avg_iou = test_val_shared_epoch(
            validation_epoch_outputs, num_seg_classes=self.num_seg_classes, seg_class_map=self.seg_class_map)

        self.log('val_accuracy', accuracy, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_class_avg_accuracy', class_avg_accuracy, on_step=False, on_epoch=True, sync_dist=True) # NAN

        for cat in sorted(shape_ious.keys()):
            self.log(f'eval mIoU of {cat}', shape_ious[cat], on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_class_avg_iou', class_avg_iou, on_step=False, on_epoch=True, sync_dist=True) # NAN
        self.log('val_instance_avg_iou', instance_avg_iou, on_step=False, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y, cls_id = batch
        prediction, _ = self.model(x, to_categorical(cls_id, self.num_classes))

        return test_val_shared_step(x, y, prediction, self.seg_label_to_cat, self.seg_class_map, self.num_seg_classes)

    def test_epoch_end(self, validation_epoch_outputs):
        shape_ious, accuracy, class_avg_accuracy, class_avg_iou, instance_avg_iou = test_val_shared_epoch(
            validation_epoch_outputs, num_seg_classes=self.num_seg_classes, seg_class_map=self.seg_class_map)
        self.log('test_accuracy', accuracy, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test_class_avg_accuracy', class_avg_accuracy, on_step=False, on_epoch=True, sync_dist=True) # NAN

        for cat in sorted(shape_ious.keys()):
            self.log(f'eval mIoU of {cat}', shape_ious[cat], on_step=False, on_epoch=True, sync_dist=True)
        self.log('test_class_avg_iou', class_avg_iou, on_step=False, on_epoch=True, sync_dist=True) # NAN
        self.log('test_instance_avg_iou', instance_avg_iou, on_step=False, on_epoch=True, sync_dist=True)

    def inference_step(self, x, cls_id):
        prediction, _ = self.model(x, to_categorical(cls_id, self.num_classes))
        prediction = prediction.contiguous().view(-1, self.num_seg_classes)
        pred_choice = prediction.data.max(1)[1]
        return pred_choice

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # model params

        # training params
        parser.add_argument("--fast_dev_run", default=0, type=int)
        parser.add_argument("--num_nodes", default=1, type=int, help="number of nodes for training")
        parser.add_argument("--gpus", default=1, type=int, help="number of gpus to train on")
        parser.add_argument("--num_workers", default=4, type=int, help="num of workers per GPU")
        parser.add_argument("--optimizer", default="adam", type=str, help="choose between adam/lars")
        parser.add_argument('--exclude_bn_bias', action='store_true', help="exclude bn/bias from weight decay")
        parser.add_argument("--max_epochs", default=200, type=int, help="number of total epochs to run")
        parser.add_argument("--max_steps", default=-1, type=int, help="max steps")
        parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
        parser.add_argument("--batch_size", default=32, type=int, help="batch size per gpu")
        parser.add_argument("--fp32", action='store_true')
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

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_checkpoint = ModelCheckpoint(save_last=True, save_top_k=1, monitor='val_instance_avg_iou', mode='max')
    callbacks = [model_checkpoint, lr_monitor]

    print(args.gpus)

    dm = PartSegmentationDataModule(batch_size=args.batch_size, fine_tuning=True, num_workers=args.num_workers)

    args.num_seg_classes = dm.num_seg_classes
    args.num_classes = dm.num_classes
    args.npoints = dm.npoints
    args.seg_class_map = dm.seg_class_map

    model = SupervisedPointNet(**args.__dict__)


    # dm.train_transforms = SimCLRTrainDataTransform([
    #     GaussianWhiteNoise(p=0.7),
    #     Rotation(0.5)
    # ])
    # dm.val_transforms = SimCLREvalDataTransform([
    #     GaussianWhiteNoise(p=0.7),
    #     Rotation(0.5)
    # ])

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

    trainer.fit(model, datamodule=dm)
    trainer.test(datamodule=dm)


if __name__ == '__main__':
    cli_main()
