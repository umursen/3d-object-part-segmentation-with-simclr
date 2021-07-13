from argparse import ArgumentParser

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from datasets.shapenet_parts.shapenet_parts import ShapeNetParts
from simclr_module import SimCLR

from typing import List

import pytorch_lightning as pl
import torch

from models.pointnet import PointNetDecoder, get_supervised_loss
from transforms import FineTuningEvalDataTransform, FineTuningTrainDataTransform
from datasets.data_modules import PartSegmentationDataModule
from augmentations.augmentations import *

from util.logger import get_logger
from util.training import inplace_relu, weights_init, to_categorical, test_val_shared_step, test_val_shared_epoch


class SSLFineTuner(pl.LightningModule):
    """
    Finetunes a self-supervised learning backbone.
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        epochs: int = 100,
        learning_rate: float = 0.1,
        weight_decay: float = 1e-6,
        nesterov: bool = False,
        scheduler_type: str = 'cosine',
        decay_epochs: List = [60, 80],
        gamma: float = 0.1,
        final_lr: float = 0.,
        num_classes: int = 16,
        num_seg_classes: int = 50,
        npoints: int = 2500,
        seg_class_map: dict = None,
        batch_size: int = 16
    ):
        """
        Args:
            backbone: a pretrained model
            in_features: feature dim of backbone outputs
            num_classes: classes of the dataset
            hidden_dim: dim of the MLP (1024 default used in self-supervised literature)
        """
        super().__init__()

        self.learning_rate = learning_rate
        self.nesterov = nesterov
        self.weight_decay = weight_decay

        self.scheduler_type = scheduler_type
        self.decay_epochs = decay_epochs
        self.gamma = gamma
        self.epochs = epochs
        self.final_lr = final_lr
        self.batch_size = batch_size

        self.backbone = backbone
        self.backbone.encoder.return_point_features = True

        # Define fine-tuning model
        self.decoder = PointNetDecoder(num_seg_classes)

        #
        self.loss_criterion = get_supervised_loss()
        self.num_classes = num_classes
        self.num_seg_classes = num_seg_classes
        self.npoints = npoints
        self.decoder.apply(inplace_relu)
        self.decoder.apply(weights_init)
        self.seg_class_map = seg_class_map

        self.seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
        for cat in self.seg_class_map.keys():
            for label in self.seg_class_map[cat]:
                self.seg_label_to_cat[label] = cat

    def on_train_epoch_start(self) -> None:
        self.backbone.eval()

    def training_step(self, batch, batch_idx):
        loss, prediction, y = self.shared_step(batch)

        prediction = prediction.contiguous().view(-1, self.num_seg_classes)
        target = y.view(-1, 1)[:, 0]
        pred_choice = prediction.data.max(1)[1]

        correct = pred_choice.eq(target.data).cpu().sum()
        mean_correct = correct.item() / (self.batch_size * self.npoints)

        self.log('train_loss', loss, on_step=True, on_epoch=False)
        return {'loss': loss, 'mean_correct': mean_correct}

    def training_epoch_end(self, training_step_outputs):
        mean_corrects = []
        for out in training_step_outputs:
            mean_corrects.append(out['mean_correct'])
        train_instance_acc = np.mean(mean_corrects)
        self.log('train_acc', train_instance_acc, on_step=False, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        x, y, class_id = batch
        loss, prediction, target = self.shared_step(batch)
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
        x, y, class_id = batch
        loss, prediction, target = self.shared_step(batch)
        return test_val_shared_step(x, y, prediction, self.seg_label_to_cat, self.seg_class_map, self.num_seg_classes)

    def test_epoch_end(self, validation_epoch_outputs):
        shape_ious, accuracy, class_avg_accuracy, class_avg_iou, instance_avg_iou = test_val_shared_epoch(
            validation_epoch_outputs, num_seg_classes=self.num_seg_classes, seg_class_map=self.seg_class_map)
        self.log('test_accuracy', accuracy, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test_class_avg_accuracy', class_avg_accuracy, on_step=False, on_epoch=True, sync_dist=True) # NAN

        for cat in sorted(shape_ious.keys()):
            self.log(f'test mIoU of {cat}', shape_ious[cat], on_step=False, on_epoch=True, sync_dist=True)
        self.log('test_class_avg_iou', class_avg_iou, on_step=False, on_epoch=True, sync_dist=True) # NAN
        self.log('test_instance_avg_iou', instance_avg_iou, on_step=False, on_epoch=True, sync_dist=True)

    def shared_step(self, batch):
        x, y, class_id = batch

        with torch.no_grad():
            representations, concat, trans_feat = self.backbone(x)

        prediction = self.decoder(
            representations,
            x.size(),
            to_categorical(class_id, self.num_classes),
            concat
        )

        prediction_flatten = prediction.contiguous().view(-1, self.num_seg_classes)
        target = y.view(-1, 1)[:, 0]

        loss = self.loss_criterion(prediction_flatten, target)

        return loss, prediction, y

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.decoder.parameters(),
            lr=self.learning_rate,
            nesterov=self.nesterov,
            momentum=0.9,
            weight_decay=self.weight_decay,
        )

        # set scheduler
        if self.scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.decay_epochs, gamma=self.gamma)
        elif self.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                self.epochs,
                eta_min=self.final_lr  # total epochs to run
            )

        return [optimizer], [scheduler]


def cli_main():
    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, help='dataset to train', default='shapenet')
    parser.add_argument('--ckpt_path', type=str, help='path to ckpt')

    parser.add_argument("--batch_size", default=16, type=int, help="batch size per gpu")
    parser.add_argument("--num_workers", default=8, type=int, help="num of workers per GPU")
    parser.add_argument("--gpus", default=1, type=int, help="number of GPUs")
    parser.add_argument('--num_epochs', default=100, type=int, help="number of epochs")

    # fine-tuner params
    parser.add_argument('--in_features', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--learning_rate', type=float, default=0.3)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--scheduler_type', type=str, default='cosine')
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--final_lr', type=float, default=0.)

    parser.add_argument('--nesterov', type=float, default=0.)

    args = parser.parse_args()

    if args.dataset == 'all':
        # TODO: Set data loader
        dm = ...
    elif args.dataset == 'shapenet':
        dm = PartSegmentationDataModule(
            args.batch_size,
            limit_ratio=0.1,
            fine_tuning=True
        )

        dm.train_transforms = FineTuningTrainDataTransform([
            RandomCuboid(p=1),
            GaussianNoise(0.7),
            Rescale(0.5)
        ])
        dm.val_transforms = FineTuningEvalDataTransform()

        args.num_seg_classes = dm.num_seg_classes
        args.num_classes = dm.num_classes
    elif args.dataset == 'coseg':
        # TODO: Set data loader
        dm = ...
    elif args.dataset == 'shapenet_toy_dataset':
        dm = ...
    else:
        raise NotImplementedError("other datasets have not been implemented till now")

    backbone = SimCLR(
        gpus=args.gpus,
        nodes=1,
        num_samples=0,
        batch_size=args.batch_size,
        dataset=args.dataset,
    ).load_from_checkpoint(args.ckpt_path, strict=False)

    tuner = SSLFineTuner(
        backbone,
        num_classes=args.num_classes,
        num_seg_classes=args.num_seg_classes,
        epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov,
        scheduler_type=args.scheduler_type,
        gamma=args.gamma,
        final_lr=args.final_lr,
        seg_class_map=dm.seg_class_map,
        batch_size=dm.batch_size
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_checkpoint = ModelCheckpoint(save_last=True, save_top_k=1, monitor='val_instance_avg_iou', mode='max')
    callbacks = [model_checkpoint, lr_monitor]

    trainer = pl.Trainer(
        logger=get_logger(),
        gpus=args.gpus,
        num_nodes=1,
        precision=16,
        max_epochs=args.num_epochs,
        distributed_backend='ddp' if args.gpus > 1 else None,
        sync_batchnorm=True if args.gpus > 1 else False,
        callbacks=callbacks,
    )

    trainer.fit(tuner, dm)
    trainer.test(datamodule=dm)


if __name__ == '__main__':
    cli_main()
