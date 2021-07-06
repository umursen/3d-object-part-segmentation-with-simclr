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

import pdb

def to_categorical(y, num_classes):  #num
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)


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



    def shared_step(self, y, prediction):

        iou = 0
        for part in range(self.num_seg_classes):
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


    # TEST AND VAL
    def test_val_shared_step(self, batch):
        x, y, cls_id = batch
        cur_batch_size, _, NUM_POINT = x.size()
        prediction, _ = self.model(x, to_categorical(cls_id, self.num_classes))
        # prediction = prediction.view(-1, self.num_seg_classes) # TODO: MODIFIED HERE
        cur_pred_val = prediction.cpu().data.numpy()
        cur_pred_val_logits = cur_pred_val
        cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
        target = y.cpu().data.numpy()

        # loss = self.loss_criterion(prediction.contiguous().view(cur_batch_size, -1, self.num_seg_classes), target)
        # self.log('val_loss', loss, on_step=True, on_epoch=False)

        for i in range(cur_batch_size):
            cat = self.seg_label_to_cat[target[i, 0]]
            logits = cur_pred_val_logits[i, :, :]
            cur_pred_val[i, :] = np.argmax(logits[:, self.seg_class_map[cat]], 1) + self.seg_class_map[cat][0]

        correct = np.sum(cur_pred_val == target)
        total_correct = correct
        total_seen = (cur_batch_size * NUM_POINT)

        total_seen_class = np.zeros(self.num_seg_classes)
        total_correct_class = np.zeros(self.num_seg_classes)

        for l in range(self.num_seg_classes):
            total_seen_class[l] = np.sum(target == l)
            total_correct_class[l] = (np.sum((cur_pred_val == l) & (target == l)))

        shape_ious = {cat: [] for cat in self.seg_class_map.keys()}
        for i in range(cur_batch_size):
            segp = cur_pred_val[i, :]
            segl = target[i, :]
            cat = self.seg_label_to_cat[segl[0]]
            part_ious = np.zeros(len(self.seg_class_map[cat]))
            for l in self.seg_class_map[cat]:
                if (np.sum(segl == l) == 0) and (
                        np.sum(segp == l) == 0):  # part is not present, no prediction as well
                    part_ious[l - self.seg_class_map[cat][0]] = 1.0
                else:
                    part_ious[l - self.seg_class_map[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                        np.sum((segl == l) | (segp == l)))
            shape_ious[cat].append(np.mean(part_ious))

        # self.log('instance_avg_iou', iou, on_step=False, on_epoch=True, sync_dist=True)
        return {'total_correct': total_correct,
                'total_seen': total_seen, 'total_seen_class': total_seen_class,
                'total_correct_class': total_correct_class, 'shape_ious': shape_ious}

    def test_val_shared_epoch(self, outputs):
        total_correct = 0
        total_seen = 0
        total_seen_class = np.zeros(self.num_seg_classes)
        total_correct_class = np.zeros(self.num_seg_classes)
        shape_ious = {cat: [] for cat in self.seg_class_map.keys()}

        for output in outputs:
            total_correct += output['total_correct']
            total_seen += output['total_seen']
            total_seen_class += output['total_seen_class']
            total_correct_class += output['total_correct_class']

            for cat, values in output['shape_ious'].items():
                shape_ious[cat] += values

        all_shape_ious = []

        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat])
        mean_shape_ious = np.mean(list(shape_ious.values()))
        return shape_ious, \
               total_correct / float(total_seen),\
               np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float)),\
               mean_shape_ious,\
               np.mean(all_shape_ious)

    def validation_step(self, batch, batch_idx):
        return self.test_val_shared_step(batch)

    def validation_epoch_end(self, validation_epoch_outputs):
        shape_ious, accuracy, class_avg_accuracy, class_avg_iou, instance_avg_iou = self.test_val_shared_epoch(validation_epoch_outputs)
        self.log('val_accuracy', accuracy, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_class_avg_accuracy', class_avg_accuracy, on_step=False, on_epoch=True, sync_dist=True) # NAN

        for cat in sorted(shape_ious.keys()):
            print('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
        self.log('val_class_avg_iou', class_avg_iou, on_step=False, on_epoch=True, sync_dist=True) # NAN
        self.log('val_instance_avg_iou', instance_avg_iou, on_step=False, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        return self.test_val_shared_step(batch)

    def test_epoch_end(self, validation_epoch_outputs):
        shape_ious, accuracy, class_avg_accuracy, class_avg_iou, instance_avg_iou = self.test_val_shared_epoch(validation_epoch_outputs)
        self.log('test_accuracy', accuracy, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test_class_avg_accuracy', class_avg_accuracy, on_step=False, on_epoch=True, sync_dist=True) # NAN

        for cat in sorted(shape_ious.keys()):
            print('test mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
        self.log('test_class_avg_iou', class_avg_iou, on_step=False, on_epoch=True, sync_dist=True) # NAN
        self.log('test_instance_avg_iou', instance_avg_iou, on_step=False, on_epoch=True, sync_dist=True)

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
        parser.add_argument("--num_workers", default=8, type=int, help="num of workers per GPU")
        parser.add_argument("--optimizer", default="adam", type=str, help="choose between adam/lars")
        parser.add_argument('--exclude_bn_bias', action='store_true', help="exclude bn/bias from weight decay")
        parser.add_argument("--max_epochs", default=100, type=int, help="number of total epochs to run")
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

    dm = PartSegmentationDataModule(batch_size=args.batch_size, fine_tuning=True)

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
