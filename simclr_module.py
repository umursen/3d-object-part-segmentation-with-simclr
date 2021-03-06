import os
import sys; sys.path.append(os.getcwd())

import math
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch import nn
from torch.nn import functional as F

from transforms import SimCLRTrainDataTransform, SimCLREvalDataTransform

from optimizers.lars import LARS
from optimizers.lr_scheduler import linear_warmup_decay

from models.pointnet import PointNetEncoder
from datasets.data_modules import PartSegmentationDataModule
from augmentations.augmentations import *
from util.logger import get_logger
import pdb

from callbacks.online_evaluator import SSLOnlineEvaluator


class Projection(nn.Module):

    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim), nn.BatchNorm1d(self.hidden_dim), nn.ReLU(),
            # nn.Linear(self.hidden_dim, self.hidden_dim), nn.BatchNorm1d(self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False)
        )

    def forward(self, x):
        x = self.model(x)
        return F.normalize(x, dim=1)


class SimCLR(pl.LightningModule):

    def __init__(
        self,
        gpus: int,
        num_samples: int,
        batch_size: int,
        dataset: str,
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
        save_frequency: int = 50,
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
        self.dataset = dataset
        self.num_samples = num_samples
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

        self.encoder = self.init_model()

        self.projection = Projection(input_dim=self.hidden_mlp, hidden_dim=self.hidden_mlp, output_dim=self.feat_dim)

        self.save_frequency = save_frequency

        # compute iters per epoch
        global_batch_size = self.num_nodes * self.gpus * self.batch_size if self.gpus > 0 else self.batch_size
        self.train_iters_per_epoch = self.num_samples // global_batch_size

    def init_model(self):
        if self.arch == 'pointnet':
            backbone = PointNetEncoder(return_point_features=True)
        else:
            raise ValueError('Unknown architecture!')

        return backbone

    def forward(self, x):
        return self.encoder(x)

    def shared_step(self, batch):
        # final image in tuple is for online eval
        (img1, img2, _), _, cls_id = batch

        # get h representations
        h1, _, _ = self(img1)
        h2, _, _ = self(img2)

        # get z representations
        z1 = self.projection(h1)
        z2 = self.projection(h2)

        loss = self.nt_xent_loss(z1, z2, self.temperature)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)

        self.log('train_loss', loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)

        self.log('val_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        if self.trainer.current_epoch % self.save_frequency == 0:
            logger_save_path = f'{self.logger.save_dir}/3dpart-simclr/{self.logger._id}/checkpoints'
            print(f'Logger save dir: {logger_save_path}')
            self.trainer.save_checkpoint(filepath=f'{logger_save_path}/every_{self.save_frequency}_epoch_{self.trainer.current_epoch}.ckpt')

    def exclude_from_wt_decay(self, named_params, weight_decay, skip_list=['bias', 'bn']):
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [{
            'params': params,
            'weight_decay': weight_decay
        }, {
            'params': excluded_params,
            'weight_decay': 0.,
        }]

    def configure_optimizers(self):
        if self.exclude_bn_bias:
            params = self.exclude_from_wt_decay(self.named_parameters(), weight_decay=self.weight_decay)
        else:
            params = self.parameters()

        if self.optim == 'lars':
            optimizer = LARS(
                params,
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay,
                trust_coefficient=0.001,
            )
        elif self.optim == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)

        warmup_steps = self.train_iters_per_epoch * self.warmup_epochs
        total_steps = self.train_iters_per_epoch * self.max_epochs

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                linear_warmup_decay(warmup_steps, total_steps, cosine=True),
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]

    def nt_xent_loss(self, out_1, out_2, temperature, eps=1e-6):
        """
            assume out_1 and out_2 are normalized
            out_1: [batch_size, dim]
            out_2: [batch_size, dim]
        """
        # gather representations in case of distributed training
        # out_1_dist: [batch_size * world_size, dim]
        # out_2_dist: [batch_size * world_size, dim]

        out_1_dist = out_1
        out_2_dist = out_2

        # out: [2 * batch_size, dim]
        # out_dist: [2 * batch_size * world_size, dim]
        out = torch.cat([out_1, out_2], dim=0)
        out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)

        # cov and sim: [2 * batch_size, 2 * batch_size * world_size]
        # neg: [2 * batch_size]
        cov = torch.mm(out, out_dist.t().contiguous())
        sim = torch.exp(cov / temperature)
        neg = sim.sum(dim=-1)

        # from each row, subtract e^(1/temp) to remove similarity measure for x1.x1
        row_sub = torch.Tensor(neg.shape).fill_(math.e**(1 / temperature)).to(neg.device)
        neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

        # Positive similarity, pos becomes [2 * batch_size]
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)

        loss = -torch.log(pos / (neg + eps)).mean()

        return loss

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # model params
        parser.add_argument("--arch", default="pointnet", type=str, help="architecture")
        # specify flags to store false
        parser.add_argument("--hidden_mlp", default=2048, type=int, help="hidden layer dimension in projection head")
        parser.add_argument("--feat_dim", default=128, type=int, help="feature dimension")
        parser.add_argument("--online_ft", default=0, type=int, help='online fine tuning')
        parser.add_argument("--fp32", action='store_true')

        # transform params
        parser.add_argument("--dataset", type=str, default="shapenet", help="dataset to train")
        parser.add_argument("--data_dir", type=str, default=".", help="path to download data")

        # training params
        parser.add_argument("--fast_dev_run", default=0, type=int)
        parser.add_argument("--num_nodes", default=1, type=int, help="number of nodes for training")
        parser.add_argument("--gpus", default=1, type=int, help="number of gpus to train on")
        parser.add_argument("--num_workers", default=4, type=int, help="num of workers per GPU")
        parser.add_argument("--optimizer", default="adam", type=str, help="choose between adam/lars")
        parser.add_argument('--exclude_bn_bias', action='store_true', help="exclude bn/bias from weight decay")
        parser.add_argument("--max_epochs", default=100, type=int, help="number of total epochs to run")
        parser.add_argument("--max_steps", default=-1, type=int, help="max steps")
        parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
        parser.add_argument("--batch_size", default=128, type=int, help="batch size per gpu")
        parser.add_argument("--save_frequency", default=50, type=int, help="save model every n epoch")

        parser.add_argument("--temperature", default=0.1, type=float, help="temperature parameter in training loss")
        parser.add_argument("--weight_decay", default=1e-6, type=float, help="weight decay")
        parser.add_argument("--learning_rate", default=1e-3, type=float, help="base learning rate")
        parser.add_argument("--start_lr", default=0, type=float, help="initial warmup learning rate")
        parser.add_argument("--final_lr", type=float, default=1e-6, help="final learning rate")

        return parser


def cli_main():

    parser = ArgumentParser()

    # model args
    parser = SimCLR.add_model_specific_args(parser)
    args = parser.parse_args()

    if args.dataset == 'all':
        # TODO: Set data loader
        dm = ...
    elif args.dataset == 'shapenet':
        dm = PartSegmentationDataModule(
            args.batch_size,
            #limit_ratio=0.05,
            num_workers=args.num_workers
        )

        dm.train_transforms = SimCLRTrainDataTransform([
            GaussianNoise(0.7),
            Rotation(0.5)
        ])
        dm.val_transforms = SimCLREvalDataTransform([
            GaussianNoise(0.7),
            Rotation(0.5)
        ])
    elif args.dataset == 'coseg':
        # TODO: Set data loader
        dm = ...
    elif args.dataset == 'shapenet_toy_dataset':
        dm = ...
    else:
        raise NotImplementedError("other datasets have not been implemented till now")

    args.num_samples = len(dm.train_dataloader().dataset)

    model = SimCLR(**args.__dict__)

    online_evaluator = None

    if args.online_ft:
        online_evaluator = SSLOnlineEvaluator(
            num_classes=dm.num_classes,
            num_seg_classes=dm.num_seg_classes,
            npoints=dm.npoints,
            seg_class_map=dm.seg_class_map
        )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_checkpoint = ModelCheckpoint(save_last=True, save_top_k=1, monitor='val_loss')
    model_checkpoint_online = ModelCheckpoint(save_last=True, save_top_k=1, monitor='online_val_accuracy')
    callbacks = [model_checkpoint, online_evaluator, model_checkpoint_online] if args.online_ft else [model_checkpoint, lr_monitor]
    
    # TODO Q: Shouldn't we add this? 
    #callbacks.append(lr_monitor)
    
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
        fast_dev_run=args.fast_dev_run,
        # resume_from_checkpoint=''
    )

    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    cli_main()
