from argparse import ArgumentParser

from pl_bolts.models.self_supervised.simclr.simclr_module import SimCLR

from typing import List, Optional

import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torchmetrics import Accuracy


class SSLFineTuner(pl.LightningModule):
    """
    Finetunes a self-supervised learning backbone.
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        in_features: int = 2048,
        num_classes: int = 1000,
        epochs: int = 100,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.,
        learning_rate: float = 0.1,
        weight_decay: float = 1e-6,
        nesterov: bool = False,
        scheduler_type: str = 'cosine',
        decay_epochs: List = [60, 80],
        gamma: float = 0.1,
        final_lr: float = 0.
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

        self.backbone = backbone

        # Define fine-tuning model
        self.linear_layer = ... # SSLEvaluator(n_input=in_features, n_classes=num_classes, p=dropout, n_hidden=hidden_dim)

        # metrics
        self.train_acc = Accuracy()
        self.val_acc = Accuracy(compute_on_step=False)
        self.test_acc = Accuracy(compute_on_step=False)

    def on_train_epoch_start(self) -> None:
        self.backbone.eval()

    def training_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch)
        acc = self.train_acc(logits.softmax(-1), y)

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc_step', acc, prog_bar=True)
        self.log('train_acc_epoch', self.train_acc)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch)
        self.val_acc(logits.softmax(-1), y)

        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_acc', self.val_acc)

        return loss

    def test_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch)
        self.test_acc(logits.softmax(-1), y)

        self.log('test_loss', loss, sync_dist=True)
        self.log('test_acc', self.test_acc)

        return loss

    def shared_step(self, batch):
        x, y = batch

        with torch.no_grad():
            feats = self.backbone(x)

        feats = feats.view(feats.size(0), -1)
        logits = self.linear_layer(feats)
        loss = F.cross_entropy(logits, y)

        return loss, logits, y

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.linear_layer.parameters(),
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

    parser.add_argument("--batch_size", default=64, type=int, help="batch size per gpu")
    parser.add_argument("--num_workers", default=8, type=int, help="num of workers per GPU")
    parser.add_argument("--gpus", default=4, type=int, help="number of GPUs")
    parser.add_argument('--num_epochs', default=100, type=int, help="number of epochs")

    # fine-tuner params
    parser.add_argument('--in_features', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--learning_rate', type=float, default=0.3)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--scheduler_type', type=str, default='cosine')
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--final_lr', type=float, default=0.)

    args = parser.parse_args()

    if args.dataset == 'all':
        # TODO: Set data loader
        dm = ...
    elif args.dataset == 'shapenet':
        # TODO: Set data loader
        dm = ...
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
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        dataset=args.dataset,
    ).load_from_checkpoint(args.ckpt_path, strict=False)

    tuner = SSLFineTuner(
        backbone,
        in_features=args.in_features,
        num_classes=dm.num_classes,
        epochs=args.num_epochs,
        hidden_dim=None,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov,
        scheduler_type=args.scheduler_type,
        gamma=args.gamma,
        final_lr=args.final_lr
    )

    trainer = pl.Trainer(
        gpus=args.gpus,
        num_nodes=1,
        precision=16,
        max_epochs=args.num_epochs,
        distributed_backend='ddp',
        sync_batchnorm=True if args.gpus > 1 else False,
    )

    trainer.fit(tuner, dm)
    trainer.test(datamodule=dm)


if __name__ == '__main__':
    cli_main()
