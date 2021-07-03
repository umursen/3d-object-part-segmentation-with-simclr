from typing import Sequence, Tuple, Union
import torch
from pytorch_lightning import Callback,LightningModule,Trainer
from torch import device, Tensor
from torch.nn import functional as F

from datasets.shapenet_parts.shapenet_parts import ShapeNetParts

class SSLOnlineEvaluator(Callback):
    def __init__(
        self,
        z_dim: int = 1088,
        num_classes: int = None):
        
        self.z_dim = z_dim
        self.num_classes = num_classes

    def on_pretrain_routine_start(self,trainer: Trainer, pl_module: LightningModule) -> None:
        from models.pointnet import PointNetDecoder
        pl_module.non_linear_evaluator = PointNetDecoder(num_classes=self.num_classes).to(pl_module.device)
        self.optimizer = torch.optim.Adam(pl_module.non_linear_evaluator.parameters(),lr=1e-4)

    def get_representations(self, pl_module: LightningModule, x: Tensor) -> Tensor:
        representations = pl_module(x)
        return representations

    def to_device(self, batch, device):
        inputs, y = batch

        # last input is for online eval
        x = inputs[-1]
        x = x.to(device)
        y = y.to(device)

        return x, y

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        x,y = self.to_device(batch,pl_module.device)
        with torch.no_grad():
            representations = self.get_representations(pl_module,x) # No train for encoder.

        representations = representations.detach()
        decoder_logits = pl_module.non_linear_evaluator(representations)
        decoder_loss = F.cross_entropy(decoder_logits,y)
        
        # Finetune decoder 
        decoder_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Log Metrics
        # TODO: Can add accuracy IoU later. For now it is only loss.
        pl_module.log('online_train_loss', decoder_loss, on_step=True, on_epoch=False)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        x, y = self.to_device(batch, pl_module.device)

        with torch.no_grad():
            representations = self.get_representations(pl_module, x)

        representations = representations.detach()

        
        decoder_logits = pl_module.non_linear_evaluator(representations)
        decoder_loss = F.cross_entropy(decoder_logits, y)

        # Log metrics
        pl_module.log('online_val_loss', decoder_loss, on_step=False, on_epoch=True, sync_dist=True)
'''
    def on_train_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        x,y = self.to_device(pl_module.device) # x tilda
        with torch.no_grad():
            representations = self.get_representations(pl_module,x) # No train for encoder. This is h.

        representations = representations.detach()
        decoder_logits = pl_module.non_linear_evaluator(representations)
        decoder_loss = F.cross_entropy(decoder_logits,y)

        # Finetune decoder 
        decoder_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Log Metrics
        # TODO: Can add accuracy IoU later. For now it is only loss.
        pl_module.log('online_train_loss', decoder_loss, on_step=True, on_epoch=False)
'''