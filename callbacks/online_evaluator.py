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
        pl_module.decoder = PointNetDecoder(num_classes=self.num_classes).to(pl_module.device)
        self.optimizer = torch.optim.Adam(pl_module.decoder.parameters(),lr=1e-4)

    def get_representations(self, pl_module: LightningModule, x: Tensor, class_id: Tensor) -> Tensor:
        representations = pl_module(x, class_id)
        return representations

    def to_device(self, batch, device):
        inputs, y, class_id = batch

        # last input is for online eval
        x_online = inputs[-1]
        y_online = y[-1]
        x_online = x_online.to(device)
        y_online = y_online.to(device)
        class_id = class_id.to(device)

        return x_online, y_online, class_id

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        x, y, class_id = self.to_device(batch,pl_module.device)
        with torch.no_grad():
            representations = self.get_representations(pl_module, x, class_id) # No train for encoder.

        representations = representations.detach()
        decoder_logits = pl_module.decoder(representations) # TODO: Check if we need class id here.
        # TODO: We need transpose for decoder_logits
        decoder_loss = F.cross_entropy(decoder_logits,y)
        
        # Finetune decoder 
        decoder_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Log Metrics
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
        x, y, class_id = self.to_device(batch, pl_module.device)

        with torch.no_grad():
            representations = self.get_representations(pl_module, x, class_id)

        representations = representations.detach()

        # TODO: Check if we need class id here.
        # TODO: We need transpose for decoder_logits
        decoder_logits = pl_module.decoder(representations)
        decoder_loss = F.cross_entropy(decoder_logits, y)

        # Log metrics
        # TODO: Can add accuracy IoU later. For now it is only loss.
        pl_module.log('online_val_loss', decoder_loss, on_step=False, on_epoch=True, sync_dist=True)