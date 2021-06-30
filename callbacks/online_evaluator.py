from typing import Sequence, Tuple, Union
import torch
from pytorch_lightning import Callback,LightningModule,Trainer
from torch import device, Tensor

class SSLOnlineEvaluator(Callback):
    def __init__(
        self,
        dataset: str,
        z_dim: int = None,
        num_classes: int = None):
        
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.dataset = dataset

    def on_pretrain_routine_start(self,trainer: Trainer, pl_module: LightningModule) -> None:
        from models.pointnet import PointNetDecoder
        pl_module.non_linear_evaluator = PointNetDecoder(
            z_dim = self.z_dim,
            num_classes=self.num_classes
        ).to(pl_module.device)

        self.optimizer = torch.optim.Adam(pl_module.non_linear_evaluator.parameters(),lr=1e-4)

    def get_representations(self, pl_module: LightningModule, x: Tensor) -> Tensor:
        representations = pl_module(x)
        return representations

    def to_device(self):
        pass
        #TODO
    
    def on_train_batch_end(self):
        pass
        #TODO

    def on_validation_batch_end(self):
        pass
        #TODO    