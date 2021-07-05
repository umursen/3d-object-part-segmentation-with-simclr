from typing import Sequence
import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from torch import Tensor
import numpy as np

from util.training import to_categorical, test_val_shared_step, test_val_shared_epoch
from models.pointnet import get_supervised_loss


class SSLOnlineEvaluator(Callback):
    def __init__(
            self,
            num_classes: int = 16,
            num_seg_classes: int = 50,
            npoints: int = 2500,
            seg_class_map: dict = None,
    ):
        self.num_classes = num_classes
        self.num_seg_classes = num_seg_classes
        self.npoints = npoints
        self.seg_class_map = seg_class_map

        self.seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
        for cat in self.seg_class_map.keys():
            for label in self.seg_class_map[cat]:
                self.seg_label_to_cat[label] = cat

        self.loss_criterion = get_supervised_loss()

        self.optimizer = None

        # Results
        self.loss = []
        self.mean_correct = []

        self.eval_results = []

    def on_pretrain_routine_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        from models.pointnet import PointNetDecoder
        pl_module.decoder = PointNetDecoder(part_num=self.num_seg_classes).to(pl_module.device)
        self.optimizer = torch.optim.Adam(pl_module.decoder.parameters(), lr=1e-4)

    def get_representations(self, pl_module: LightningModule, x: Tensor) -> Tensor:
        representations = pl_module(x)
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
        x, y, class_id = self.to_device(batch, pl_module.device)
        with torch.no_grad():
            representations, concat, trans_feat = self.get_representations(pl_module, x)  # No train for encoder.

        # x = x.detach()
        prediction = pl_module.decoder(
            representations,
            x.size(),
            to_categorical(class_id, self.num_classes),
            concat
        )

        #
        prediction = prediction.contiguous().view(-1, self.num_seg_classes)
        target = y.view(-1, 1)[:, 0]
        pred_choice = prediction.data.max(1)[1]

        correct = pred_choice.eq(target.data).cpu().sum()
        mean_correct = correct.item() / (prediction.shape[0] * self.npoints)

        loss = self.loss_criterion(prediction, target, trans_feat)

        # Finetune decoder
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        pl_module.log('online_train_loss', loss, on_step=True, on_epoch=False)

        # Save outputs
        self.loss.append(loss)
        self.mean_correct.append(mean_correct)

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        train_instance_acc = np.mean(self.mean_correct)
        pl_module.log('online_train_acc', train_instance_acc, on_step=False, on_epoch=True)

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
            representations, concat, trans_feat = self.get_representations(pl_module, x)

        prediction = pl_module.decoder(
            representations,
            x.size(),
            to_categorical(class_id, self.num_classes),
            concat
        )

        self.eval_results.append(test_val_shared_step(
            x, y, prediction, self.seg_label_to_cat,
            self.seg_class_map, self.num_seg_classes
        ))

    def on_validation_epoch_end(self, trainer, pl_module):
        shape_ious, accuracy, class_avg_accuracy, class_avg_iou, instance_avg_iou = test_val_shared_epoch(
            self.eval_results, self.seg_class_map, self.num_seg_classes)
        pl_module.log('online_val_accuracy', accuracy, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log('online_val_class_avg_accuracy', class_avg_accuracy, on_step=False, on_epoch=True, sync_dist=True) # NAN

        for cat in sorted(shape_ious.keys()):
            pl_module.log(f'eval mIoU of {cat}', shape_ious[cat], on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log('online_val_class_avg_iou', class_avg_iou, on_step=False, on_epoch=True, sync_dist=True) # NAN
        pl_module.log('online_val_instance_avg_iou', instance_avg_iou, on_step=False, on_epoch=True, sync_dist=True)