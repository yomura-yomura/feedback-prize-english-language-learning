import attrdict
import omegaconf
import torch
import torch.nn as nn
from torch.optim import Adam, SGD, AdamW
from transformers import AutoModel, AutoConfig
from transformers import get_scheduler
import pytorch_lightning as pl
from typing import Any
import numpy as np
from .awp import AWP
import bitsandbytes as bnb
from typing import Optional


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class RMSELoss(nn.Module):
    def __init__(self, reduction='mean', eps=1e-9):
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)
        self.reduction = reduction
        self.eps = eps

    def forward(self, y_pred, y_true):
        return torch.sqrt(self.mse(y_pred, y_true) + self.eps)


class FPELLModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.model = AutoModel.from_pretrained(self.cfg.model.name)
        if self.cfg.model.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.config = AutoConfig.from_pretrained(self.cfg.model.name)
        # self.criterion = nn.SmoothL1Loss(reduction="mean")
        self.criterion = RMSELoss(reduction="mean")

        self.train_loss_meter = AverageMeter()

        self.Linear = nn.Linear(self.config.hidden_size, len(self.cfg.dataset.target_columns))
        if self.cfg.model.use_multi_dropout:
            self.dropout1 = nn.Dropout(0.1)
            self.dropout2 = nn.Dropout(0.2)
            self.dropout3 = nn.Dropout(0.3)
            self.dropout4 = nn.Dropout(0.4)
            self.dropout5 = nn.Dropout(0.5)

        self.awp = None
        self.automatic_optimization = False

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit":
            self._init_weights(self.Linear)

            # print("self.trainer.accumulate_grad_batches", self.trainer.accumulate_grad_batches)
            if isinstance(self.trainer.accumulate_grad_batches, int):
                pass
            # elif self.trainer.accumulate_grad_batches is None:
            #     self.trainer.accumulate_grad_batches = 1
            elif isinstance(self.trainer.accumulate_grad_batches, dict):
                raise NotImplementedError
            else:
                raise TypeError(type(self.trainer.accumulate_grad_batches))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.cfg.model.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.cfg.model.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, batch: dict):
        if "label" in batch.keys():
            batch = {k: v for k, v in batch.items() if k != "label"}
        if "labels" in batch.keys():
            batch = {k: v for k, v in batch.items() if k != "labels"}

        out = self.model(**batch, output_hidden_states=False)

        preds = out.last_hidden_state[:, 0, :]

        if self.cfg.model.use_multi_dropout:
            # ノートブックを参考に追加
            logits1 = self.Linear(self.dropout1(preds))
            logits2 = self.Linear(self.dropout2(preds))
            logits3 = self.Linear(self.dropout3(preds))
            logits4 = self.Linear(self.dropout4(preds))
            logits5 = self.Linear(self.dropout5(preds))
            preds = (logits1 + logits2 + logits3 + logits4 + logits5) / 5
        else:
            preds = self.Linear(preds)

        return preds

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()

        preds = self.forward(batch)
        loss = self.criterion(preds, batch["labels"])
        self.manual_backward(loss)

        if (
            self.awp is not None
            and
            (
                self.global_step >= (
                    self.trainer.estimated_stepping_batches / self.trainer.max_steps * self.cfg.train.awp.start_epoch
                )
            )
        ):
            self.awp.attack_backward()
            preds = self.forward(batch)
            awp_loss = self.criterion(preds, batch["labels"])
            self.manual_backward(awp_loss)
            self.awp._restore()
            loss += awp_loss

        if (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0:
            opt.step()
            opt.zero_grad()

        sch = self.lr_schedulers()
        sch.step()

        self.train_loss_meter.update(loss.item(), len(batch["labels"]))
        self.log("train/loss", self.train_loss_meter.avg, on_step=True, on_epoch=True, prog_bar=True)

        # return {
        #     'loss': loss,
        #     'batch_preds': preds,
        #     'batch_labels': batch["labels"]
        # }

    def training_step_end(self, *args, **kwargs):
        if (
            self.global_step > self.cfg.train.evaluate_after_steps
            and self.trainer.limit_val_batches == 0.0
        ):
            self.trainer.limit_val_batches = 1.0
            self.trainer.reset_val_dataloader()

    def validation_step(self, batch, batch_idx):
        preds = self.forward(batch)
        loss = self.criterion(preds, batch["labels"])
        self.log(f"valid/loss", loss, prog_bar=True)
        return {
            'loss': loss,
            # 'batch_preds': preds,
            # 'batch_labels': labels
        }

    # def training_epoch_end(self, train_step_outputs, mode="val"):
    #     # loss計算
    #     epoch_preds = torch.cat([x['batch_preds'] for x in train_step_outputs])
    #     epoch_labels = torch.cat([x['batch_labels'] for x in train_step_outputs])
    #     epoch_loss = self.criterion(epoch_preds, epoch_labels)
    #     self.log(f"train_epoch_loss", epoch_loss, logger=True)

    # def validation_epoch_end(self, val_step_outputs, mode="val"):
    #     epoch_loss = torch.mean(torch.tensor([x['loss'] for x in val_step_outputs]))
    #     self.log(f"valid/loss", epoch_loss)

    def configure_optimizers(self):
        if self.cfg.model.optimizer.bits is None:
            optimizer = AdamW(
                self.parameters(),
                lr=self.cfg.train.learning_rate,
                weight_decay=self.cfg.train.weight_decay
            )
        else:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.cfg.train.weight_decay,
                },
                {
                    "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            if self.cfg.model.optimizer.bits == 32:
                optimizer = bnb.optim.AdamW32bit(
                    optimizer_grouped_parameters,
                    lr=self.cfg.train.learning_rate,
                    weight_decay=self.cfg.train.weight_decay
                )
            elif self.cfg.model.optimizer.bits == 8:
                optimizer = bnb.optim.AdamW8bit(
                    optimizer_grouped_parameters,
                    lr=self.cfg.train.learning_rate,
                    weight_decay=self.cfg.train.weight_decay
                )

                # Thank you @gregorlied https://www.kaggle.com/nbroad/8-bit-adam-optimization/comments#1661976
                for module in self.modules():
                    if isinstance(module, torch.nn.Embedding):
                        bnb.optim.GlobalOptimManager.get_instance().register_module_override(
                            module, 'weight', {'optim_bits': 32}
                        )
            else:
                raise ValueError(self.cfg.model.optimizer.bits)

        if self.cfg.train.awp is not None:
            self.awp = AWP(
                model=self,
                optimizer=optimizer,
                adv_lr=self.cfg.train.awp.adv_lr,
                adv_eps=self.cfg.train.awp.adv_eps,
                adv_param="weight",
                use_amp=True
            )

        if self.cfg.model.optimizer.scheduler.name is None:
            return [optimizer]
        else:
            if self.cfg.model.optimizer.scheduler.cycle_interval_for_full_epochs is not None:
                assert not hasattr(self.cfg.model.optimizer.scheduler.kwargs, "num_training_steps")
                with omegaconf.open_dict(self.cfg):
                    self.cfg.model.optimizer.scheduler.kwargs.num_training_steps = int(
                        self.cfg.model.optimizer.scheduler.cycle_interval_for_full_epochs
                        *
                        self.trainer.estimated_stepping_batches
                    )

            scheduler = get_scheduler(
                self.cfg.model.optimizer.scheduler.name,
                optimizer,
                **self.cfg.model.optimizer.scheduler.kwargs
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
