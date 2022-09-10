import torch
import torch.nn as nn

from transformers import AutoModel, AutoConfig, AdamW
from transformers import get_scheduler
import pytorch_lightning as pl
from typing import Any


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


class FPELLModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.model = AutoModel.from_pretrained(self.cfg.model_name)
        if self.cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.config = AutoConfig.from_pretrained(self.cfg.model_name)
        self.criterion = nn.CrossEntropyLoss()
        # self.drop = nn.Dropout(p=0.2)
        # self.pooler = MeanPooling()
        self.Linear = nn.Linear(self.config.hidden_size, self.cfg.num_classes)

        if self.cfg.use_multi_dropout:
            # ノートブックを参考に追加
            self.dropout1 = nn.Dropout(0.1)
            self.dropout2 = nn.Dropout(0.2)
            self.dropout3 = nn.Dropout(0.3)
            self.dropout4 = nn.Dropout(0.4)
            self.dropout5 = nn.Dropout(0.5)

        self.train_loss_meter = AverageMeter()

    # def _init_weights(self, module):
    #     if isinstance(module, nn.Linear):
    #         module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
    #         if module.bias is not None:
    #             module.bias.data.zero_()
    #     elif isinstance(module, nn.Embedding):
    #         module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
    #         if module.padding_idx is not None:
    #             module.weight.data[module.padding_idx].zero_()
    #     elif isinstance(module, nn.LayerNorm):
    #         module.bias.data.zero_()
    #         module.weight.data.fill_(1.0)

    def forward(self, batch):
        out = self.model(**batch, output_hidden_states=False)

        preds = out.last_hidden_state[:, 0, :]

        if self.cfg.use_multi_dropout:
            # ノートブックを参考に追加
            logits1 = self.Linear(self.dropout1(preds))
            logits2 = self.Linear(self.dropout2(preds))
            logits3 = self.Linear(self.dropout3(preds))
            logits4 = self.Linear(self.dropout4(preds))
            logits5 = self.Linear(self.dropout5(preds))
            preds = (logits1 + logits2 + logits3 + logits4 + logits5) / 5

        # out = self.pooler(out.last_hidden_state, mask)
        # out = self.drop(out)
        # outputs = self.fc(out)
        return preds

    # trainのミニバッチに対して行う処理
    def training_step(self, batch, batch_idx):
        labels = batch.pop("labels")
        preds = self.forward(batch)
        loss = self.criterion(preds, labels)

        self.train_loss_meter.update(loss.item(), len(labels))
        self.log("train/loss", self.train_loss_meter.avg)

        return {
            'loss': loss,
            'batch_preds': preds,
            'batch_labels': labels
        }

    def training_step_end(self, outputs):
        if (
            self.global_step > self.cfg.evaluate_after_steps
            and self.trainer.limit_val_batches == 0.0
        ):
            self.trainer.limit_val_batches = 1.0
            self.trainer.reset_val_dataloader()

    # validation、testでもtrain_stepと同じ処理を行う
    def validation_step(self, batch, batch_idx):
        labels = batch.pop("labels")
        preds = self.forward(batch)
        loss = self.criterion(preds, labels)

        # self.log("valid_loss", loss, logger=True)
        return {
            'loss': loss,
            'batch_preds': preds,
            'batch_labels': labels
        }

    # def training_epoch_end(self, train_step_outputs, mode="val"):
    #     # loss計算
    #     epoch_preds = torch.cat([x['batch_preds'] for x in train_step_outputs])
    #     epoch_labels = torch.cat([x['batch_labels'] for x in train_step_outputs])
    #     epoch_loss = self.criterion(epoch_preds, epoch_labels)
    #     self.log(f"train_epoch_loss", epoch_loss, logger=True)

    # epoch終了時のlossの計算
    def validation_epoch_end(self, val_step_outputs, mode="val"):
        # loss計算
        epoch_preds = torch.cat([x['batch_preds'] for x in val_step_outputs])
        epoch_labels = torch.cat([x['batch_labels'] for x in val_step_outputs])
        epoch_loss = self.criterion(epoch_preds, epoch_labels)
        self.log(f"valid/loss", epoch_loss)

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay
        )
        if self.cfg.scheduler is not None:
            scheduler = get_scheduler(
                self.cfg.scheduler, optimizer,
                num_warmup_steps=self.cfg.num_warmup_steps, num_training_steps=self.cfg.num_training_steps
            )
            return [optimizer], [scheduler]
        else:
            return [optimizer]
