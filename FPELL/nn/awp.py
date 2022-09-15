"""
Copied from https://github.com/Shujun-He/TeamSKT-Feedback-Prize---Predicting-Effective-Arguments-2nd-Place-solution/blob/main/Tom_solution/22_BaselineMLM/code/awp.py
Original: https://www.kaggle.com/code/skraiii/pppm-tokenclassificationmodel-train-8th-place
"""
import gc

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss


class AWP:
    def __init__(
            self,
            model: Module,
            # criterion: _Loss,
            optimizer: Optimizer,
            adv_lr: float = 1.0,
            adv_eps: float = 0.01,
            adv_param: str = "weight",
            use_amp: bool = True
    ) -> None:
        self.model = model
        # self.criterion = criterion
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.use_amp = use_amp
        self.backup = {}
        # self.backup_eps = {}
        self.grad_eps = {}

    # def attack_backward(self, inputs: dict) -> Tensor:
    def attack_backward(self):
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            self._save()
            self._attack_step()  # モデルを近傍の悪い方へ改変
            # y_preds = self.model(inputs)
            # _, _, adv_loss = self.model.training_step(inputs)
            # self.optimizer.zero_grad()
        # return adv_loss

    def _attack_step(self) -> None:
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    # 直前に損失関数に通してパラメータの勾配を取得できるようにしておく必要あり
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    # print(name, param.data.size())

                    # Equivalent to
                    # clip(
                    #   param.data,
                    #   self.backup[name] - self.grad_eps[name],
                    #   self.backup[name] + self.grad_eps[name]
                    # )
                    # TODO: it may cause round-off error.
                    self.backup[name] -= self.grad_eps[name]
                    param.data = torch.max(
                        param.data, self.backup[name]
                    )
                    self.backup[name] += 2 * self.grad_eps[name]
                    param.data = torch.min(
                        param.data, self.backup[name]
                    )
                    self.backup[name] -= self.grad_eps[name]

    def _save(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    self.grad_eps[name] = self.adv_eps * param.abs().detach()
                    # self.backup_eps[name] = (
                    #     self.backup[name] - grad_eps,
                    #     self.backup[name] + grad_eps,
                    # )

    def _restore(self) -> None:
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]

        self.backup = {}
        # self.backup_eps = {}
        self.grad_eps = {}
