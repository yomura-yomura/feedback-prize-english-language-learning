import FPELL.nn
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from collections.abc import Iterable
import FPELL.data
import pathlib
import torch


def train(cfg):
    wandb.login()

    if isinstance(cfg.dataset.cv.fold, Iterable):
        for fold in list(cfg.dataset.cv.fold):
            cfg.dataset.cv.fold = fold
            train(cfg.copy())
        return
    elif isinstance(cfg.dataset.cv.fold, int) or cfg.dataset.cv.fold is None:
        pass
    else:
        raise TypeError(type(cfg.dataset.cv.fold))

    if cfg.dataset.cv.fold is None or cfg.dataset.cv.fold < 0:
        cfg.dataset.cv.fold = list(range(cfg.dataset.cv.n_folds))
        train(cfg)
        return

    pl.seed_everything(cfg.seed_everything)

    print(f"* fold {cfg.dataset.cv.fold}")
    df = FPELL.data.io_with_cfg.get_df(cfg)

    train_df = df[df["fold"] != cfg.dataset.cv.fold].reset_index(drop=True)
    valid_df = df[df["fold"] == cfg.dataset.cv.fold].reset_index(drop=True)

    module = FPELL.nn.module.FPELLModule(cfg)

    if cfg.train.checkpoint_to_start_from is not None:
        print(f"[Info] training starts from {cfg.train.checkpoint_to_start_from}")
        state_dict = module.model.state_dict()
        loaded_state_dict = torch.load(cfg.train.checkpoint_to_start_from)["state_dict"]
        state_dict.update({
            k.replace("model.", ""): v
            for k, v in loaded_state_dict.items()
            if k.startswith("model.")
        })
        module.model.load_state_dict(state_dict)

    data_module = FPELL.nn.datamodule.FPELLDataModule(cfg, train_df, valid_df)

    name = "_".join(
        [
            f"{cfg.train.name_prefix or ''}{cfg.model.name}",
            f"fold{cfg.dataset.cv.fold}-of-{cfg.dataset.cv.n_folds}",
            cfg.train.name_suffix
        ]
    ).replace("/", "-")

    callbacks = [
        ModelCheckpoint(
            dirpath=pathlib.Path(cfg.train.model_path) / "checkpoints",
            filename=name,
            verbose=True,
            monitor='valid/loss', mode='min',
            save_weights_only=True
        ),
        LearningRateMonitor(logging_interval="step")
    ]

    # Trainerに設定
    trainer = pl.Trainer(
        logger=WandbLogger(
            project="FeedBack-Prize-English-Language-Learning", name=name
        ),
        accelerator="gpu",
        devices=1,
        precision=cfg.model.precision,

        max_epochs=cfg.train.epochs,
        callbacks=callbacks,

        limit_val_batches=0.0 if cfg.train.evaluate_after_steps > 0 else 1.0,
        val_check_interval=cfg.train.val_check_interval,
        log_every_n_steps=1,
        accumulate_grad_batches=cfg.train.accumulate_grad_batches

        # benchmark=True
    )
    trainer.fit(module, data_module)

    wandb.finish()
