import pytorch_lightning.utilities.seed
import FPELL.nn
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from collections.abc import Iterable
import FPELL.data


def train(cfg):
    wandb.login()

    if isinstance(cfg.fold, Iterable):
        for fold in list(cfg.fold):
            cfg.fold = fold
            train(cfg)
        return
    elif isinstance(cfg.fold, int):
        pass
    else:
        raise TypeError(type(cfg.fold))

    if cfg.fold < 0:
        for fold in range(cfg.n_fold):
            cfg.fold = fold
            train(cfg)
        return

    print(f"* fold {cfg.fold}")

    # 必要ないかもしれないが
    FPELL.nn.set_seed(cfg.seed)
    if cfg.seed != 3655:
        pytorch_lightning.utilities.seed.seed_everything(cfg.seed)

    df = FPELL.data.io.get_df(
        cfg,
        fix_sentences=True, drop_duplicates=True
    )

    train_df = df[df.kfold != cfg.fold].reset_index(drop=True)
    valid_df = df[df.kfold == cfg.fold].reset_index(drop=True)

    model = FPELL.nn.module.FPELLModule(cfg)
    data_module = FPELL.nn.datamodule.FPEDataModule(cfg, train_df, valid_df)

    name = f'{cfg.model_name}_fold{cfg.fold}_{cfg.text}'.replace('/', '-')

    # Trainerに設定
    trainer = pl.Trainer(
        logger=WandbLogger(project=cfg.competition, name=name),
        accelerator="gpu",
        devices=1,
        precision=cfg.precision,

        max_epochs=cfg.epochs,
        callbacks=[
            ModelCheckpoint(
                dirpath=cfg.model_save,
                filename=f'{cfg.model_name}_fold{cfg.fold}_{cfg.text}'.replace('/', '-'),
                verbose=True, monitor='valid/loss', mode='min'
            ),
            LearningRateMonitor(logging_interval="step")
        ],

        limit_val_batches=0.0 if cfg.evaluate_after_steps > 0 else 1.0,
        val_check_interval=cfg.val_check_interval,
        log_every_n_steps=1

        # benchmark=True
    )
    trainer.fit(model, data_module)

    wandb.finish()
