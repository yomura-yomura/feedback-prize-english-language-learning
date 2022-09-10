import pathlib
import torch
import FPELL.nn
import omegaconf
from pytorch_lightning import Trainer
import gc


__all__ = ["predict", "validate", "load_cfg_and_checkpoint_paths"]


def predict(ckpt_path, cfg, df):
    print(f"* load from checkpoint {ckpt_path}")
    module = FPELL.nn.module.FPELLModule.load_from_checkpoint(
        str(ckpt_path), cfg=cfg, map_location=torch.device("cuda")
    )
    tl = Trainer(
        accelerator="gpu", devices=1,
        max_epochs=1000,
        precision=cfg.precision
    )
    datamodule = FPELL.nn.datamodule.FPEDataModule(cfg, test_df=df)
    logits = tl.predict(module, datamodule)
    with torch.no_grad():
        predicted = torch.concat(logits).float().softmax(dim=1).numpy()
    del tl, module, datamodule, logits
    gc.collect()
    torch.cuda.empty_cache()
    return predicted


def validate(ckpt_path, cfg, df):
    print(f"* load from checkpoint {ckpt_path}")
    module = FPELL.nn.module.FPELLModule.load_from_checkpoint(
        str(ckpt_path), cfg=cfg, map_location=torch.device("cuda")
    )
    tl = Trainer(
        accelerator="gpu", devices=1,
        max_epochs=1000,
        precision=cfg.precision
    )
    datamodule = FPELL.nn.datamodule.FPEDataModule(cfg, valid_df=df)
    validated = tl.validate(module, datamodule)
    del tl, module, datamodule
    gc.collect()
    torch.cuda.empty_cache()
    return validated


def load_cfg_and_checkpoint_paths(model_weight_root_path, is_test, data_root_path=None):
    model_weight_root_path = pathlib.Path(model_weight_root_path)

    _yaml_files = list(model_weight_root_path.glob("*.yaml"))
    assert len(_yaml_files) == 1
    yaml_file = _yaml_files[0]

    cfg = omegaconf.OmegaConf.load(yaml_file)
    cfg.gradient_checkpointing = False
    cfg.test_batch_size = 2

    if is_test:
        cfg.dataset_type = "test"
    if data_root_path is not None:
        cfg.data_root_path = data_root_path

    ckpt_paths = sorted(model_weight_root_path.glob("*.ckpt"))
    return cfg, ckpt_paths
