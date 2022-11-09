import os.path

import omegaconf

import FPELL.nn
import FPELL.data
import argparse
import pathlib
import shutil
import sys


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args, unknown_args = parser.parse_known_args()

    cfg = FPELL.data.io.load_yaml_config(args.config)

    cfg.train.model_path = "./models_trained_for_each"

    cfg.dataset.cv.fold = 0
    print(cfg)

    for target_column in cfg.dataset.target_columns:
        copied_cfg = cfg.copy()
        copied_cfg.dataset.target_columns = [target_column]

        model_path = (
            pathlib.Path(cfg.train.model_path) /
            f"{cfg.model.name}_folds{cfg.dataset.cv.n_folds}_{cfg.train.name_suffix}".replace("/", "-") /
            target_column
        )
        if model_path.exists():
            print(f"skipped {model_path}")
            continue
        model_path.mkdir(exist_ok=False, parents=True)
        with open(model_path / os.path.basename(args.config), "w") as f:
            omegaconf.OmegaConf.save(copied_cfg, f)

        cfg.train.model_path = model_path

        FPELL.nn.training.train(copied_cfg)
