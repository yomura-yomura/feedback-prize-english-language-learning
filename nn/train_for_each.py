import os.path
import omegaconf
import FPELL.nn
import FPELL.data
import argparse
import pathlib
import shutil
import sys


# model_root_path = pathlib.Path("./models_trained_for_each")
model_root_path = pathlib.Path("./models_trained_for_each2")


def get_model_path(cfg):
    return (
        model_root_path /
        f"{cfg.model.name}_folds{cfg.dataset.cv.n_folds}_{cfg.train.name_suffix}".replace("/", "-")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = FPELL.data.io.load_yaml_config(args.config)

    cfg.dataset.cv.fold = 0

    print(cfg)

    for target_column in cfg.dataset.target_columns:
        copied_cfg = cfg.copy()
        copied_cfg.dataset.target_columns = [target_column]

        model_path = get_model_path(copied_cfg) / target_column
        if model_path.exists():
            print(f"skipped {model_path}")
            continue
        copied_cfg.train.model_path = model_path

        model_path.mkdir(exist_ok=False, parents=True)
        with open(model_path / os.path.basename(args.config), "w") as f:
            omegaconf.OmegaConf.save(copied_cfg, f)

        FPELL.nn.training.train(copied_cfg)
