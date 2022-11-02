import os.path

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
    config = args.config
    # config = "deberta-xlarge.yaml"

    cfg = FPELL.data.io.load_yaml_config(config)
    # cfg.merge_with_dotlist(unknown_args)

    model_path = pathlib.Path(cfg.train.model_path) / f"{cfg.model.name}_folds{cfg.dataset.cv.n_folds}_{cfg.train.name_suffix}".replace("/", "-")
    model_path.mkdir(exist_ok=False, parents=True)
    shutil.copy(config, model_path / os.path.basename(config))

    cfg.train.model_path = model_path
    print(cfg)
    FPELL.nn.training.train(cfg)
