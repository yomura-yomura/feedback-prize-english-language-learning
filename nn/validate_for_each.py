import FPELL.data
from validate import main
from train_for_each import get_model_path
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    model_path = get_model_path(cfg=FPELL.data.io.load_yaml_config(args.config))

    for cfg_path in model_path.glob("*/*.yaml"):
        cfg = FPELL.data.io.load_yaml_config(cfg_path)
        print(f"* {cfg.train.model_path.relative_to(model_path)}")
        main(cfg.train.model_path)