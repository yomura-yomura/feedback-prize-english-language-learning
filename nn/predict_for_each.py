import FPELL.data
from predict import main
from train_for_each import get_model_path
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    model_path = get_model_path(cfg=FPELL.data.io.load_yaml_config(args.config))

    for target_dir in model_path.glob("*"):
        main(target_dir)
