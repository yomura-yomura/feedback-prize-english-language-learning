import FPELL.nn
from omegaconf import OmegaConf
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args, unknown_args = parser.parse_known_args()

    cfg = OmegaConf.load(args.config)
    cfg.merge_with_dotlist(unknown_args)

    print(f"config = {cfg}")

    FPELL.nn.training.train(cfg)
