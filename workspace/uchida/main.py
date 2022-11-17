import argparse
from pathlib import Path

import numpy as np

from workspace.uchida.model.summarized_model import SummaryModel

rand = np.random.randint(1, 100000)


def main(config_path: str, save_path: str, file_name: str, is_save: bool) -> None:

    summarizer = SummaryModel(cfg_path=config_path)
    df_summary = summarizer.summarize()

    if is_save:
        df_summary.to_csv(
            Path(save_path) / f"Summarized_train_data_{file_name}_{rand}.csv",
            index=False,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-config_path",
        "--config_path",
        type=str,
        default="config/summarized_config.yaml",
    )
    parser.add_argument("-is_save", "--is_save", type=bool, default=True)
    parser.add_argument("-name", "--name", type=str, default="test")
    parser.add_argument("-save_path", "--save_path", type=str, default="result")
    args = parser.parse_args()

    config_path = args.config_path
    save_path = args.save_path
    is_save = args.is_save
    _name = args.name

    main(config_path=config_path, save_path=save_path, file_name=_name, is_save=is_save)
