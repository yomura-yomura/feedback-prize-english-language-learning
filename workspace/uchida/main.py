from workspace.uchida.model import SummaryModel
from pathlib import Path
import argparse


def main(config_path: str, save_path: str, is_save: bool) -> None:

    summarizer = SummaryModel(cfg_path=config_path)
    df_summary = summarizer.summarize()

    if is_save:
        df_summary.to_csv(Path(save_path) / "Summarized_train_data.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-config_path", "--config_path", type=str, default="config.yaml"
    )
    parser.add_argument("-is_save", "--is_save", type=bool, default=True)
    parser.add_argument("-save_path", "--save_path", type=str, default="result")
    args = parser.parse_args()

    config_path = args.config_path
    save_path = args.save_path
    is_save = args.is_save

    main(config_path=config_path, save_path=save_path, is_save=is_save)
