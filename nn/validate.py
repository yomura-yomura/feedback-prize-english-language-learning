import pathlib
import re
import FPELL.nn.from_checkpoint
import pandas as pd
import numpy as np
import sys


def rmse_loss(pred, true):
    return np.mean(np.sqrt(np.mean((pred - true) ** 2, axis=1)), axis=0)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        target_dir = "models/microsoft-deberta-xlarge_folds4_v0.2"
    else:
        target_dir = sys.argv[1]
    target_dir = pathlib.Path(target_dir)

    cfg, ckpt_paths = FPELL.nn.from_checkpoint.load_cfg_and_checkpoint_paths(target_dir)
    df = FPELL.data.io.get_df(cfg.dataset, cfg.seed)

    cv_dict = {}
    for csv_path in sorted((target_dir / "predicted_csv").glob("*.csv")):
        fold = int(csv_path.name[4:-4])
        predicted = pd.read_csv(csv_path)[df["fold"] == fold]
        true = df.loc[df["fold"] == fold, cfg.dataset.target_columns]
        cv_dict[fold] = rmse_loss(predicted, true)

    str_cv_list = " ".join(map("{:.2f}".format, cv_dict.values()))
    print(f"CV: {np.mean(list(cv_dict.values())):.2f} ({str_cv_list})")
