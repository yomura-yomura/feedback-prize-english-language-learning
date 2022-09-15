import pathlib
import sys
import pandas as pd
import os
import FPELL.nn.from_checkpoint
import tqdm
import numpy as np
import collections
from validate import rmse_loss


if __name__ == "__main__":
    if len(sys.argv) == 1:
        target_dirs = [
            "models/microsoft-deberta-xlarge_folds4_v0.2",
            "models/microsoft-deberta-v2-xlarge_folds4_v0.3"
        ]
    else:
        target_dirs = sys.argv[1:3]

    target_dirs = list(map(pathlib.Path, target_dirs))

    df = None
    csv_paths_dict = collections.defaultdict(list)
    for target_dir in tqdm.tqdm(target_dirs, desc="loading df"):
        cfg, ckpt_paths = FPELL.nn.from_checkpoint.load_cfg_and_checkpoint_paths(target_dir)
        df_ = FPELL.data.io.get_df(cfg.dataset, cfg.seed)
        if df is None:
            df = df_
        else:
            assert np.all(df == df_)

        for csv_path in sorted((target_dir / "predicted_csv").glob("*.csv")):
            fold = int(csv_path.name[4:-4])
            csv_paths_dict[fold].append(csv_path)

    def calc_score(weights_list):
        if isinstance(weights_list, tuple):
            weights_list = [weights_list] * len(csv_paths_dict)

        train_scores = []
        valid_scores = []
        for (fold, ckpt_paths), weights in zip(csv_paths_dict.items(), weights_list):
            predicted_df_list = [pd.read_csv(ckpt_path) for ckpt_path in ckpt_paths]
            averaged_predicted = np.average(predicted_df_list, weights=weights, axis=0)
            is_valid = df["fold"] == fold
            train_scores.append(rmse_loss(averaged_predicted[~is_valid], df[~is_valid][cfg.dataset.target_columns]))
            valid_scores.append(rmse_loss(averaged_predicted[is_valid], df[is_valid][cfg.dataset.target_columns]))
        return train_scores, valid_scores


    import plotly.express as px
    import plotly_utility
    fig = px.line(
        pd.DataFrame([
            (
                w,
                np.mean(
                    calc_score(
                        (w, 1 - w)
                    )[0]
                )
            )
            for w in np.linspace(0, 1, 10)
        ]),
        x=0, y=1,
        labels={"0": "weight", "1": "train core"}
    )
    plotly_utility.offline.mpl_plot(fig)

    ret = pd.DataFrame([
        (
            w1, w2, w3, w4,
            np.mean(
                calc_score([
                    (w1, 1 - w1),
                    (w2, 1 - w2),
                    (w3, 1 - w3),
                    (w4, 1 - w4)
                ])[0]
            )
        )
        for w1 in tqdm.tqdm(np.linspace(0, 1, 10))
        for w2 in np.linspace(0, 1, 10)
        for w3 in np.linspace(0, 1, 10)
        for w4 in np.linspace(0, 1, 10)
    ])

    print(target_dirs)
    train_scores, valid_scores = calc_score((0.5, 0.5))
    print(f"train: {np.mean(train_scores):.2f} ({' '.join(map('{:.2f}'.format, train_scores))})")
    print(f"valid: {np.mean(valid_scores):.2f} ({' '.join(map('{:.2f}'.format, valid_scores))})")

    train_scores, valid_scores = calc_score((0.8, 0.2))
    print(f"train: {np.mean(train_scores):.2f} ({' '.join(map('{:.2f}'.format, train_scores))})")
    print(f"valid: {np.mean(valid_scores):.2f} ({' '.join(map('{:.2f}'.format, valid_scores))})")
