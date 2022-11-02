import pathlib
import pickle
import sys
import pandas as pd
import os
import FPELL.nn.from_checkpoint
import tqdm
import numpy as np
import collections
from validate import rmse_loss
import plotly.express as px
import plotly_utility


if __name__ == "__main__":
    # method = "weighting"
    method = "elastic"

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
        df_ = FPELL.data.io_with_cfg.get_df(cfg)
        if df is None:
            df = df_
        else:
            assert np.all(df == df_)

        for csv_path in sorted((target_dir / "predicted_csv").glob("*.csv")):
            fold = int(csv_path.name[4:-4])
            csv_paths_dict[fold].append(csv_path)

    if method == "weighting":
        def calc_score_weighting(weights_list, print_result=True):
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

            if print_result:
                print(f"train: {np.mean(train_scores):.2f} ({' '.join(map('{:.2f}'.format, train_scores))})")
                print(f"valid: {np.mean(valid_scores):.2f} ({' '.join(map('{:.2f}'.format, valid_scores))})")
            return train_scores, valid_scores

        fig = px.line(
            pd.DataFrame([
                (
                    w,
                    np.mean(
                        calc_score_weighting(
                            (w, 1 - w), print_result=False
                        )[0]
                    )
                )
                for w in np.linspace(0, 1, 10)
            ]),
            x=0, y=1,
            labels={"0": "weight", "1": "train core"}
        )
        plotly_utility.offline.mpl_plot(fig)

        # ret = pd.DataFrame([
        #     (
        #         w1, w2, w3, w4,
        #         np.mean(
        #             calc_score([
        #                 (w1, 1 - w1),
        #                 (w2, 1 - w2),
        #                 (w3, 1 - w3),
        #                 (w4, 1 - w4)
        #             ])[0]
        #         )
        #     )
        #     for w1 in tqdm.tqdm(np.linspace(0, 1, 10))
        #     for w2 in np.linspace(0, 1, 10)
        #     for w3 in np.linspace(0, 1, 10)
        #     for w4 in np.linspace(0, 1, 10)
        # ])
        print(target_dirs)
        train_scores, valid_scores = calc_score_weighting((0.5, 0.5))
        train_scores, valid_scores = calc_score_weighting((0.8, 0.2))
    elif method == "elastic":
        from sklearn.linear_model import ElasticNet, Ridge, Lasso

        def calc_score_elastic(alpha=0.01, l1_ratio=0.5, print_result=True):
            train_scores = []
            valid_scores = []
            reg_list = []
            for fold, csv_paths in csv_paths_dict.items():
                X = pd.concat([
                    pd.read_csv(csv_path)[cfg.dataset.target_columns] for csv_path in csv_paths
                ], axis=1).values
                if l1_ratio == 0:
                    reg = Lasso(alpha=alpha, max_iter=10_000, random_state=42)
                elif l1_ratio == 1:
                    reg = Ridge(alpha=alpha, max_iter=10_000, random_state=42)
                else:
                    reg = ElasticNet(
                        alpha=alpha, l1_ratio=l1_ratio,
                        max_iter=10_000,
                        random_state=42
                    )
                is_valid = df["fold"] == fold
                y = df[cfg.dataset.target_columns].values
                reg.fit(X[~is_valid], y[~is_valid])
                predicted = reg.predict(X)

                train_scores.append(rmse_loss(predicted[~is_valid], y[~is_valid]))
                valid_scores.append(rmse_loss(predicted[is_valid], y[is_valid]))
                reg_list.append(reg)
            if print_result:
                print(f"train: {np.mean(train_scores):.2f} ({' '.join(map('{:.2f}'.format, train_scores))})")
                print(f"valid: {np.mean(valid_scores):.2f} ({' '.join(map('{:.2f}'.format, valid_scores))})")
            return train_scores, valid_scores, reg_list

        fig = px.line(
            pd.DataFrame([
                (
                    w,
                    np.mean(
                        calc_score_elastic(
                            w, print_result=False
                        )[0]
                    )
                )
                for w in np.logspace(-8, 0, 10)
            ]),
            x=0, y=1, log_x=True, log_y=True,
            labels={"0": "weight", "1": "train core"}
        )
        fig.update_xaxes(exponentformat="power")
        plotly_utility.offline.mpl_plot(fig)

        fig = px.line(
            pd.DataFrame([
                (
                    w,
                    np.mean(
                        calc_score_elastic(
                            alpha=1e-3, l1_ratio=w, print_result=False
                        )[0]
                    )
                )
                for w in np.linspace(0, 1, 10)
            ]),
            x=0, y=1,
            labels={"0": "weight", "1": "train core"}
        )
        fig.update_xaxes(exponentformat="power")
        plotly_utility.offline.mpl_plot(fig)

        train_scores, valid_scores, reg_list = calc_score_elastic(alpha=1e-3, l1_ratio=0.1)
        coef = reg_list[0].coef_.copy()
        coef /= np.linalg.norm(coef, axis=1, keepdims=True)
        fig = px.imshow(coef, title=f"fold 0", color_continuous_scale="picnic", range_color=[-1, 1])
        plotly_utility.offline.mpl_plot(fig)
        stacking_model_root_path = pathlib.Path("models_for_stacking")
        stacking_model_root_path /= method
        stacking_model_root_path.mkdir(exist_ok=True, parents=True)
        for fold, reg in enumerate(reg_list):
            with open(stacking_model_root_path / f"fold{fold}.pkl", "wb") as f:
                pickle.dump(reg, f)
    elif method == "svr":
        from sklearn.svm import SVR

        def calc_score_svr(C=1, epsilon=0.1, print_result=True):
            train_scores = []
            valid_scores = []
            reg_list = []
            for fold, csv_paths in csv_paths_dict.items():
                X = pd.concat([
                    pd.read_csv(csv_path)[cfg.dataset.target_columns] for csv_path in csv_paths
                ], axis=1).values
                is_valid = df["fold"] == fold
                y = df[cfg.dataset.target_columns].values

                predicted = []
                reg = []
                for i in range(y.shape[1]):
                    reg.append(
                        SVR(
                            C=C, epsilon=epsilon
                        )
                    )
                    reg[i].fit(X[~is_valid], y[~is_valid][:, i])
                    predicted.append(reg[i].predict(X))
                predicted = np.stack(predicted, axis=1)

                train_scores.append(rmse_loss(predicted[~is_valid], y[~is_valid]))
                valid_scores.append(rmse_loss(predicted[is_valid], y[is_valid]))
                reg_list.append(reg)
            if print_result:
                print(f"train: {np.mean(train_scores):.2f} ({' '.join(map('{:.2f}'.format, train_scores))})")
                print(f"valid: {np.mean(valid_scores):.2f} ({' '.join(map('{:.2f}'.format, valid_scores))})")
            return train_scores, valid_scores, reg_list

        fig = px.line(
            pd.DataFrame([
                (
                    w,
                    np.mean(
                        calc_score_svr(
                            w, print_result=False
                        )[0]
                    )
                )
                for w in np.logspace(-8, 0, 10)
            ]),
            x=0, y=1, log_x=True, log_y=True,
            labels={"0": "weight", "1": "train core"}
        )
        fig.update_xaxes(exponentformat="power")
        plotly_utility.offline.mpl_plot(fig)
