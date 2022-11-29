import pathlib
import pandas as pd
import numpy as np

import FPELL.data.io

if __name__ == "__main__":
    model_root_paths = [
        "20221121-025718-deberta-v3-base-10folds-8batch",
        "20221122-170337-deberta-v3-large-10folds-8batch"
    ]

    model_root_paths = list(map(pathlib.Path, model_root_paths))

    target_cols = [
        'cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions'
    ]

    all_predicted_data = []
    for model_root_path in model_root_paths:
        fold_csv_paths = [
            (int(p.name[4:-4]), p)
            for p in (model_root_path / "predicted_csv").glob("fold*.csv")
        ]
        fold_csv_paths = sorted(fold_csv_paths, key=lambda row: row[0])
        assert np.all(np.arange(len(fold_csv_paths)) == [row[0] for row in fold_csv_paths])

        df_list = [pd.read_csv(p) for _, p in fold_csv_paths]
        assert len(df_list) > 0
        common_folds = df_list[0]["fold"].to_numpy()
        assert all(np.all(common_folds == df["fold"]) for df in df_list)

        stacked_predicted = np.stack([df[target_cols].to_numpy() for df in df_list], axis=0)

        all_predicted_data.append(
            (common_folds, stacked_predicted)
        )

    assert len(all_predicted_data) > 0
    common_folds = all_predicted_data[0][0]
    assert all(np.all(folds == common_folds) for folds, _ in all_predicted_data)

    all_stacked_predicted = np.stack([stacked_predicted for _, stacked_predicted in all_predicted_data], axis=0)
    assert all_stacked_predicted.ndim == 4  # (n_models, n_folds, n_records, n_target_cols)

    df = FPELL.data.io.get_df("../data/feedback-prize-english-language-learning")

    from nn.validate import column_wise_rmse_loss

    oof_predicted_list = [
        all_stacked_predicted[:, fold, common_folds == fold, :]
        for fold in np.unique(common_folds)
    ]

    cv_list = [
        np.mean(
            column_wise_rmse_loss(
                np.mean(oof_predicted[:1], axis=0),
                df[target_cols][common_folds == fold].values
            )
        )
        for fold, oof_predicted in enumerate(oof_predicted_list)
    ]
    np.mean(cv_list)


