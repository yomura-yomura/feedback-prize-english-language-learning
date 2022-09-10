import pathlib
import pandas as pd
import FPELL.nn
import omegaconf
from sklearn.metrics import confusion_matrix
import plotly.express as px
import plotly_utility
import numpy as np


all_predicted_paths = [
    # "../lgbm-classification-v3",
    "../lgbm-classification-v2",
    # "../deberta-v3-large-ver2",
    # "../deberta-v3-large",
    # "../deberta-large-batch1",
    # "../deberta-large-linear",
    # "../lgbm-classification-smote",
    # "../lgbm-classification",
    # "../lgbm-regression-classification",
    "../models-lightning-deberta-xlarge",
    # "../models-lightning-deberta-v2-xlarge"
]


def _validate_df(df):
    df = df[["Adequate", "Effective", "Ineffective"]]
    if len(df) == 36765:
        df = FPELL.fuyu.io._drop_duplicates(df).reset_index(drop=True)
    return df


def get_predicted(target_model_dir):
    target_model_dir = pathlib.Path(target_model_dir)
    assert target_model_dir.exists()

    predicted = [
        _validate_df(pd.read_csv(p))
        for p in sorted(target_model_dir.glob("train-predicted-fold*.csv"))
    ]
    assert len(predicted) == 4

    class CFG:
        dataset_type = "train"
        debug = False
        n_fold = 4
        data_root_path = "../../data/feedback-prize-effectiveness"

    df = FPELL.fuyu.io.get_df(CFG, False, drop_duplicates=True)
    return predicted, df["kfold"], df["discourse_effectiveness"]


def plot_confusion_matrix(target_model_dir):
    predicted, folds, labels = get_predicted(target_model_dir)

    predicted_for_valid = pd.concat([
        pdf[folds == fold]
        for fold, pdf in enumerate(predicted)
    ]).sort_index()

    cm = confusion_matrix(labels, predicted_for_valid.values.argmax(axis=1))

    fig = px.imshow(
        np.round(cm / cm.sum(axis=1, keepdims=True), 2),
        title=target_model_dir, color_continuous_scale="Blues", text_auto=True
    )
    plotly_utility.offline.mpl_plot(fig)


if __name__ == "__main__":
    for p in all_predicted_paths:
        plot_confusion_matrix(p)
