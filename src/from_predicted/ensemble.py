from plot_confusion_matrix import get_predicted, all_predicted_paths
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import plotly.express as px
import plotly_utility


def multi_log_loss(true, predicted):
    assert len(true) == len(predicted)
    predicted = np.clip(predicted, 1e-15, 1 - 1e-15)
    return -np.sum(true * np.log(predicted)) / len(true)


def run_logistic(
        plot_learning_curve=True, save_at_minimum_valid=True, use_better_model=True,
        penalty="l2", solver="newton-cg", C=0.01
):
    np.random.seed(42)

    previous_cv_list = []
    new_cv_list = []

    fold_clf_list = []

    for fold, df in enumerate(predicted):
        print(f"* fold {fold}")
        train_X = df[folds != fold].values
        valid_X = df[folds == fold].values
        train_Y = labels[folds != fold].values
        valid_Y = labels[folds == fold].values

        order = np.random.choice(len(train_X), size=len(train_X), replace=False)
        train_X = train_X[order]
        train_Y = train_Y[order]

        score_data = []
        clf_list = []
        for i_end in np.logspace(2, np.log10(len(train_X)), 20).astype(int):
            clf = LogisticRegression(
                penalty=penalty, solver=solver,
                tol=1e-3, C=C,
                max_iter=1000, multi_class="multinomial",
            )
            clf.fit(train_X[:i_end], train_Y[:i_end])
            percent = i_end / len(train_X) * 100
            score_data.append(
                (percent, "train", multi_log_loss(pd.get_dummies(train_Y).values, clf.predict_proba(train_X)))
            )
            score_data.append(
                (percent, "valid", multi_log_loss(pd.get_dummies(valid_Y).values, clf.predict_proba(valid_X)))
            )
            clf_list.append(clf)
        score_df = pd.DataFrame(score_data, columns=["Data used for training", "type", "score"])
        min_valid_score = np.min(score_df["score"][score_df["type"] == "valid"])

        if plot_learning_curve:
            fig = px.line(
                score_df, title=f"fold {fold}",
                x="Data used for training", y="score", color="type"
            ).update_xaxes(ticksuffix="%")
            fig.add_vline(
                x=score_df["Data used for training"].iloc[np.argmax(score_df["score"] == min_valid_score)],
                line_dash="dash"
            )
            # fig.write_image(f"{fold}.png", scale=2)
            plotly_utility.offline.mpl_plot(fig)

        if save_at_minimum_valid:
            clf = clf_list[np.argmax(score_df["score"][score_df["type"] == "valid"] == min_valid_score)]
        else:
            clf = clf_list[-1]

        previous_cv = [multi_log_loss(pd.get_dummies(valid_Y).values, valid_X[:, 3 * i: 3 * (i + 1)]) for i in
                       range(valid_X.shape[1] // 3)]
        previous_cv_list.append(previous_cv)
        new_cv = multi_log_loss(pd.get_dummies(valid_Y).values, clf.predict_proba(valid_X))
        new_train_cv = multi_log_loss(pd.get_dummies(train_Y).values, clf.predict_proba(train_X))

        if use_better_model and min(previous_cv) < new_cv:
            new_cv = min(previous_cv)
            clf = int(np.argmin(previous_cv))

        precious_cv = ", ".join(map("{:.3f}".format, previous_cv))

        new_cv_list.append(new_cv)
        tag = f"@{clf}" if isinstance(clf, int) else ""
        print(f"({precious_cv}) -> {new_cv:.3f}{tag} ({new_train_cv:.3f})")

        fold_clf_list.append(clf)

    mean_previous_cv = ", ".join(map("{:.3f}".format, np.mean(previous_cv_list, axis=0)))
    print(f"\n* 4-folds model: ({mean_previous_cv}) -> {np.mean(new_cv_list, axis=0):.3f}")
    return fold_clf_list, (previous_cv_list, np.mean(new_cv_list, axis=0))


# def run_weighted_average():
#     for fold, df in enumerate(predicted):
#         print(f"* fold {fold}")
#         train_X = df[folds != fold].values
#         valid_X = df[folds == fold].values
#         train_Y = labels[folds != fold].values
#         valid_Y = labels[folds == fold].values
#
#         np.sum([train_X[:, 3 * i: 3 * (i + 1)] for i in range(valid_X.shape[1] // 3)], axis=0)


if __name__ == "__main__":
    predicted = {}
    folds = {}
    labels = {}
    for p in all_predicted_paths:
        print(p)
        predicted[p], folds[p], labels[p] = get_predicted(p)

    import pathlib
    print(tuple(pathlib.Path(p).name for p in all_predicted_paths))

    predicted = [
        pd.concat([dfs[fold].add_suffix(f"_{i}") for i, dfs in enumerate(predicted.values())], axis=1)
        for fold in range(4)
    ]

    first, *others = folds.values()
    for other in others:
        assert np.all(first == other)
    folds = first
    
    first, *others = labels.values()
    for other in others:
        assert np.all(first == other)
    labels = first

    # Fitting
    import os

    # clf_list, cv = run_logistic()
    clf_list, cv = run_logistic(use_better_model=False, penalty="l2", solver="saga")
    # clf_list, _ = run_logistic(save_at_minimum_valid=False)

    logistic_regression_model_path = pathlib.Path("logistic-regression-models") / "_".join([os.path.basename(p) for p in all_predicted_paths])
    logistic_regression_model_path.mkdir(exist_ok=True, parents=True)
    import pickle
    for fold, clf in enumerate(clf_list):
        with open(logistic_regression_model_path / f"clf-fold{fold}.pkl", "wb") as f:
            pickle.dump(clf, f)
