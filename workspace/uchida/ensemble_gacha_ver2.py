import optuna
from sqlite3 import connect
import numpy as np
from omegaconf import OmegaConf
import itertools
import multiprocessing as mp
import pandas as pd
from pathlib import Path
import random
from logging import FileHandler, Formatter, StreamHandler, getLogger
import warnings

warnings.simplefilter("ignore")

n_cpus = 8
use_multiprocessing = False


def log_handler(output_dir: str):
    verbose_fmt = Formatter(
        "%(asctime)s %(levelname)-6s %(name)s %(lineno)d [%(funcName)s] %(message)s"
    )

    verbose_logger = optuna.logging.get_logger("verbose")
    verbose_logger.setLevel("DEBUG")

    handler = StreamHandler()
    handler.setLevel("INFO")
    handler.setFormatter(verbose_fmt)
    verbose_logger.addHandler(handler)

    handler = FileHandler(f"{output_dir}/optuna.log", mode="a", encoding="utf8")
    handler.setLevel("DEBUG")
    handler.setFormatter(verbose_fmt)
    verbose_logger.addHandler(handler)
    return verbose_logger


def column_wise_rmse_loss(pred, true):
    return np.sqrt(np.mean((pred - true) ** 2, axis=0))


def get_score(fold: int, list_key: list, list_df: list, list_params_cat: list):
    sum_params = np.array(list_params_cat).sum()
    score_pred = None
    for num_params, df in enumerate(list_df):
        df_fold = df.query("fold == @fold")
        pred = df_fold[list_key[0]]
        pred *= list_params_cat[num_params]
        if num_params == 0:
            score_pred = pred
        else:
            score_pred += pred

    df_fold = list_df[0].query("fold == @fold")
    true = df_fold[list_key[1]]
    score_pred /= sum_params
    score = column_wise_rmse_loss(score_pred, true)
    return score


def each_step(args):
    (list_params, list_dirs, list_key) = args

    list_df = [pd.read_csv(Path(dirs)) for dirs in list_dirs]

    score = []
    for fold in sorted(list(list_df[0]["fold"].unique())):
        # Cohesion # Suntax # Vocabulary # Phraseology # Grammar # Conventions
        score.append(
            get_score(
                fold=fold,
                list_key=list_key,
                list_df=list_df,
                list_params_cat=list_params,
            )
        )

    score = np.array(score)

    return score


def objective(trial):
    optuna.logging.disable_default_handler()
    optuna.logging.disable_propagation()
    logging = log_handler(output_dir=output_path)
    list_sub_params = [
        trial.suggest_float(f"sub_{i}", 0, 1.0, step=0.00001)
        for i in range(len(list_dirs))
    ]

    iters = zip(
        [itertools.repeat(sub) for sub in list_sub_params],
        list_dirs,
    )
    # if use_multiprocessing:
    #    with mp.Pool(n_cpus) as p:
    #        results = p.imap_unordered(each_step, iters)
    #        # scores = list(results)
    scores = each_step(args=(list_sub_params, list_dirs, list_key))

    mean_scores = scores.mean()
    max_scores = scores.max()
    try:
        best_score = study.best_value
        num_trial = trial.number
    except:
        best_score = np.nan
        num_trial = 0

    if num_trial % (n_trials / 10) == 0:
        logging.info(
            f"Best_score:{best_score} Avg_score:{mean_scores} Max_score:{max_scores} Each_score: {scores}"
        )
    return_score = mean_scores
    return return_score


if __name__ == "__main__":
    rand = random.randint(1, 100000)
    config = OmegaConf.load("workspace/uchida/config/gachav2.yaml")
    n_trials = config.params.n_trials
    db_path = Path(config.params.db_path)
    study_name = config.params.study_name + "_" + str(rand)
    seed = config.params.seed

    output_path = "result_optuna/" + str(study_name)

    Path(output_path).mkdir(exist_ok=False)
    OmegaConf.save(config, Path(output_path) / "config.yaml")

    db_path = Path(output_path) / db_path

    list_dirs = [
        "nn_MOZATTT/20221125-110622-deberta-v3-base/oof_df.csv",  # CV 0.4546
        # "nn_MOZATTT2/20221127-003609-deberta-v3-base_fb2021/oof_df.csv",  # CV 0.4507
        # "nn_MOZATTT3/20221127-044726-deberta-v3-base_fb2021/oof_df.csv",  # CV 0.4468
        "nn_MOZATTT3/20221126-084223-deberta-v3-base_fb2021/oof_df.csv",  # CV 0.4451
    ]

    list_pred_true = [
        ["pred_cohesion", "cohesion"],
        ["pred_syntax", "syntax"],
        ["pred_vocabulary", "vocabulary"],
        ["pred_phraseology", "phraseology"],
        ["pred_grammar", "grammar"],
        ["pred_conventions", "conventions"],
    ]
    list_best_params = [list_dirs]
    list_best_scores = []
    for key_num, list_key in enumerate(list_pred_true):
        _study_name = study_name + "_" + list_key[1]
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=seed),
            # pruner=optuna.pruners.MedianPruner(
            #    n_min_trials=5, n_warmup_steps=5, n_startup_trials=10
            # ),
            # storage=f"sqlite:///{db_path}",
            study_name=_study_name,
        )
        study.optimize(
            objective,
            n_trials=n_trials,
        )

        list_best_params.append(list(study.best_params.values()))
        list_best_scores.append(study.best_value)

    list_best_scores.append(np.array(list_best_scores).mean())
    score_df = pd.DataFrame(
        list_best_scores,
        index=[
            "cohesion",
            "syntax",
            "vocabulary",
            "phraseology",
            "grammar",
            "conventions",
            "Mean_Score",
        ],
    )
    result_df = pd.DataFrame(
        list_best_params,
        index=[
            "file_name",
            "cohesion",
            "syntax",
            "vocabulary",
            "phraseology",
            "grammar",
            "conventions",
        ],
    )
    result_df.to_csv(Path(output_path) / "result.csv")
    score_df.to_csv(Path(output_path) / "score.csv")
    print("debug")
