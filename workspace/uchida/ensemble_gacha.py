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
    # optuna.logging.enable_default_handler()
    # optuna.logging.enable_propagation()

    verbose_logger = optuna.logging.get_logger("optuna")
    verbose_logger.setLevel("DEBUG")

    handler = StreamHandler()
    handler.setLevel("INFO")
    handler.setFormatter(verbose_fmt)
    verbose_logger.addHandler(handler)

    handler = FileHandler(f"{output_dir}/optuna.log", mode="a", encoding="utf8")
    handler.setLevel("DEBUG")
    handler.setFormatter(verbose_fmt)
    verbose_logger.addHandler(handler)
    return None


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
    (list_params, list_dirs) = args

    list_df = [pd.read_csv(Path(dirs)) for dirs in list_dirs]

    score = [[] for _ in range(6)]
    list_pred_true = [
        ["pred_cohesion", "cohesion"],
        ["pred_syntax", "syntax"],
        ["pred_vocabulary", "vocabulary"],
        ["pred_phraseology", "phraseology"],
        ["pred_grammar", "grammar"],
        ["pred_conventions", "conventions"],
    ]
    for fold in sorted(list(list_df[0]["fold"].unique())):
        # Cohesion # Suntax # Vocabulary # Phraseology # Grammar # Conventions
        for i in range(6):
            score[i].append(
                get_score(
                    fold=fold,
                    list_key=list_pred_true[i],
                    list_df=list_df,
                    list_params_cat=list_params[i],
                )
            )

    score = np.array(score)

    return score


def objective(trial):
    log_handler(output_dir=output_path)
    list_sub_params = [
        [
            round(trial.suggest_float(f"sub_{i}_{j}", 0, 1), 3)
            for i in range(len(list_dirs))
        ]
        for j in range(6)
    ]
    iters = zip(
        [[itertools.repeat(sub) for sub in list_sub] for list_sub in list_sub_params],
        list_dirs,
    )
    # if use_multiprocessing:
    #    with mp.Pool(n_cpus) as p:
    #        results = p.imap_unordered(each_step, iters)
    #        # scores = list(results)
    scores = each_step(args=(list_sub_params, list_dirs))

    scores = scores.mean(axis=1)

    # return_score = np.array(scores).mean()
    return_score = scores.mean()
    return return_score


if __name__ == "__main__":
    rand = random.randint(1, 100000)
    config = OmegaConf.load("workspace/uchida/config/gacha.yaml")
    n_trials = config.params.n_trials
    db_path = Path(config.params.db_path)
    study_name = config.params.study_name + "_" + str(rand)
    seed = config.params.seed

    output_path = "result_optuna/" + str(study_name)

    Path(output_path).mkdir(exist_ok=False)
    OmegaConf.save(config, Path(output_path) / "config.yaml")

    db_path = Path(output_path) / db_path

    list_dirs = [
        "nn_MOZATTT3/20221127-044726-deberta-v3-base_fb2021/oof_df.csv",
        "nn_MOZATTT/20221125-110622-deberta-v3-base/oof_df.csv",
        "nn_MOZATTT2/20221127-003609-deberta-v3-base_fb2021/oof_df.csv",
    ]

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        storage=f"sqlite:///{db_path}",
        study_name=study_name,
    )
    study.optimize(
        objective,
        n_trials=n_trials,
    )

    print("debug")
