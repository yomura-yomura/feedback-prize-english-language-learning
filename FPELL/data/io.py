import os
import pathlib
import pandas as pd
from text_unidecode import unidecode
from typing import Dict, List, Tuple
import codecs
from sklearn.preprocessing import LabelEncoder
import sklearn.model_selection
import numpy as np
import iterstrat.ml_stratifiers
import omegaconf
import warnings


default_cfg = dict(
    dataset=dict(exclude_escape_characters=True, replace_full_text_with_summary=False),
    model=dict(
        optimizer=dict(bits=None, scheduler=dict(cycle_interval_for_full_epochs=None))
    ),
    train=dict(awp=None, accumulate_grad_batches=None),
)


this_dir_path = pathlib.Path(__file__).resolve().parent
_fold_csv_root_path = this_dir_path / "_fold_csv"


def get_essay(data_path, essay_id):
    essay_path = os.path.join(data_path, f"{essay_id}.txt")
    essay_text = open(essay_path, "r").read()
    return essay_text


def replace_encoding_with_utf8(error: UnicodeError) -> Tuple[bytes, int]:
    return error.object[error.start : error.end].encode("utf-8"), error.end  # type: ignore


def replace_decoding_with_cp1252(error: UnicodeError) -> Tuple[str, int]:
    return error.object[error.start : error.end].decode("cp1252"), error.end  # type: ignore


# Register the encoding and decoding error handlers for `utf-8` and `cp1252`.
codecs.register_error("replace_encoding_with_utf8", replace_encoding_with_utf8)
codecs.register_error("replace_decoding_with_cp1252", replace_decoding_with_cp1252)


def _resolve_encodings_and_normalize(text: str) -> str:
    """Resolve the encoding problems and normalize the abnormal characters."""
    text = (
        text.encode("raw_unicode_escape")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
        .encode("cp1252", errors="replace_encoding_with_utf8")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
    )
    text = unidecode(text)
    return text


def _exclude_escape_characters(full_text):
    for escape_character_to_exclude in ("\n", "\r"):
        if escape_character_to_exclude not in full_text:
            continue
        full_text = " ".join(
            seperated
            for seperated in full_text.split(escape_character_to_exclude)
            if seperated != ""
        )
    return full_text


def get_df(
    data_root_path,
    dataset_type="train",
    cv_n_folds=None,
    cv_seed=None,
    cv_target_columns=None,
    exclude_escape_characters=True,
    replace_full_text_with_summary=False,
):
    train_path = os.path.join(data_root_path, f"{dataset_type}.csv")
    df = pd.read_csv(train_path)

    # if resolve_encodings_and_normalize:
    #     assert np.all(df["full_text"].apply(_resolve_encodings_and_normalize) == df["full_text"])  # 変換しても変化なかったので必要なし
    #     df["full_text"] = df["full_text"].apply(_resolve_encodings_and_normalize)
    if exclude_escape_characters:
        df["full_text"] = df["full_text"].apply(_exclude_escape_characters)
    if replace_full_text_with_summary:
        summary_df = pd.read_csv(
            pathlib.Path(data_root_path).resolve().parent
            / "Summarized_train_data_42454.csv"
        )
        assert len(summary_df) == len(df), (len(df), len(summary_df))
        assert df["text_id"].unique().size == len(df), len(df)
        assert summary_df["text_id"].unique().size == len(summary_df), len(summary_df)
        summary_df = summary_df.set_index("text_id")
        df = df.set_index("text_id")
        df["full_text"] = summary_df["Pegasus_large"]
        df = df.reset_index()

    if dataset_type == "train":
        if cv_n_folds is not None:
            if cv_seed is None or cv_target_columns is None:
                raise ValueError(
                    f"cv_seed/cv_target_columns must not be None if cv_n_folds is not None"
                )

            if len(cv_target_columns) == 1:
                cv_target_columns = list(cv_target_columns) * 2
            gf = iterstrat.ml_stratifiers.MultilabelStratifiedKFold(
                n_splits=cv_n_folds, random_state=cv_seed, shuffle=True
            )
            # else:
            #     assert len(cv_target_columns) == 1
            #     cv_target_columns = cv_target_columns[0]
            #     gf = sklearn.model_selection.StratifiedKFold(
            #         n_splits=cv_n_folds, random_state=cv_seed, shuffle=True
            #     )

            for fold, (_, val) in enumerate(gf.split(df, df[cv_target_columns])):
                df.loc[val, "fold"] = fold
            df["fold"] = df["fold"].astype(int)
            fold_csv_path = (
                _fold_csv_root_path
                / f"{cv_n_folds}_{cv_seed}_{'-'.join(cv_target_columns)}.csv"
            )
            if fold_csv_path.exists():
                fold_df = pd.read_csv(fold_csv_path)
                assert np.all(df["text_id"] == fold_df["text_id"])
                df["fold"] = fold_df["fold"]
            else:
                if len(cv_target_columns) == 1:
                    cv_target_columns = list(cv_target_columns) * 2
                gf = iterstrat.ml_stratifiers.MultilabelStratifiedKFold(
                    n_splits=cv_n_folds, random_state=cv_seed, shuffle=True
                )
                # else:
                #     assert len(cv_target_columns) == 1
                #     cv_target_columns = cv_target_columns[0]
                #     gf = sklearn.model_selection.StratifiedKFold(
                #         n_splits=cv_n_folds, random_state=cv_seed, shuffle=True
                #     )

                for fold, (_, val) in enumerate(gf.split(df, df[cv_target_columns])):
                    df.loc[val, "fold"] = fold
                df["fold"] = df["fold"].astype(int)
                df[["text_id", "fold"]].to_csv(fold_csv_path, index=False)

    return df


def _update_recursively_if_not_defined(cfg, base_cfg: dict):
    for k, v in base_cfg.items():
        if hasattr(cfg, k):
            if isinstance(getattr(cfg, k), (dict, omegaconf.DictConfig)) and isinstance(
                v, dict
            ):
                _update_recursively_if_not_defined(getattr(cfg, k), v)
            continue

        warnings.warn(
            f"Given cfg does not have key '{k}'. Tt will be given with default value '{v}'",
            UserWarning,
        )
        with omegaconf.open_dict(cfg):
            setattr(cfg, k, v)


def load_yaml_config(path):
    cfg = omegaconf.OmegaConf.load(path)
    omegaconf.OmegaConf.set_struct(cfg, True)
    _update_recursively_if_not_defined(cfg, default_cfg)
    return cfg
