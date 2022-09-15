import os
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
    model=dict(
        optimizer=dict(
            bits=None,
            scheduler=dict(
                cycle_interval_for_full_epochs=None
            )
        )
    ),
    train=dict(
        awp=None,
        accumulate_grad_batches=None
    )
)


def get_essay(data_path, essay_id):
    essay_path = os.path.join(data_path, f"{essay_id}.txt")
    essay_text = open(essay_path, 'r').read()
    return essay_text


def replace_encoding_with_utf8(error: UnicodeError) -> Tuple[bytes, int]:
    return error.object[error.start: error.end].encode("utf-8"), error.end


def replace_decoding_with_cp1252(error: UnicodeError) -> Tuple[str, int]:
    return error.object[error.start: error.end].decode("cp1252"), error.end


# Register the encoding and decoding error handlers for `utf-8` and `cp1252`.
codecs.register_error("replace_encoding_with_utf8", replace_encoding_with_utf8)
codecs.register_error("replace_decoding_with_cp1252", replace_decoding_with_cp1252)


def resolve_encodings_and_normalize(text: str) -> str:
    """Resolve the encoding problems and normalize the abnormal characters."""
    text = (
        text.encode("raw_unicode_escape")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
        .encode("cp1252", errors="replace_encoding_with_utf8")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
    )
    text = unidecode(text)
    return text


def get_df(cfg_dataset, seed):
    train_path = os.path.join(cfg_dataset.data_root_path, f"{cfg_dataset.dataset_type}.csv")
    df = pd.read_csv(train_path)

    df["full_text"] = df["full_text"].apply(resolve_encodings_and_normalize)  # TODO: いらないかもしれない

    if cfg_dataset.dataset_type == "train":
        # scores = np.where(df[cfg.target_columns] <= 2, "bad", np.where(df[cfg.target_columns] <= 3.5, "normal", "good"))
        # gf = sklearn.model_selection.KFold(n_splits=cfg.n_fold, random_state=cfg.seed, shuffle=True)
        gf = iterstrat.ml_stratifiers.MultilabelStratifiedKFold(
            n_splits=cfg_dataset.cv.n_folds, random_state=seed, shuffle=True
        )

        for fold, (_, val) in enumerate(gf.split(df, df[cfg_dataset.target_columns])):
            df.loc[val, "fold"] = fold
        df['fold'] = df['fold'].astype(int)

    return df


def _update_recursively_if_not_defined(cfg, base_cfg: dict):
    for k, v in base_cfg.items():
        if hasattr(cfg, k):
            if (
                isinstance(getattr(cfg, k), (dict, omegaconf.DictConfig))
                and
                isinstance(v, dict)
            ):
                _update_recursively_if_not_defined(getattr(cfg, k), v)
            continue

        warnings.warn(
            f"Given cfg does not have key '{k}'. Tt will be given with default value '{v}'",
            UserWarning
        )
        with omegaconf.open_dict(cfg):
            setattr(cfg, k, v)


def load_yaml_config(path):
    cfg = omegaconf.OmegaConf.load(path)
    omegaconf.OmegaConf.set_struct(cfg, True)
    _update_recursively_if_not_defined(cfg, default_cfg)
    return cfg
