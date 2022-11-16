from . import io as _io_module


def get_df(cfg):
    return _io_module.get_df(
        data_root_path=cfg.dataset.data_root_path, dataset_type=cfg.dataset.dataset_type,
        cv_n_folds=cfg.dataset.cv.n_folds, cv_seed=cfg.seed, cv_target_columns=cfg.dataset.target_columns,
        exclude_escape_characters=cfg.dataset.exclude_escape_characters,
        replace_full_text_with_summary=cfg.dataset.replace_full_text_with_summary
    )
