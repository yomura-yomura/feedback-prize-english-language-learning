import pathlib
import sys
import pandas as pd
import os
import FPELL.nn.from_checkpoint


if __name__ == "__main__":
    if len(sys.argv) == 1:
        target_dir = "models/microsoft-deberta-xlarge_folds4_v0.2"
    else:
        target_dir = sys.argv[1]

    target_dir = pathlib.Path(target_dir)
    assert target_dir.exists()

    cfg, ckpt_paths = FPELL.nn.from_checkpoint.load_cfg_and_checkpoint_paths(target_dir)
    cfg.dataset.dataset_type = "train"
    cfg.dataset.test_batch_size = 4

    df = FPELL.data.io.get_df(cfg.dataset, cfg.seed)

    # if is_test:
    #     predicted_dict = {
    #         ckpt_path.name: FPELL.nn.from_checkpoint.predict(ckpt_path, cfg, df)
    #         for ckpt_path in ckpt_paths
    #     }
    #     predicted = np.mean(list(predicted_dict.values()), axis=0)
    #
    #     submission_df = pd.read_csv(os.path.join(cfg.data_root_path, "sample_submission.csv"))
    #     submission_df["Adequate"] = predicted[:, 0]
    #     submission_df["Effective"] = predicted[:, 1]
    #     submission_df["Ineffective"] = predicted[:, 2]
    #
    #     submission_df.to_csv('submission.csv', index=False)
    # else:

    target_dir /= "predicted_csv"
    target_dir.mkdir(exist_ok=True)

    import re
    folds = [int(re.match(r".+fold(\d+).+", p.name)[1]) for p in ckpt_paths]
    for fold, ckpt_path in zip(folds, ckpt_paths):
        target_fn = target_dir / f"fold{fold}.csv"
        if target_fn.exists():
            print(f"* Skipped fold {fold}")
            continue
        predicted = FPELL.nn.from_checkpoint.predict(ckpt_path, cfg, df)
        predicted_df = pd.DataFrame(predicted, columns=cfg.dataset.target_columns)
        predicted_df.to_csv(target_fn, index=False)
