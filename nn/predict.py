import pathlib
import pandas as pd
import os
import numpy as np
import FPELL.nn.from_checkpoint


os.environ["TOKENIZERS_PARALLELISM"] = "false"


if __name__ == "__main__":
    # is_test = True
    is_test = False
    target_dir = "models-lightning-deberta-v2-xlarge"

    target_dir = pathlib.Path(target_dir)
    assert target_dir.exists()

    cfg, ckpt_paths = FPELL.nn.from_checkpoint.load_cfg_and_checkpoint_paths(target_dir, is_test=is_test)
    df = FPELL.nn.io.get_df(cfg, False, False)

    if is_test:
        predicted_dict = {
            ckpt_path.name: FPELL.nn.from_checkpoint.predict(ckpt_path, cfg, df)
            for ckpt_path in ckpt_paths
        }
        predicted = np.mean(list(predicted_dict.values()), axis=0)

        submission_df = pd.read_csv(os.path.join(cfg.data_root_path, "sample_submission.csv"))
        submission_df["Adequate"] = predicted[:, 0]
        submission_df["Effective"] = predicted[:, 1]
        submission_df["Ineffective"] = predicted[:, 2]

        submission_df.to_csv('submission.csv', index=False)
    else:
        import re
        folds = [int(re.match(r".+fold(\d+).+", p.name)[1]) for p in ckpt_paths]
        for fold, ckpt_path in zip(folds, ckpt_paths):
            target_fn = target_dir / f"train-predicted-fold{fold}.csv"
            if target_fn.exists():
                print(f"* Skipped fold {fold}")
                continue
            predicted = FPELL.nn.from_checkpoint.predict(ckpt_path, cfg, df)
            predicted_df = pd.DataFrame(predicted, columns=["Adequate", "Effective", "Ineffective"])
            predicted_df.to_csv(target_fn, index=False)
