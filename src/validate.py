import re
import FPELL.nn.from_checkpoint


if __name__ == "__main__":
    cfg, ckpt_paths = FPELL.nn.from_checkpoint.load_cfg_and_checkpoint_paths("models-lightning-deberta-xlarge", is_test=False)
    df = FPELL.nn.io.get_df(cfg)

    folds = [int(re.match(r".+fold(\d+).+", p.name)[1]) for p in ckpt_paths]

    validates = {
        ckpt_path.name: FPELL.nn.from_checkpoint.validate(ckpt_path, cfg, df[df["kfold"] == fold])
        for fold, ckpt_path in zip(folds, ckpt_paths)
    }
