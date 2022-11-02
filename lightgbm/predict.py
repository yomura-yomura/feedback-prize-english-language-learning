# import nltk
# nltk.download('all')
import FPELL.nn.lightgbm
import torch
import numpy as np
from nltk.tokenize import sent_tokenize
import os


class CFG:
    seed = 3655
    train_batch_size = 8  # 8以下にする,16だとGPUに載らないので落ちる
    valid_batch_size = 16
    max_length = 512
    learning_rate = 1e-5
    epochs = 4
    n_fold = 4
    num_classes = 3
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_root_path = "../../data/feedback-prize-effectiveness"
    nn_model_name = "microsoft/deberta-large"
    nn_checkpoint_path = "deberta-large-batch1"
    # lgbm_checkpoint_path = "feedback-lgbm-ver1"
    lgbm_checkpoint_path = "lgbm-smote"
    # nn_model_name = "../input/deberta-large"
    # nn_checkpoint_path = "../input/deberta-large-batch1"
    # lgbm_checkpoint_path = "../input/feedback-lgbm-ver1"

    reg_lgbm_checkpoint_path = "feedback-reg-lgbm-ver1"
    clf_lgbm_checkpoint_path = "feedback-lgbm-ver2"

    # dataset_type = "test"
    # debug = False
    dataset_type = "train"
    debug = True


if __name__ == "__main__":
    cfg = CFG
    model_type = "clf"
    # model_type = "reg+clf"

    is_train = cfg.dataset_type == "train"
    df = FPELL.data.io_with_cfg.get_df(cfg)

    emb_df_list = FPELL.lightgbm.add_features(cfg, df)
    if model_type == "clf":
        predicted_list = [
            FPELL.lightgbm.predict(
                emb_df, checkpoint_path=os.path.join(cfg.lgbm_checkpoint_path, f'lgbm_fold{fold}.pkl')
            )
            for fold, emb_df in enumerate(emb_df_list)
        ]
        predicted = np.mean(predicted_list, axis=0)
    elif model_type == "reg+clf":
        predicted_list = [
            FPELL.lightgbm.reg_clf_predict(
                emb_df,
                reg_path=os.path.join(cfg.reg_lgbm_checkpoint_path, f'lgbm_fold{fold}.pkl'),
                clf_path=os.path.join(cfg.clf_lgbm_checkpoint_path, f'lgbm_fold{fold}.pkl')
            )
            for fold, emb_df in enumerate(emb_df_list)
        ]

        predicted = np.mean(predicted_list, axis=0)
    else:
        raise ValueError(model_type)

    for i in range(len(predicted_list)):
        print(f"* fold {i}")
        print(predicted_list[i][:5])
    print("* averaged")
    print(predicted[:5])

