import argparse
import FPELL.data.io
import pandas as pd
from tqdm.auto import tqdm
import torch
import pathlib
import gc
from torch.utils.data import DataLoader
import tokenizers
import transformers
print(f"tokenizers.__version__: {tokenizers.__version__}")
print(f"transformers.__version__: {transformers.__version__}")
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from helper_functions import *
from model import CustomModel
from dataset import TestDataset
from nn.validate import column_wise_rmse_loss



def inference_fn(test_loader, model, device):
    preds = []
    model.eval()
    model.to(device)
    tk0 = tqdm(test_loader, total=len(test_loader))
    for inputs in tk0:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(inputs)
        preds.append(y_preds.to('cpu').numpy())
    predictions = np.concatenate(preds)
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    args = parser.parse_args()
    model_root_path = args.model_path

    class CFG:
        num_workers=4
        path=model_root_path + "/"
        config_path=os.path.join(model_root_path, 'config', 'config.json')
        # model="microsoft/deberta-v3-base"
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_root_path, 'tokenizer'))
        gradient_checkpointing=False
        # batch_size = 24  # 4
        batch_size = 8
        target_cols=['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
        seed=42
    #     n_fold=4
    #     trn_fold=list(range(n_fold))
    #     pooling = 'attention'
        layer_start = 4


    import re
    matched = re.match(".*(deberta-(?:v3)-(?:large|base)).*", pathlib.Path(model_root_path).name)
    model_name = matched[1]
    CFG.model = f"microsoft/{model_name}"
    if pathlib.Path(model_root_path).name == "KOJIMAR-deberta-v3-large":
        CFG.pooling = "mean"
    else:
        CFG.pooling = "attention"

    df = FPELL.data.io.get_df("../data/feedback-prize-english-language-learning")
    oof_df = pd.read_csv(pathlib.Path(model_root_path) / "oof_df.csv")
    df = pd.merge(df, oof_df[["text_id", "fold"]], on="text_id")

    print(pathlib.Path(CFG.path))
    pth_paths = []
    for p in pathlib.Path(CFG.path).glob("*.pth"):
        matched = re.match(f"{CFG.model.replace('/', '-')}_fold(\d+)_best.pth", p.name)
        if matched is None:
            continue
        fold = int(matched[1])
        pth_paths.append(
            (fold, p)
        )
    pth_paths = sorted(pth_paths, key=lambda row: row[0])

    test_dataset = TestDataset(CFG, df)
    test_loader = DataLoader(test_dataset,
                             batch_size=CFG.batch_size,
                             shuffle=False,
                             collate_fn=DataCollatorWithPadding(tokenizer=CFG.tokenizer, padding='longest'),
                             num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
    predictions = []
    for fold, model_weights_path in pth_paths:
        predicted_csv_path = pathlib.Path(model_root_path) / "predicted_csv" / f"fold{fold}.csv"
        print(f"* {predicted_csv_path}")
        predicted_csv_path.parent.mkdir(exist_ok=True)
        if predicted_csv_path.exists():
            prediction = pd.read_csv(predicted_csv_path)[CFG.target_cols].values
        else:
            model = CustomModel(CFG, config_path=CFG.config_path, pretrained=False)
            state = torch.load(model_weights_path, map_location=torch.device('cpu'))
            model.load_state_dict(state['model'])
            prediction = inference_fn(test_loader, model, device)
            del model, state
            prediction_df = pd.DataFrame(prediction, columns=CFG.target_cols)
            prediction_df["fold"] = df["fold"]
            prediction_df.to_csv(predicted_csv_path, index=False)

        cv_list = column_wise_rmse_loss(
            prediction[oof_df["fold"] == fold],
            df[CFG.target_cols].values[oof_df["fold"] == fold]
        )
        print(np.mean(cv_list), cv_list)

        predictions.append(prediction)
        del prediction
        gc.collect()
        torch.cuda.empty_cache()



    score_matrix = np.array([
        column_wise_rmse_loss(
            predictions[fold][oof_df["fold"] == fold],
            df[CFG.target_cols].values[oof_df["fold"] == fold]
        )
        for fold in oof_df["fold"].unique()
    ])

    for target_col, mean_score, std_score in zip(
        CFG.target_cols,
        np.mean(score_matrix, axis=0),
        np.std(score_matrix, axis=0)
    ):
        print(f"{target_col:<12} = {mean_score:.2f} ± {std_score:.2f}")

    mean_scores = np.mean(score_matrix, axis=1)
    print(f"CV: {np.mean(mean_scores):.3f} ± {np.std(mean_scores):.3f} ({' '.join(map('{:.2f}'.format, mean_scores))})")
