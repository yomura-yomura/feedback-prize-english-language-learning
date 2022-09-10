import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorWithPadding
from transformers import AutoModel, AutoConfig, AutoTokenizer
import gc
import os
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.preprocessing import LabelEncoder
import pickle


class FPELLDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.df = df
        self.max_len = max_length
        self.tokenizer = tokenizer
        self.discourse = df['discourse_text'].values
        self.type = df['discourse_type'].values
        self.essay = df['essay_text'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        discourse = self.discourse[index]
        type = self.type[index]
        essay = self.essay[index]
        text = type + " " + discourse + " " + self.tokenizer.sep_token + " " + essay
        inputs = self.tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len
        )

        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask']
        }


def prepare_test_loader(CFG, test_df, tokenizer):
    test_dataset = FPELLDataset(test_df, tokenizer=tokenizer, max_length=CFG.max_length)

    collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=CFG.valid_batch_size, collate_fn=collate_fn,
                             num_workers=2, shuffle=False, pin_memory=True, drop_last=False)
    return test_loader


class Model(nn.Module):
    def __init__(self, CFG):
        super(Model, self).__init__()
        self.config = AutoConfig.from_pretrained(CFG.nn_model_name, output_hidden_states=True)
        self.model = AutoModel.from_pretrained(CFG.nn_model_name, config=self.config)

        # 使ってないやつら
        self.drop = nn.Dropout(p = 0.2)
        #self.pooler = MeanPooling()
        self.Linear = nn.Linear(self.config.hidden_size, CFG.num_classes)
        # ノートブックを参考に追加
        # https://arxiv.org/pdf/1905.09788.pdf
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, ids, mask):
        out = self.model(input_ids=ids, attention_mask=mask,
                         output_hidden_states=False)
        return out


@torch.no_grad()
def nn_predict(test_loader, model, device):
    preds = []
    model.eval()
    model.to(device)

    bar = tqdm(enumerate(test_loader), total=len(test_loader))

    for step, data in bar:
        ids = data['input_ids'].to(device, dtype=torch.long)
        mask = data['attention_mask'].to(device, dtype=torch.long)

        output = model(ids, mask)
        last_hidden_state = output["last_hidden_state"].to('cpu').numpy()
        last_hidden_state = np.mean(last_hidden_state, axis=1)

        preds.append(last_hidden_state)
    return preds


def get_embedding(CFG, test_df):
    tokenizer = AutoTokenizer.from_pretrained(CFG.nn_model_name)

    embedding = []
    test_loader = prepare_test_loader(CFG, test_df, tokenizer)

    for fold in range(CFG.n_fold):
        print("Fold {}".format(fold))

        model = Model(CFG)
        model.load_state_dict(
            torch.load(
                os.path.join(CFG.nn_checkpoint_path, f'microsoft-deberta-large_fold{fold}_7-28_batch1.pth')
            )
        )

        emb = nn_predict(test_loader, model, CFG.device)
        embedding.append(emb)
        del model, emb
        gc.collect()
        torch.cuda.empty_cache()
    return embedding


def add_features(cfg, df):
    def get_emb_df(emb, fold):
        emb_df = pd.DataFrame([j for i in emb[fold] for j in i])
        return emb_df

    df['n_text'] = df['discourse_text'].str.len()  # 文字数
    df['n_word'] = df['discourse_text'].map(lambda x: len(x.split()))  # 単語数
    df['n_sentence'] = df['discourse_text'].map(lambda x: len(sent_tokenize(x)))  # 文章数

    embedding = get_embedding(cfg, df)

    encoder = LabelEncoder()
    df['discourse_type'] = encoder.fit_transform(df['discourse_type'])

    emb_df_list = []
    for fold in range(len(embedding)):
        emb_df = get_emb_df(embedding, fold)
        emb_df['discourse_type'] = df['discourse_type']
        emb_df['n_text'] = df['n_text']
        emb_df['n_word'] = df['n_word']
        emb_df['n_sentence'] = df['n_sentence']
        emb_df_list.append(emb_df)

    return emb_df_list


def predict(df, checkpoint_path):
    model = pickle.load(open(checkpoint_path, 'rb'))
    pred = model.predict(df)
    return pred


# 回帰


# 回帰の結果をラベル化
def reg_result_bin(score):
    if score > 2.6:
        reg_label = 1   # Effective or Adequate?
    elif score < 1.4:
        reg_label = 2  # Ineffective or Adequate?
    else:
        reg_label = 0   # Adequate?
    return reg_label


def reg_clf_predict(df, reg_path, clf_path):
    model = pickle.load(open(reg_path, 'rb'))
    pred = model.predict(df)

    em_reg_df = df[np.arange(1024)].copy()
    em_reg_df['discourse_type'] = df['discourse_type']
    em_reg_df['reg_pred'] = pred
    em_reg_df['reg_label'] = em_reg_df['reg_pred'].map(lambda x: reg_result_bin(x))
    em_reg_df['n_text'] = df['n_text']
    em_reg_df['n_word'] = df['n_word']
    em_reg_df['n_sentence'] = df['n_sentence']

    return predict(em_reg_df, clf_path)
