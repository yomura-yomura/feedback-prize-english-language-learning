import torch
from transformers import DataCollatorWithPadding
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pytorch_lightning as pl
import os
import numpy as np


class FPELLDataset(Dataset):
    def __init__(self, df, tokenizer, max_length, target_columns):
        self.df = df
        self.max_length = max_length
        self.tokenizer = tokenizer
        # self.discourse = df['discourse_text'].values
        # self.type = df['discourse_type'].values
        # self.essay = df['essay_text'].values
        self.full_text = df["full_text"].values
        self.targets = (
            df[target_columns].values
            if np.all(np.isin(target_columns, df.columns)) else
            None
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = self.full_text[index]

        # text = discourse
        inputs = self.tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_length
        )
        ret = {
            'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.int),
            'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.int)
        }
        if self.targets is not None:
            ret["label"] = torch.tensor(self.targets[index], dtype=torch.float16)

        return ret


class FPELLDataModule(pl.LightningDataModule):
    """
    DataFrameからモデリング時に使用するDataModuleを作成
    """

    def __init__(self, cfg, train_df=None, valid_df=None, test_df=None):
        super().__init__()
        self.cfg = cfg
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df

        self.tokenizer = None
        self.collate_fn = None

        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.name)

        if stage == "fit":
            self.train_dataset = FPELLDataset(
                self.train_df, self.tokenizer,
                self.cfg.model.max_length, self.cfg.dataset.target_columns
            )
        if stage in ("fit", "validation"):
            self.valid_dataset = FPELLDataset(
                self.valid_df, self.tokenizer,
                self.cfg.model.max_length, self.cfg.dataset.target_columns
            )
        if stage == "predict":
            self.test_dataset = FPELLDataset(
                self.test_df, self.tokenizer,
                self.cfg.model.max_length, self.cfg.dataset.target_columns
            )

        self.collate_fn = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.dataset.train_batch_size,
            num_workers=os.cpu_count(), collate_fn=self.collate_fn,
            shuffle=True,
            persistent_workers=True,
            # pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.cfg.dataset.valid_batch_size,
            num_workers=os.cpu_count(), collate_fn=self.collate_fn,
            # shuffle=False,
            persistent_workers=True,
            # pin_memory=True
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.dataset.test_batch_size,
            num_workers=os.cpu_count(), collate_fn=self.collate_fn,
            # shuffle=False,
            persistent_workers=True,
            # pin_memory=True
        )
