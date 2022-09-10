from transformers import DataCollatorWithPadding
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pytorch_lightning as pl
import os


class FPELLDataset(Dataset):
    def __init__(self, df, tokenizer, max_length, is_test=False):
        self.df = df
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.discourse = df['discourse_text'].values
        self.type = df['discourse_type'].values
        self.essay = df['essay_text'].values
        self.targets = None if is_test else df['discourse_effectiveness'].values
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        discourse = self.discourse[index]
        type = self.type[index]
        essay = self.essay[index]
        text = type + " " + discourse + " " + self.tokenizer.sep_token + " " + essay
        # text = discourse
        inputs = self.tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_length
        )
        ret = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask']
        }
        if not self.is_test:
            ret["labels"] = self.targets[index]

        return ret


class FPEDataModule(pl.LightningDataModule):
    """
    DataFrameからモデリング時に使用するDataModuleを作成
    """

    def __init__(self, cfg, train_df=None, valid_df=None, test_df=None):
        super().__init__()

        # self._length = max(
        #     0 if train_df is None else len(train_df),
        #     0 if valid_df is None else len(valid_df),
        #     0 if test_df is None else len(test_df)
        # )
        # if train_df is not None and self._length != len(train_df):
        #     raise ValueError("train_df length mismatched")
        # if valid_df is not None and self._length != len(valid_df):
        #     raise ValueError("valid_df length mismatched")
        # if test_df is not None and self._length != len(test_df):
        #     raise ValueError("test_df length mismatched")

        self.cfg = cfg
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df

        self.tokenizer = None
        self.collate_fn = None

        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

    # def __len__(self):
    #     return self._length

    def setup(self, stage=None):
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)

        if self.train_df is not None:
            self.train_dataset = FPELLDataset(self.train_df, self.tokenizer, self.cfg.max_length)
        if self.valid_df is not None:
            self.valid_dataset = FPELLDataset(self.valid_df, self.tokenizer, self.cfg.max_length)
        if self.test_df is not None:
            self.test_dataset = FPELLDataset(self.test_df, self.tokenizer, self.cfg.max_length, is_test=True)

        self.collate_fn = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.train_batch_size,
            num_workers=os.cpu_count(), collate_fn=self.collate_fn,
            shuffle=True,
            persistent_workers=True,
            # pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.cfg.valid_batch_size,
            num_workers=os.cpu_count(), collate_fn=self.collate_fn,
            # shuffle=False,
            persistent_workers=True,
            # pin_memory=True
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.test_batch_size,
            num_workers=os.cpu_count(), collate_fn=self.collate_fn,
            # shuffle=False,
            persistent_workers=True,
            # pin_memory=True
        )
