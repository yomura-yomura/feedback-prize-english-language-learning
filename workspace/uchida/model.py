from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from summarizer import Summarizer, TransformerSummarizer  # type: ignore
from transformers import (
    AutoModelWithLMHead,
    AutoTokenizer,
    PegasusForConditionalGeneration,
    PegasusTokenizer,
    BartTokenizer,
    BartForConditionalGeneration,
)
from tqdm import tqdm
import FPELL.data
import warnings
import torch

warnings.simplefilter("ignore")


class SummaryModel:
    def __init__(self, cfg_path: str):
        self.cfg = FPELL.data.io.load_yaml_config(Path(cfg_path))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.data: pd.DataFrame = self.__load_data()
        (
            self.min_length,
            self.max_length,
            self.use_first,
            self.num_beams,
        ) = self.set_params()

    def __load_data(self) -> pd.DataFrame:
        """get train text data frame
        Returns
        -------
        pd.DataFrame
            _description_
        """
        df = FPELL.data.io_with_cfg.get_df(self.cfg)
        return df

    def set_params(self) -> Tuple[int, int, bool, int]:
        """Set params via cfg

        Returns
        -------
        Tuple[int,int,bool,int]
            min_length: 要約結果の最小単語数
            max_length: 要約結果の最大単語数
            use_first: 入力文の最初が重要であると判断するか
            num_beams: beam-searchのパラメーター
        """
        min_length = self.cfg.params.min_length
        max_length = self.cfg.params.max_length

        num_beams = self.cfg.params.num_beams
        use_first = self.cfg.params.use_first

        return min_length, max_length, use_first, num_beams

    def BERTSUMExt(self, text: str) -> str:
        """抽出型モデル BERTSUMExt
        ERTSUMExtはBERTを要約タスク用にファインチューニングすることなく、
        事前学習のみを行ったBERTを用いて要約タスクを行います。
        BERTSUMExtではBERTの中間層の値からK-means法を用いてクラスタリングを行い、
        各クラスタのセントロイドに最も近い文章を要約の候補とするということを行う。

        Parameters
        ----------
        text : str
            要約したい文章

        Returns
        -------
        str
            要約結果
        """
        model = Summarizer()
        generate_text = model(text, min_length=self.min_length, max_length=self.max_length, use_first=self.use_first)  # type: ignore
        return generate_text

    def GPT2(self, text: str) -> str:
        """抽出型モデル GPT-2

        BERTSUMExtと同じ要領で中間層の値からK-means法を用いてセントロイドから最も近い文章を要約の文章の候補とする文章要約を行う。
        文章のトークナイズ部分ではByte Pair Encoding (BPE)という出現頻度が稀な単語は文字単位に分割するという手法を用いてトークナイズしている。

        Parameters
        ----------
        text : str
            要約したい文章

        Returns
        -------
        str
            要約結果
        """
        model = TransformerSummarizer(
            transformer_type="GPT2", transformer_model_key="gpt2-medium"
        )
        generate_text = "".join(
            model(
                text,
                min_length=self.min_length,
                max_length=self.max_length,
                use_first=self.use_first,
            )
        )
        return generate_text

    def T5(self, text: str, model_name: str) -> str:
        """抽象型モデル T5
        Googleが提供しているC4で事前学習だけさせたmodel

        Parameters
        ----------
        text : str
            要約したい文章

        Returns
        -------
        str
            要約結果
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelWithLMHead.from_pretrained(model_name).to(self.device)
        inputs = tokenizer.encode(
            "summarize:" + text, return_tensors="pt", truncation=True
        ).to(self.device)
        summary_ids = model.generate(
            inputs,
            max_length=self.max_length,
            num_beams=self.num_beams,
            early_stopping=True,
        )
        generate_text = [
            tokenizer.decode(
                g, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            for g in summary_ids
        ]
        generate_text = generate_text[0]
        return generate_text

    def Pegasus(self, text: str, model_name: str) -> str:
        """抽象型モデル　PEGASUS
        Googleが提供している事前学習のみのモデル
        PEGASUSは正式名称は
        『Pretraining with Extracted Gap-sentences for Abstractive Summarization Sequence-to-sequence models』。
        モデルの構造はBERTと全く同じなのですが、事前学習にGap Sentence Generation(GSG)を用いるのが大きな特徴。
        GSGは入力文の一部をトークン単位ではなく、文章単位でマスクする。
        この手法は、事前学習の方法は転移学習やファインチューニングで用いられるタスクに似ているほど
        学習が高速かつ性能が良くなるという仮説の元採用されている。

        Parameters
        ----------
        text : str
            要約したい文章

        model_name:str
            pretrained_model

        Returns
        -------
        str
            要約結果
        """
        model = PegasusForConditionalGeneration.from_pretrained(model_name).to(
            self.device
        )  # type: ignore
        tokenizer = PegasusTokenizer.from_pretrained(model_name)
        inputs = tokenizer([text], return_tensors="pt", truncation=True).to(self.device)
        summary_ids = model.generate(  # type: ignore
            inputs["input_ids"],
            num_beams=self.num_beams,
            max_length=self.max_length,
            early_stopping=True,
        )
        generate_text = [
            tokenizer.decode(
                g, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            for g in summary_ids
        ]
        generate_text = generate_text[0]
        return generate_text

    def Bart(self, text: str, model_name: str) -> str:
        """抽象型モデル　Bart-large
        基本ファインチューニングしないと役に立たない

        Parameters
        ----------
        text : str
            要約したい文章

        Returns
        -------
        str
            要約結果
        """
        model = BartForConditionalGeneration.from_pretrained(model_name).to(self.device)
        tokenizer = BartTokenizer.from_pretrained(model_name)
        split_list = text.split(".")
        text = ".\n".join(split_list)
        inputs = tokenizer([text], return_tensors="pt", truncation=True).to(self.device)
        summary_ids = model.generate(  # type: ignore
            inputs["input_ids"],
            num_beams=self.num_beams,
            max_length=self.max_length,
            early_stopping=True,
        )
        generate_text = [
            tokenizer.decode(
                g, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            for g in summary_ids
        ]
        generate_text = generate_text[0]
        return generate_text

    def summarize(self) -> pd.DataFrame:
        """要約結果のデータフレームを返す

        Returns
        -------
        pd.DataFrame
            text_id,model_name

        Raises
        ------
        Exception
            該当しないモデル名が記入されたときに出るエラー
        """
        list_model_name = self.cfg.params.use_models
        array_summarize = [["text_id"] + list_model_name]

        for row in tqdm(self.data.values):
            list_result = [row[0]]  # add text_id
            text = row[1]
            for model_name in list_model_name:
                if "BERTSUMExt" == model_name:
                    list_result.append(self.BERTSUMExt(text=text))

                elif "GPT-2" == model_name:
                    list_result.append(self.GPT2(text=text))

                elif "T5_base" == model_name:
                    list_result.append(self.T5(text=text, model_name="t5-base"))

                elif "T5_News" == model_name:
                    list_result.append(
                        self.T5(
                            text=text,
                            model_name="mrm8488/t5-base-finetuned-summarize-news",
                        )
                    )

                elif "Pegasus_large" == model_name:
                    list_result.append(
                        self.Pegasus(text=text, model_name="google/pegasus-large")
                    )

                elif "Pegasus_X_large" == model_name:
                    list_result.append(
                        self.Pegasus(text=text, model_name="google/pegasus-x-large")
                    )

                elif "Pegasus_cnn_daily" == model_name:
                    list_result.append(
                        self.Pegasus(
                            text=text, model_name="google/pegasus-cnn_dailymail"
                        )
                    )

                elif "Pegasus_xsum" == model_name:
                    list_result.append(
                        self.Pegasus(text=text, model_name="google/pegasus-xsum")
                    )

                elif "Pegasus_big_bird_large_arxiv" == model_name:
                    list_result.append(
                        self.Pegasus(
                            text=text, model_name="google/bigbird-pegasus-large-arxiv"
                        )
                    )

                elif "Bart_large" == model_name:
                    list_result.append(
                        self.Bart(text=text, model_name="facebook/bart-large")
                    )
                elif "Bart_cnn_daily" == model_name:
                    list_result.append(
                        self.Bart(text=text, model_name="facebook/bart-large-cnn")
                    )
                elif "Bart_xsum" == model_name:
                    list_result.append(
                        self.Bart(text=text, model_name="facebook/bart-large-xsum")
                    )

                else:
                    raise Exception(f"{model_name} 対象のモデル名が存在しません。")
            array_summarize.append(list_result)  # type: ignore
        array_summarize = np.vstack(array_summarize)
        df_summarize = pd.DataFrame(
            data=array_summarize[1:], columns=array_summarize[0]
        )
        return df_summarize


if __name__ == "__main__":
    cfg_path = "/home/jumpei.uchida/develop/kaggle_1080ti_1_1/feedback-prize-english-language-learning/workspace/uchida/config.yaml"
    summarizer = SummaryModel(cfg_path=cfg_path)
    df_summary = summarizer.summarize()
    print("debug")
