import warnings

from bertopic import BERTopic
from omegaconf import OmegaConf

warnings.simplefilter("ignore")

from pathlib import Path

import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

path_summary = Path(
    "/home/jumpei.uchida/develop/kaggle_1080ti_1_1/feedback-prize-english-language-learning/workspace/uchida/result/Summarized_train_data_46637.csv"
)
docs = pd.read_csv(path_summary)["Pegasus_xsum"].to_list()
topic_model = BERTopic()
topics, probs = topic_model.fit_transform(docs)


class TopicModel:
    def __init__(self, path_cfg: str):
        self.cfg = OmegaConf.load(Path(path_cfg))
        self.docs: list = self._load_data()
        self.topic_model = None
        self.topic = None
        self.probs = None

    def _load_data(self) -> list:
        docs = pd.read_csv(Path(self.cfg.dataset.data_root_path)).iloc[:, 1].to_list()
        return docs

    def _cluster_model(self):
        cluster_model = KMeans(
            n_clusters=self.cfg.cluster_params.n_cluster,
            random_state=self.cfg.cluster_params.seed,
        )
        return cluster_model

    def _sentence_model(self):
        sentence_model = SentenceTransformer(self.cfg.params.sentence_model)
        return sentence_model

    def build_model(self) -> None:
        topic_model = BERTopic(
            language=self.cfg.params.language,
            verbose=self.cfg.params.verbose,
            hdbscan_model=self._cluster_model(),  # type: ignore
            embedding_model=self._sentence_model(),
        )
        self.topic_model = topic_model

    def fit(self) -> None:
        topic, probs = self.topic_model.fit_transform(self.docs)
        self.topic = topic
        self.probs = probs
