import os
import pandas as pd
from text_unidecode import unidecode
from typing import Dict, List, Tuple
import codecs
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold


def get_essay(data_path, essay_id):
    essay_path = os.path.join(data_path, f"{essay_id}.txt")
    essay_text = open(essay_path, 'r').read()
    return essay_text


def replace_encoding_with_utf8(error: UnicodeError) -> Tuple[bytes, int]:
    return error.object[error.start: error.end].encode("utf-8"), error.end


def replace_decoding_with_cp1252(error: UnicodeError) -> Tuple[str, int]:
    return error.object[error.start: error.end].decode("cp1252"), error.end


# Register the encoding and decoding error handlers for `utf-8` and `cp1252`.
codecs.register_error("replace_encoding_with_utf8", replace_encoding_with_utf8)
codecs.register_error("replace_decoding_with_cp1252", replace_decoding_with_cp1252)


def resolve_encodings_and_normalize(text: str) -> str:
    """Resolve the encoding problems and normalize the abnormal characters."""
    text = (
        text.encode("raw_unicode_escape")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
        .encode("cp1252", errors="replace_encoding_with_utf8")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
    )
    text = unidecode(text)
    return text


def data_maker(df, cfg):
    data_path = os.path.join(cfg.data_root_path, cfg.dataset_type)
    df['essay_text'] = df['essay_id'].apply(lambda id_: get_essay(data_path, id_))

    # よく分からないエンコーディング
    df['discourse_text'] = df['discourse_text'].apply(lambda x: resolve_encodings_and_normalize(x))
    df['essay_text'] = df['essay_text'].apply(lambda x: resolve_encodings_and_normalize(x))

    if cfg.dataset_type == "train":
        encoder = LabelEncoder()
        df['discourse_effectiveness'] = encoder.fit_transform(df['discourse_effectiveness'])
        if cfg.debug:
            df = df.iloc[:128]
    return df


def _fix_sentences(train: pd.DataFrame):
    # 文章の欠損をなおす
    train['discourse_text'][293] = 'Cl' + train['discourse_text'][293]
    train['discourse_text'][790] = 'T' + train['discourse_text'][790]
    train['discourse_text'][879] = 'I' + train['discourse_text'][879]
    train['discourse_text'][2828] = 'w' + train['discourse_text'][2828]
    train['discourse_text'][4793] = 'i' + train['discourse_text'][4793]
    train['discourse_text'][8093] = 'I' + train['discourse_text'][8093]
    train['discourse_text'][9202] = 'l' + train['discourse_text'][9202]
    train['discourse_text'][9790] = 'I' + train['discourse_text'][9790]
    train['discourse_text'][14054] = 'i' + train['discourse_text'][14054]
    train['discourse_text'][14387] = 's' + train['discourse_text'][14387]
    train['discourse_text'][15188] = 'i' + train['discourse_text'][15188]
    train['discourse_text'][15678] = 'I' + train['discourse_text'][15678]
    train['discourse_text'][16065] = 'f' + train['discourse_text'][16065]
    train['discourse_text'][16084] = 'I' + train['discourse_text'][16084]
    train['discourse_text'][16255] = 'T' + train['discourse_text'][16255]
    train['discourse_text'][17096] = 'I' + train['discourse_text'][17096]
    train['discourse_text'][17261] = 't' + train['discourse_text'][17261]
    train['discourse_text'][18691] = 'I' + train['discourse_text'][18691]
    train['discourse_text'][19967] = 't' + train['discourse_text'][19967]
    train['discourse_text'][20186] = 'b' + train['discourse_text'][20186]
    train['discourse_text'][20264] = 'I' + train['discourse_text'][20264]
    train['discourse_text'][20421] = 'i' + train['discourse_text'][20421]
    train['discourse_text'][20870] = 'h' + train['discourse_text'][20870]
    train['discourse_text'][22064] = 't' + train['discourse_text'][22064]
    train['discourse_text'][22793] = 'I' + train['discourse_text'][22793]
    train['discourse_text'][22962] = 'W' + train['discourse_text'][22962]
    train['discourse_text'][23990] = 'f' + train['discourse_text'][23990]
    train['discourse_text'][24085] = 'w' + train['discourse_text'][24085]
    train['discourse_text'][25330] = 'a' + train['discourse_text'][25330]
    train['discourse_text'][25446] = 'i' + train['discourse_text'][25446]
    train['discourse_text'][25667] = 'S' + train['discourse_text'][25667]
    train['discourse_text'][25869] = 'I' + train['discourse_text'][25869]
    train['discourse_text'][26172] = 'i' + train['discourse_text'][26172]
    train['discourse_text'][26284] = 'I' + train['discourse_text'][26284]
    train['discourse_text'][26289] = 't' + train['discourse_text'][26289]
    train['discourse_text'][26322] = 't' + train['discourse_text'][26322]
    train['discourse_text'][26511] = 't' + train['discourse_text'][26511]
    train['discourse_text'][27763] = 'I' + train['discourse_text'][27763]
    train['discourse_text'][28262] = 'P' + train['discourse_text'][28262]
    train['discourse_text'][29164] = 'bu' + train['discourse_text'][29164]
    train['discourse_text'][29519] = 'e' + train['discourse_text'][29519]
    train['discourse_text'][29532] = 't' + train['discourse_text'][29532]
    train['discourse_text'][29571] = 'A' + train['discourse_text'][29571]
    train['discourse_text'][29621] = 't' + train['discourse_text'][29621]
    train['discourse_text'][30791] = 'E' + train['discourse_text'][30791]
    train['discourse_text'][30799] = 'T' + train['discourse_text'][30799]
    train['discourse_text'][31519] = 't' + train['discourse_text'][31519]
    train['discourse_text'][31597] = 't' + train['discourse_text'][31597]
    train['discourse_text'][31992] = 'T' + train['discourse_text'][31992]
    train['discourse_text'][32086] = 'I' + train['discourse_text'][32086]
    train['discourse_text'][32204] = 'c' + train['discourse_text'][32204]
    train['discourse_text'][32341] = 'becaus' + train['discourse_text'][32341]
    train['discourse_text'][33246] = 'A' + train['discourse_text'][33246]
    train['discourse_text'][33819] = 'W' + train['discourse_text'][33819]
    train['discourse_text'][34023] = 'i' + train['discourse_text'][34023]
    train['discourse_text'][35467] = 'b' + train['discourse_text'][35467]
    train['discourse_text'][35902] = 'i' + train['discourse_text'][35902]
    return train


def _drop_duplicates(train: pd.DataFrame):
    # 同じessay_id,discourse_text,discourse_typeのデータの削除
    train = train.drop(index=35969)
    train = train.drop(index=31757)
    train = train.drop(index=35493)
    return train


def get_df(cfg, fix_sentences=True, drop_duplicates=True):
    train_path = os.path.join(cfg.data_root_path, f"{cfg.dataset_type}.csv")
    train = pd.read_csv(train_path)

    if fix_sentences:
        train = _fix_sentences(train)
    if drop_duplicates:
        train = _drop_duplicates(train)

    df_train = data_maker(train, cfg).reset_index(drop=True)

    if cfg.dataset_type == "train":
        gf = GroupKFold(n_splits=cfg.n_fold)

        # kfoldのカラムを設定(バリデーションのfoldを入力)
        for fold, (train, val) in enumerate(gf.split(df_train, groups=df_train['essay_id'])):
            df_train.loc[val, "kfold"] = fold
        df_train['kfold'] = df_train['kfold'].astype(int)

    return df_train
