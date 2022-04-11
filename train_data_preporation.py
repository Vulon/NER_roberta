import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from transformers import RobertaTokenizer, RobertaForMaskedLM
import re
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import numpy as np
import pandas as pd
import pickle
import tqdm
import json
from transformers import Trainer, TrainingArguments
from transformers.file_utils import PaddingStrategy
from sklearn.metrics import recall_score, precision_score, f1_score
from collections import Counter

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from config import get_config
config = get_config()
from dataset import NerDataset


def create_ner_tags_dict(dataframe : pd.DataFrame):
    tags_dict = set()
    for tag_line in dataframe['Tag']:
        tags_list = tag_line.replace('[', '').replace(']', '').replace("'", "")
        tags_list = [word.strip() for word in tags_list.split(',')]
        tags_list = [config.NER.NER_TAGS_SUBSTITUTION.get(word, word) for word in tags_list]
        tags_dict = tags_dict.union(tags_list)
    tags_dict = {word: i + 1 for i, word in enumerate(tags_dict)}
    tags_dict[config.NER.START_TAG] = len(tags_dict) + 1
    tags_dict[config.NER.STOP_TAG] = len(tags_dict) + 1
    tags_dict[config.NER.UNK_TAG] = len(tags_dict) + 1
    tags_dict[config.NER.CLS_TAG] = 0
    tags_dict = {key: tags_dict[key] for key in sorted(tags_dict, key=lambda x: tags_dict[x])}
    return tags_dict

def split_data(dataframe : pd.DataFrame):
    border = int(dataframe.shape[0] * config.TRAIN.TRAIN_SAMPLE_FRACTURE)
    train_df = dataframe[ : border ]
    test_df = dataframe[ border : ]

    border = int(test_df.shape[0] * config.TRAIN.VAL_VS_TEST_FRACTURE)
    val_df = test_df[ : border]
    test_df = test_df[border : ]
    return train_df, val_df, test_df


def create_pos_tags_dict(dataframe : pd.DataFrame):
    all_pos_tags = []
    for index, row in dataframe.iterrows():
        tokens = nltk.tokenize.word_tokenize(row['Sentence'])
        pos_tags = [config.POS.POS_TAGS_SUBSTITUTION.get(item[1], item[1]) for item in nltk.pos_tag(tokens)]
        all_pos_tags.extend(pos_tags)

    all_pos_tags = Counter(all_pos_tags)
    all_pos_tags = [item[0] for item in all_pos_tags.most_common() if item[1] > config.POS.MINIMUM_POS_TAG_COUNT]
    pos_tags_dict = { tag : i for i, tag in enumerate(all_pos_tags) }
    pos_tags_dict[config.POS.UNK_POS_TAG] = len(pos_tags_dict)
    pos_tags_dict[config.POS.PAD_POS_TAG] = len(pos_tags_dict)
    return pos_tags_dict


if __name__ == '__main__':
    df = pd.read_csv(config.TRAIN.RAW_INPUT_FILEPATH).sample(frac=1, random_state=42)
    tags_dict = create_ner_tags_dict(df)
    train_df, val_df, test_df = split_data(df)
    pos_tags_dict = create_pos_tags_dict(train_df)

    with open(config.POS.POS_TAGS_DICT_FILEPATH, 'w') as file:
        json.dump(pos_tags_dict, file)

    with open(config.NER.NER_TAGS_DICT_FILEPATH, 'w') as file:
        json.dump(tags_dict, file)

    tokenizer = RobertaTokenizer.from_pretrained(config.MODEL.TOKENIZER_NAME)
    train_dataset = NerDataset(train_df, tags_dict, pos_tags_dict, tokenizer, config)
    val_dataset = NerDataset(val_df, tags_dict, pos_tags_dict, tokenizer, config)
    test_dataset = NerDataset(test_df, tags_dict, pos_tags_dict, tokenizer, config)

    with open(config.TRAIN.TRAIN_DATASET_PATH, 'wb') as file:
        pickle.dump(train_dataset, file)

    with open(config.TRAIN.VAL_DATASET_PATH, 'wb') as file:
        pickle.dump(val_dataset, file)

    with open(config.TRAIN.TEST_DATASET_PATH, 'wb') as file:
        pickle.dump(test_dataset, file)