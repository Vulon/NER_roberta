import warnings
warnings.filterwarnings('ignore')
from transformers import RobertaTokenizer
import pandas as pd
import pickle
import json
from collections import Counter
import nltk
import os, sys
import dvc.api
sys.path.append(os.environ["DVC_ROOT"])
from ner_roberta.training.dataset import NerDataset



def create_ner_tags_dict(dataframe : pd.DataFrame, ner_tags_config: dict):
    tags_dict = set()
    for tag_line in dataframe['Tag']:
        tags_list = tag_line.replace('[', '').replace(']', '').replace("'", "")
        tags_list = [word.strip() for word in tags_list.split(',')]
        tags_list = [ner_tags_config["NER_TAGS_SUBSTITUTION"].get(word, word) for word in tags_list]
        tags_dict = tags_dict.union(tags_list)
    tags_dict = {word: i + 1 for i, word in enumerate(tags_dict)}
    tags_dict[ner_tags_config["START_TAG"]] = len(tags_dict) + 1
    tags_dict[ner_tags_config["STOP_TAG"]] = len(tags_dict) + 1
    tags_dict[ner_tags_config["UNK_TAG"]] = len(tags_dict) + 1
    tags_dict[ner_tags_config["CLS_TAG"]] = 0
    tags_dict = {key: tags_dict[key] for key in sorted(tags_dict, key=lambda x: tags_dict[x])}
    return tags_dict

def split_data(dataframe : pd.DataFrame, train_fracture, val_vs_test_fracture):
    border = int(dataframe.shape[0] * train_fracture)
    train_df = dataframe[ : border ]
    test_df = dataframe[ border : ]

    border = int(test_df.shape[0] * val_vs_test_fracture)
    val_df = test_df[ : border]
    test_df = test_df[border : ]
    return train_df, val_df, test_df


def create_pos_tags_dict(dataframe : pd.DataFrame, pos_tags_conf: dict):
    all_pos_tags = []
    for index, row in dataframe.iterrows():
        tokens = nltk.tokenize.word_tokenize(row['Sentence'])
        pos_tags = [pos_tags_conf["POS_TAGS_SUBSTITUTION"].get(item[1], item[1]) for item in nltk.pos_tag(tokens)]
        all_pos_tags.extend(pos_tags)

    all_pos_tags = Counter(all_pos_tags)
    all_pos_tags = [item[0] for item in all_pos_tags.most_common() if item[1] > pos_tags_conf["MINIMUM_POS_TAG_COUNT"]]
    pos_tags_dict = { tag : i for i, tag in enumerate(all_pos_tags) }
    pos_tags_dict[pos_tags_conf["UNK_POS_TAG"]] = len(pos_tags_dict)
    pos_tags_dict[pos_tags_conf["PAD_POS_TAG"]] = len(pos_tags_dict)
    return pos_tags_dict


print("Entered data preparation")
if __name__ == '__main__':
    print("Started preparation")
    params = dvc.api.params_show()

    df = pd.read_csv(params["DATA"]["RAW_INPUT_FILEPATH"]).sample(frac=1, random_state=params["RANDOM_SEED"])
    tags_dict = create_ner_tags_dict(df, params["tags"]["NER"])

    train_df, val_df, test_df = split_data(df, params["DATA"]["TRAIN_SAMPLE_FRACTURE"], params["DATA"]["VAL_VS_TEST_FRACTURE"])
    pos_tags_dict = create_pos_tags_dict(train_df, params["tags"]["POS"])

    with open(params['tags']["POS"]["POS_TAGS_DICT_FILEPATH"], 'w') as file:
        json.dump(pos_tags_dict, file)

    with open(params['tags']["NER"]["NER_TAGS_DICT_FILEPATH"], 'w') as file:
        json.dump(tags_dict, file)

    tokenizer = RobertaTokenizer.from_pretrained(params["MODEL"]["TOKENIZER_NAME"])
    train_dataset = NerDataset(train_df, tags_dict, pos_tags_dict, tokenizer, params)
    val_dataset = NerDataset(val_df, tags_dict, pos_tags_dict, tokenizer, params)
    test_dataset = NerDataset(test_df, tags_dict, pos_tags_dict, tokenizer, params)

    with open(params["DATA"]["TRAIN_DATASET_PATH"], 'wb') as file:
        pickle.dump(train_dataset, file)

    with open(params["DATA"]["VAL_DATASET_PATH"], 'wb') as file:
        pickle.dump(val_dataset, file)

    with open(params["DATA"]["TEST_DATASET_PATH"], 'wb') as file:
        pickle.dump(test_dataset, file)