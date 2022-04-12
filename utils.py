from config import MainConfig
import os
import shutil
from model import RobertaNER
from transformers import RobertaTokenizer
import torch
import json
import nltk


def build_output_package(trained_model: RobertaNER, config: MainConfig):
    nltk_path = os.path.join(config.SCORE.PACKAGE_FOLDER, "NLTK")
    os.mkdir(nltk_path)

    nltk.download('punkt', download_dir=nltk_path)
    nltk.download('averaged_perceptron_tagger', download_dir=nltk_path)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    shutil.copyfile(config.POS.POS_TAGS_DICT_FILEPATH, os.path.join(config.SCORE.PACKAGE_FOLDER, os.path.basename(config.POS.POS_TAGS_DICT_FILEPATH)) )
    shutil.copyfile(config.NER.NER_TAGS_DICT_FILEPATH, os.path.join(config.SCORE.PACKAGE_FOLDER, os.path.basename(config.NER.NER_TAGS_DICT_FILEPATH)))

    mini_config = {
        "config.MODEL.POS_EMBEDDINGS_SIZE": config.MODEL.POS_EMBEDDINGS_SIZE,
        "config.MODEL.DEFAULT_SENTENCE_LEN": config.MODEL.DEFAULT_SENTENCE_LEN,
        "config.POS.UNK_POS_TAG" : config.POS.UNK_POS_TAG,
        "config.POS.PAD_POS_TAG" : config.POS.PAD_POS_TAG
    }
    tokenizer.save_pretrained("model_package/tokenizer")

    with open(os.path.join(config.SCORE.PACKAGE_FOLDER, "config.json"), 'w') as file:
        json.dump( mini_config, file)
    torch.save(trained_model.state_dict(), os.path.join(config.SCORE.PACKAGE_FOLDER, 'model.pt'))
    print("CREATED PACKAGE")